import math
import random
from functools import partial
from typing import Optional, Tuple, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_

try:
    from xformers.ops import memory_efficient_attention
    use_xformer = True
except ImportError:
    use_xformer = False


def interleave_tokens(seq1, seq2):
    """Interleave two sequences (following RandAR)"""
    result = torch.zeros_like(torch.cat((seq1, seq2), dim=1))
    result[:, ::2] = seq1
    result[:, 1::2] = seq2
    return result


def cosine_schedule(step, total_steps, total_tokens):
    """
    Cosine schedule for RandAR token generation.
    
    Args:
        step: Current step (0 to total_steps-1)
        total_steps: Total number of generation steps
        total_tokens: Total number of tokens to generate
    
    Returns:
        Number of tokens to unmask at this step
    """
    if step >= total_steps:
        return 0
    
    # Cosine schedule: faster at the beginning, slower at the end
    # This follows the insight that early tokens are easier to predict
    ratio = step / total_steps
    # Use cosine annealing from pi to 0
    cosine_ratio = 0.5 * (1 + math.cos(math.pi * ratio))
    
    # Calculate cumulative tokens generated up to this step
    tokens_so_far = int(total_tokens * (1 - cosine_ratio))
    
    # Calculate tokens generated up to previous step
    if step > 0:
        prev_ratio = (step - 1) / total_steps
        prev_cosine_ratio = 0.5 * (1 + math.cos(math.pi * prev_ratio))
        prev_tokens = int(total_tokens * (1 - prev_cosine_ratio))
    else:
        prev_tokens = 0
    
    # Tokens to generate at this step
    tokens_this_step = tokens_so_far - prev_tokens
    
    # Ensure we generate at least 1 token per step and don't exceed total
    tokens_this_step = max(1, tokens_this_step)
    remaining_tokens = total_tokens - prev_tokens
    tokens_this_step = min(tokens_this_step, remaining_tokens)
    
    return tokens_this_step


def calculate_num_query_tokens_for_parallel_decoding(
    cur_inference_step, num_inference_steps, total_tokens, 
    query_token_idx_cur_step, num_query_token_cur_step):
    """
    Calculate number of query tokens for current step in parallel decoding.
    
    Args:
        cur_inference_step: Current inference step (0-indexed)
        num_inference_steps: Total number of inference steps
        total_tokens: Total number of tokens to generate
        query_token_idx_cur_step: Current query token index
        num_query_token_cur_step: Number of query tokens in previous step (ignored for step 0)
    
    Returns:
        Number of query tokens for current step
    """
    if num_inference_steps == -1 or num_inference_steps >= total_tokens:
        # No parallel decoding, one token at a time
        return 1
    
    # For first step, calculate initial tokens
    if cur_inference_step == 0:
        # Distribute tokens evenly across steps, with any remainder in early steps
        base_tokens = total_tokens // num_inference_steps
        remainder = total_tokens % num_inference_steps
        return base_tokens + (1 if remainder > 0 else 0)
    
    # Calculate remaining tokens and steps
    remaining_tokens = total_tokens - query_token_idx_cur_step
    remaining_steps = num_inference_steps - cur_inference_step
    
    if remaining_steps <= 0 or remaining_tokens <= 0:
        return 0
    
    # Distribute remaining tokens across remaining steps
    # Use ceiling division to ensure we don't underestimate
    tokens_per_step = (remaining_tokens + remaining_steps - 1) // remaining_steps
    
    # Don't exceed remaining tokens
    return min(tokens_per_step, remaining_tokens)


def get_cosine_schedule_indices(total_tokens, total_steps):
    """
    Pre-compute token indices for each step using cosine schedule.
    
    Args:
        total_tokens: Total number of tokens
        total_steps: Total number of generation steps
    
    Returns:
        List of (start_idx, end_idx) tuples for each step
    """
    # Pre-compute all boundaries using cosine schedule
    boundaries = [0]
    
    for step in range(1, total_steps + 1):
        ratio = step / total_steps
        # Use cosine annealing from pi to 0
        cosine_ratio = 0.5 * (1 + math.cos(math.pi * ratio))
        # Calculate cumulative tokens generated up to this step
        tokens_so_far = int(total_tokens * (1 - cosine_ratio))
        boundaries.append(tokens_so_far)
    
    # Ensure the last boundary is exactly total_tokens
    boundaries[-1] = total_tokens
    
    # Create indices from boundaries
    indices = []
    for i in range(total_steps):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        # Ensure at least 1 token per step if possible
        if end_idx == start_idx and start_idx < total_tokens:
            end_idx = min(start_idx + 1, total_tokens)
            # Adjust next boundaries if needed
            for j in range(i + 1, len(boundaries)):
                if boundaries[j] < end_idx:
                    boundaries[j] = end_idx
        indices.append((start_idx, end_idx))
    
    return indices


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class ScaleAR(nn.Module):
    def __init__(
        self, vae_local,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, token_dropout_p=0.05,
        attn_l2_norm=False,
        randar_scale=4,  # Starting scale for RandAR generation (e.g., 4 means 4x4)
        randar_mode='maskgit',  # 'randar' or 'maskgit' mode
        patch_nums=(4, 5, 6, 8, 10, 13, 16),  # Default inference schedule
        flash_if_available=True, fused_if_available=True,
        enable_var_kv=False,  # Enable KV caching for VAR scales during training
        share_randar_var_pos_embed=False,  # Share positional embeddings between RandAR and VAR steps
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        # RandAR specific
        self.randar_scale = randar_scale  # Starting scale for RandAR generation
        self.randar_mode = randar_mode
        self.randar_steps = self.randar_scale ** 2  # Number of tokens at randar_scale
        self.randar_tokens = self.randar_scale ** 2  # Same as randar_steps
        self.enable_var_kv = enable_var_kv  # Enable KV caching for VAR scales
        self.share_randar_var_pos_embed = share_randar_var_pos_embed  # Share pos embed between RandAR and VAR
        
        # VAR specific (for scales after 4x4)
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        # Learnable mask token for RandAR/MaskGIT training
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.C) * init_std)
        
        # RandAR position embeddings
        if self.randar_mode == 'randar':
            # RandAR mode: use position instruction tokens (following RandAR paper)
            self.pos_instruct_embeddings = nn.Parameter(torch.randn(1, self.C) * init_std)
        elif not self.share_randar_var_pos_embed:
            # MaskGIT/MAR mode: use learned position embeddings (only if not sharing with VAR)
            self.randar_pos_embed = nn.Parameter(torch.empty(1, self.randar_tokens, self.C))
            nn.init.trunc_normal_(self.randar_pos_embed.data, mean=0, std=init_std)
        
        # 3. absolute position embedding (for VAR upsampling)
        pos_1LC = self.generate_2d_rotary_position_embedding(h=32, w=32, d = self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        # 5. attention mask for VAR (causal for upsampling phase)
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.tok_dropout = nn.Dropout(token_dropout_p)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))

    def get_num_params(self, ):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h, cond_BD))

    def generate_2d_rotary_position_embedding(self, h, w, d):
        assert d % 2 == 0, "Dimension d must be an even number."
        
        pos_encoding = torch.zeros(h, w, d)
        y_coords = torch.arange(h, dtype=torch.float32)
        x_coords = torch.arange(w, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
        
        for i in range(h):
            for j in range(w):
                pos_encoding[i, j, 0::2] = torch.sin(y_grid[i, j] * div_term)
                pos_encoding[i, j, 1::2] = torch.cos(x_grid[i, j] * div_term)
        
        return pos_encoding.unsqueeze(0).permute(0,3,1,2)

    def generate_random_order(self, B, device):
        """Generate random order for RandAR generation"""
        orders = []
        for _ in range(B):
            order = torch.randperm(self.randar_tokens, device=device)
            orders.append(order)
        return torch.stack(orders)  # B x randar_tokens
    
    def get_position_instruction_tokens(self, token_order):
        """Get position instruction tokens with rotary embeddings applied (RandAR mode)"""
        # Reshape position instruction embeddings to match head dimensions
        position_instruct_tokens = self.pos_instruct_embeddings.view(1, 1, self.num_heads, self.C // self.num_heads)
        position_instruct_tokens = position_instruct_tokens.repeat(token_order.shape[0], self.randar_tokens, 1, 1)
        
        # Apply 2D rotary embedding based on token order
        # Get the 4x4 positions from the full position embeddings
        pos_4x4 = F.interpolate(self.pos_1LC, size=(4, 4), mode='area')
        pos_4x4 = pos_4x4.reshape(1, self.C, -1).permute(0, 2, 1)  # 1 x 16 x C
        
        # Apply rotary based on order
        position_instruct_tokens = position_instruct_tokens.view(token_order.shape[0], self.randar_tokens, self.C)
        for b in range(token_order.shape[0]):
            position_instruct_tokens[b] = position_instruct_tokens[b][token_order[b].argsort()]
        
        return position_instruct_tokens
    
    def create_var_attention_mask(self, total_len, device):
        """Create VAR-only attention mask for student forcing"""
        # Pad to multiple of 8 for memory efficient attention
        padded_len = ((total_len + 7) // 8) * 8
        attn_bias_padded = torch.zeros(1, 1, padded_len, padded_len, device=device)
        attn_bias_padded[:, :, :total_len, total_len:] = -torch.inf
        attn_bias_padded[:, :, total_len:, :total_len] = -torch.inf

        # VAR attention: bidirectional within scales, no attention across scales
        cur_pos = 1  # Start after class token
        for pn in self.patch_nums:
            scale_size = pn ** 2
            scale_end = cur_pos + scale_size
            
            # Block attention to previous scales
            if cur_pos > 1:
                attn_bias_padded[0, 0, cur_pos:scale_end, 1:cur_pos] = -torch.inf
            
            # Block attention to future scales
            if scale_end < total_len:
                attn_bias_padded[0, 0, cur_pos:scale_end, scale_end:total_len] = -torch.inf
            
            cur_pos = scale_end
        
        return attn_bias_padded
    
    def process_randar_tokens(self, x_randar, orders, num_masked, cond_BD, B, device):
        """Process randar_scale tokens based on the mode (RandAR or MaskGIT)"""
        if self.randar_mode == 'randar':
            # RandAR mode: interleave position instruction tokens with masked tokens
            x_randar_ordered = torch.gather(x_randar, 1, orders.unsqueeze(-1).expand(-1, -1, x_randar.shape[-1]))
            x_randar_embed = self.word_embed(x_randar_ordered)
            
            # Apply mask after embedding - batched
            mask_positions = orders < num_masked  # B x randar_tokens
            x_randar_embed[mask_positions] = self.mask_token.squeeze(0)
            
            # Get position instruction tokens and interleave
            position_instruction_tokens = self.get_position_instruction_tokens(orders)
            x_randar_interleaved = interleave_tokens(position_instruction_tokens, x_randar_embed)
            
            # Concatenate class embedding at the front
            return torch.cat([cond_BD.unsqueeze(1), x_randar_interleaved], dim=1)
        else:
            # MaskGIT/MAR mode: use learned position embeddings
            x_randar_embed = self.word_embed(x_randar)
            
            # Apply mask after embedding - batched
            mask_positions = orders < num_masked  # B x randar_tokens
            x_randar_embed[mask_positions] = self.mask_token.squeeze(0)
            
            if self.share_randar_var_pos_embed:
                # Use the first randar_tokens from pos_1LC (randar_scale positions)
                # Note: pos_1LC needs to be prepared before this function is called
                pos_randar = F.interpolate(self.pos_1LC, size=(self.randar_scale, self.randar_scale), mode='area')
                pos_randar = pos_randar.reshape(1, self.C, -1).permute(0, 2, 1).expand(B, -1, -1)
                x_randar_embed = x_randar_embed + pos_randar
            else:
                x_randar_embed = x_randar_embed + self.randar_pos_embed.expand(B, -1, -1)
            
            # Concatenate class embedding at the front
            return torch.cat([cond_BD.unsqueeze(1), x_randar_embed], dim=1)
    
    def create_attention_mask(self, total_len, randar_end, device):
        """Create hybrid attention mask for RandAR/MaskGIT + VAR"""
        # Pad to multiple of 8 for memory efficient attention
        padded_len = ((total_len + 7) // 8) * 8
        attn_bias_padded = torch.zeros(1, 1, padded_len, padded_len, device=device)
        attn_bias_padded[:, :, :total_len, total_len:] = -torch.inf
        attn_bias_padded[:, :, total_len:, :total_len] = -torch.inf
        
        if self.randar_mode == 'randar':
            # RandAR part: causal attention for the interleaved sequence
            for i in range(randar_end):
                attn_bias_padded[0, 0, i, i+1:] = -torch.inf
        else:
            # MaskGIT mode: bidirectional attention for class token + 4x4
            attn_bias_padded[0, 0, :randar_end, randar_end:] = -torch.inf
        
        # VAR part: bidirectional attention within each scale
        var_start = randar_end
        scale_end = randar_end
        for i, pn in enumerate(self.patch_nums[1:], 1):  # Skip first scale (4x4)
            scale_size = pn ** 2
            scale_end = var_start + scale_size
            
            if self.enable_var_kv:
                # Block attention to RandAR tokens (the 4x4 tokens)
                attn_bias_padded[0, 0, var_start:scale_end, 1:randar_end] = -torch.inf
            else:
                # Block attention to previous scales (including 4x4)
                attn_bias_padded[0, 0, var_start:scale_end, 1:var_start] = -torch.inf
            
            # Block attention to future scales
            if scale_end < total_len:
                attn_bias_padded[0, 0, var_start:scale_end, scale_end:total_len] = -torch.inf
            
            var_start = scale_end
        
        return attn_bias_padded
    
    def extract_and_reorder_logits(self, logits_BLV, orders, randar_end):
        """Extract and reorder logits based on the mode"""
        if self.randar_mode == 'randar':
            # Extract content logits from interleaved positions
            content_logits = logits_BLV[:, 2:randar_end:2, :]
            
            # Re-order content logits back to original spatial order
            unsort = orders.argsort(dim=1)
            idx_expand = unsort.unsqueeze(-1).expand(-1, -1, content_logits.shape[-1])
            content_logits = content_logits.gather(1, idx_expand)
            
            # Remaining VAR logits
            var_logits = logits_BLV[:, randar_end:, :]
        else:
            # MaskGIT/MAR mode
            content_logits = logits_BLV[:, 1:randar_end, :]
            var_logits = logits_BLV[:, randar_end:, :]
        
        return torch.cat((content_logits, var_logits), dim=1)

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, vqvae, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        infer_patch_nums: Optional[List[int]] = None,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, used_llamagen_cfg=False, invalid_ids=None,
        cosine_steps=8, return_intermediate_scales=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, torch.Tensor]]]:
        """
        Hybrid inference: RandAR for 4x4, then VAR for 4x4->16x16
        
        Args:
            cosine_steps: Number of steps for cosine schedule (default 8).
                         Each step uses parallel decoding for its tokens.
            return_intermediate_scales: If True, also returns decoded images for intermediate scales.
        
        Returns:
            If return_intermediate_scales is False: reconstructed image (B, 3, H, W) in [-1, 1]
            If return_intermediate_scales is True: tuple of (final_image, dict of {resolution: decoded_image})
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.pos_1LC.device)

        device = label_B.device
        
        # Double batch for CFG
        cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        vae_embedding = F.normalize(vqvae.quantize.embedding.weight, p=2, dim=-1)
        
        # Set patch nums for inference
        infer_patch_nums = [4, 5, 6, 8, 10, 13, 16] if infer_patch_nums is None else infer_patch_nums
        self.patch_nums = infer_patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        
        # Phase 1: RandAR generation at 4x4
        # Initialize with zeros - in MaskGIT mode, ungenerated positions will be masked with mask_token
        generated_indices = torch.zeros((B, self.randar_tokens), dtype=torch.long, device=device)
        
        # Generate random orders for RandAR
        orders = self.generate_random_order(B, device)
        
        # Prepare position embeddings
        pos_1LC = []
        for hw in infer_patch_nums:
            curr_pos_1LC = F.interpolate(self.pos_1LC, size=(hw, hw), mode='area')
            curr_pos_1LC = curr_pos_1LC.reshape(1, self.pos_1LC.shape[1], -1).permute(0,2,1)
            pos_1LC.append(curr_pos_1LC)
        pos_1LC = torch.cat(pos_1LC, dim = 1)
        
        # Get cosine schedule for RandAR generation
        schedule_indices = get_cosine_schedule_indices(self.randar_tokens, cosine_steps)
        
        # Get position instruction tokens for RandAR mode
        if self.randar_mode == 'randar':
            position_instruction_tokens = self.get_position_instruction_tokens(orders)
            # Enable KV caching for RandAR mode
            for b in self.blocks: b.attn.kv_caching(True)
        
        # RandAR generation loop with cosine schedule
        for step, (start_idx, end_idx) in enumerate(schedule_indices):
            # Tokens to generate in this cosine step
            num_tokens_this_step = end_idx - start_idx
            step_tokens_idx_list = list(range(start_idx, end_idx))
            
            if self.randar_mode == 'randar':
                # RandAR mode: Build interleaved sequence with parallel queries
                
                # 1. Previously generated tokens (if any) - batchized
                if step > 0:
                    # Batchized gathering of previous indices
                    # orders.argsort() gives us the inverse mapping: order_value -> actual_position
                    inverse_orders = orders.argsort(dim=1)
                    prev_positions = inverse_orders[:, :start_idx]  # B x start_idx
                    
                    # Gather previous indices using advanced indexing
                    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(prev_positions)
                    prev_indices = generated_indices[batch_indices, prev_positions]  # B x start_idx
                    
                    prev_token_embeds = vae_embedding[prev_indices]
                    prev_pos_tokens = position_instruction_tokens[:, :start_idx]
                    prev_interleaved = interleave_tokens(prev_pos_tokens, self.word_embed(prev_token_embeds))
                else:
                    prev_interleaved = None
                
                # 2. Current step query tokens (parallel)
                curr_pos_tokens = position_instruction_tokens[:, step_tokens_idx_list]
                
                # 3. Build full sequence - batchized
                if prev_interleaved is not None:
                    # Previous tokens + current queries
                    # Format: [class_token] + [prev_interleaved] + [pos_i, mask_i] for each i in current step
                    
                    # Batchized interleaving of current position tokens with mask placeholders
                    if num_tokens_this_step > 1:
                        # Create mask placeholders for all but the last token
                        mask_placeholders = torch.zeros(B, num_tokens_this_step - 1, self.C, device=device)
                        
                        # Interleave current position tokens with mask placeholders
                        # curr_pos_tokens: B x num_tokens_this_step x C
                        # We want: [pos_0, mask_0, pos_1, mask_1, ..., pos_n-1, mask_n-1, pos_n]
                        curr_tokens_part = curr_pos_tokens[:, :-1]  # All but last
                        last_token = curr_pos_tokens[:, -1:] # Last token
                        
                        # Interleave tokens and masks: B x (num_tokens_this_step-1) x 2 x C
                        interleaved_part = torch.stack([curr_tokens_part, mask_placeholders], dim=2)
                        interleaved_part = interleaved_part.view(B, -1, self.C)  # B x (2*(num_tokens_this_step-1)) x C
                        
                        # Concatenate: prev + interleaved_part + last_token
                        x_randar = torch.cat([prev_interleaved, interleaved_part, last_token], dim=1)
                    else:
                        # Only one token, no interleaving needed
                        x_randar = torch.cat([prev_interleaved, curr_pos_tokens], dim=1)
                else:
                    # First step: class token + position queries with mask placeholders - batchized
                    if num_tokens_this_step > 1:
                        # Create mask placeholders for all but the last token
                        mask_placeholders = torch.zeros(B, num_tokens_this_step - 1, self.C, device=device)
                        
                        # Interleave position tokens with mask placeholders
                        curr_tokens_part = curr_pos_tokens[:, :-1]  # All but last
                        last_token = curr_pos_tokens[:, -1:]  # Last token
                        
                        # Interleave tokens and masks
                        interleaved_part = torch.stack([curr_tokens_part, mask_placeholders], dim=2)
                        interleaved_part = interleaved_part.view(B, -1, self.C)
                        
                        # Concatenate: class + interleaved_part + last_token
                        x_randar = torch.cat([interleaved_part, last_token], dim=1)
                    else:
                        # Only one token, no interleaving needed
                        x_randar = curr_pos_tokens
            else:
                # MaskGIT mode: Standard masked tokens
                # Create token embeddings - start with ALL positions as mask tokens
                token_embeds_embedded = self.mask_token.expand(B, self.randar_tokens, -1).clone()

                # Fill in already generated tokens (only if we've generated some) - batchized
                if step > 0:
                    generated_mask = orders < start_idx  # B x randar_tokens

                    # Batchized token filling using advanced indexing
                    if generated_mask.any():
                        # Direct masking is simpler and more efficient
                        generated_token_indices = generated_indices[generated_mask]

                        # Get embeddings and apply word embedding
                        token_embeds = vae_embedding[generated_token_indices]
                        embedded_tokens = self.word_embed(token_embeds)

                        # Replace mask tokens with generated tokens - ensure dtype match
                        token_embeds_embedded[generated_mask] = embedded_tokens.to(token_embeds_embedded.dtype)
                
                if self.share_randar_var_pos_embed:
                    # Use the first 16 tokens from pos_1LC (4x4 positions)
                    x_randar = token_embeds_embedded + pos_1LC[:, :self.randar_tokens]
                else:
                    x_randar = token_embeds_embedded + self.randar_pos_embed
            
            # Double for CFG
            x_combined = x_randar.repeat(2, 1, 1)
            # Concatenate class token at the front
            x_combined = torch.cat([cond_BD.unsqueeze(1), x_combined], dim=1)
            
            # Create attention mask
            if self.randar_mode == 'randar':
                # Causal attention for RandAR
                seq_len = x_combined.shape[1]
                padded_len = ((seq_len + 7) // 8) * 8
                attn_bias_padded = torch.zeros(1, 1, padded_len, padded_len, device=device)
                for i in range(seq_len):
                    attn_bias_padded[0, 0, i, i+1:seq_len] = -torch.inf
                attn_bias = attn_bias_padded
            else:
                # Bidirectional attention for MaskGIT
                seq_len = 1 + self.randar_tokens  # class token + randar tokens
                padded_len = ((seq_len + 7) // 8) * 8
                attn_bias_padded = torch.zeros(1, 1, padded_len, padded_len, device=device)
                attn_bias_padded[:, :, :seq_len, seq_len:] = -torch.inf
                attn_bias_padded[:, :, seq_len:, :seq_len] = -torch.inf
                attn_bias = attn_bias_padded
            
            # Forward through transformer
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = x_combined
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            
            # Get logits and extract for current positions
            logits_BlV = self.get_logits(x, cond_BD)
            
            if self.randar_mode == 'randar':
                # Extract logits for content positions (odd indices after class token and previous content)
                if start_idx > 0:
                    # Account for class token at position 0
                    # After class token + prev_interleaved, content positions are at odd indices
                    content_start = 1 + 2 * start_idx + 1
                    content_positions = list(range(content_start, content_start + 2 * num_tokens_this_step, 2))
                else:
                    # First step: class token at 0, then content at odd positions starting from 2
                    content_positions = list(range(2, 1 + 2 * num_tokens_this_step, 2))
                logits_BlV = logits_BlV[:, content_positions, :]
            else:
                # MaskGIT: skip class token at position 0, then use token positions
                logits_BlV = logits_BlV[:, 1:, :]
            
            if invalid_ids is not None:
                logits_BlV[:, :, invalid_ids] = -100.0
            
            # Apply CFG
            if not used_llamagen_cfg:
                ratio = step / cosine_steps
                t = cfg * 0.5 * (1 + ratio)
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            else:
                cond_logits, uncond_logits = torch.split(logits_BlV, B, dim=0)
                logits_BlV = uncond_logits + (cond_logits - uncond_logits) * cfg * 0.5
            
            # Sample tokens for all positions in this step
            sampled = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            # Get the actual positions for tokens we're generating this step
            # Update generated_indices using the sampled values
            # sampled has shape B x num_tokens_this_step
            generated_indices[orders < end_idx] = sampled[orders < end_idx]
        
        # Phase 2: VAR generation for subsequent scales
        if self.enable_var_kv:
            for b in self.blocks:
                b.attn.cached_k = None
                b.attn.cached_v = None
                b.attn.kv_caching(True)
        else:
            for b in self.blocks:
                b.attn.cached_k = None
                b.attn.cached_v = None
                b.attn.kv_caching(False)
        
        # Generate progressively through all scales after 4x4
        all_indices = [generated_indices]  # Start with 4x4 indices
        
        # Progressive generation through scales
        for si in range(1, len(infer_patch_nums)):
            curr_hw = infer_patch_nums[si]
            prev_hw = infer_patch_nums[si-1]
            
            # Get embeddings from previous scale
            prev_indices = all_indices[-1]
            prev_quant = vae_embedding[prev_indices].reshape(B, prev_hw, prev_hw, -1).permute(0,3,1,2)
            
            # Upsample to current scale
            curr_quant_up = F.interpolate(prev_quant, size=(curr_hw, curr_hw), mode='bicubic')
            curr_quant_up = curr_quant_up.reshape(B, -1, curr_hw * curr_hw).permute(0,2,1)
            
            # Build sequence for current scale
            cur_L = sum(pn ** 2 for pn in infer_patch_nums[:si+1])
            
            # Add upsampled embeddings for current scale
            x_curr = self.word_embed(curr_quant_up)
            curr_start = sum(pn ** 2 for pn in infer_patch_nums[:si])
            curr_end = curr_start + curr_hw ** 2
            x_curr = x_curr + pos_1LC[:, curr_start:curr_end]
            
            # cond_BD concatenate both conditional and unconditional class embeddings
            if si == 1 or not self.enable_var_kv:
                # If enable_var_kv is False, concatenate class embedding for each forward
                x_combined = torch.cat([cond_BD.unsqueeze(1), x_curr.repeat(2, 1, 1)], dim=1)
                # Total sequence is class token + current scale tokens
                query_len = 1 + curr_hw ** 2
            else:
                # If enable_var_kv is True, the class embedding is already in the kv cache from scale 1
                x_combined = x_curr.repeat(2, 1, 1)
                # Total sequence is current scale tokens
                query_len = curr_hw ** 2
            
            
            # Apply proper masking based on cached KV content
            if self.enable_var_kv and si > 1:
                # When we have cached KV from previous scales, we need to block attention
                # to the RandAR tokens (4x4 tokens) which are positions 1 to randar_tokens
                # in the cached KV (after class token at position 0)
                
                # Calculate cached KV length
                # KV cache contains: class token + all previous scale tokens
                cached_kv_len = 1 + sum(infer_patch_nums[j] ** 2 for j in range(1, si))
                
                # Pad query length to multiple of 8 (this is what actually gets passed to attention)
                padded_q_len = ((query_len + 7) // 8) * 8
                
                # Total KV length after concatenation = cached + padded current
                total_kv_len = cached_kv_len + padded_q_len
                
                # Pad KV length to ensure memory alignment for xformers
                # The stride needs to be a multiple of 4
                padded_kv_len = ((total_kv_len + 3) // 4) * 4
                if padded_kv_len < total_kv_len + 4:
                    padded_kv_len += 4  # Add extra padding to allow slicing
                
                # Create attention bias with proper memory alignment
                # Create a larger tensor and slice it to ensure proper stride
                attn_bias_full = torch.zeros(1, 1, padded_q_len, padded_kv_len, device=device)
                attn_bias_padded = attn_bias_full[:, :, :, :total_kv_len]
                
                # Block attention from current tokens to RandAR tokens in KV cache
                # RandAR tokens are at positions 1 to self.randar_tokens (after class token)
                # in the cached KV portion
                attn_bias_padded[:, :, :query_len, 1:self.randar_tokens+1] = -torch.inf
                
                attn_bias = attn_bias_padded
            else:
                # For the current scale tokens, we want bidirectional attention among themselves
                # No additional masking needed for the current tokens
                # Pad to multiple of 8 for memory efficient attention
                total_seq_len = query_len
                padded_len = ((total_seq_len + 7) // 8) * 8
                
                # Create attention bias for the full sequence (cached + current)
                attn_bias_padded = torch.zeros(1, 1, padded_len, padded_len, device=device)
                attn_bias_padded[:, :, :total_seq_len, total_seq_len:] = -torch.inf
                attn_bias_padded[:, :, total_seq_len:, :total_seq_len] = -torch.inf
                attn_bias = attn_bias_padded
            
            # Forward through transformer
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = x_combined
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            
            # Get logits
            logits_BlV = self.get_logits(x, cond_BD)
            
            if not self.enable_var_kv or si == 1:
                logits_BlV = logits_BlV[:, 1:, :]
            
            if invalid_ids is not None:
                logits_BlV[:, :, invalid_ids] = -100.0
            
            # Apply CFG with scale-dependent strength
            ratio = si / (len(infer_patch_nums) - 1)
            if not used_llamagen_cfg:
                t = cfg * 0.5 * (1 + ratio)  # Gradually increase CFG
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            else:
                cond_logits, uncond_logits = torch.split(logits_BlV, B, dim=0)
                logits_BlV = uncond_logits + (cond_logits - uncond_logits) * cfg
            
            # Sample all tokens for current scale
            idx_curr = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            all_indices.append(idx_curr)
        
        # Decode images
        if return_intermediate_scales:
            # Decode all scales
            intermediate_decoded = {}
            for si, indices in enumerate(all_indices):
                hw = infer_patch_nums[si]
                z_shape = [B, self.Cvae, hw, hw]
                decoded = vqvae.decode_code(indices, shape=z_shape)
                intermediate_decoded[hw] = decoded
            
            # Final image is the last scale
            samples = intermediate_decoded[infer_patch_nums[-1]]
        else:
            # Get final scale indices for decoding
            final_indices = all_indices[-1]  # Should be 16x16
            
            # Decode to image
            z_shape = [B, self.Cvae, infer_patch_nums[-1], infer_patch_nums[-1]]
            samples = vqvae.decode_code(final_indices, shape=z_shape)
        
        # Disable KV caching after inference
        for b in self.blocks: 
            b.attn.kv_caching(False)
        
        if return_intermediate_scales:
            return samples, intermediate_decoded
        else:
            return samples
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, infer_patch_nums, 
                randar_mask_ratio: float = 0.5, student_forcing: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward pass: RandAR for 4x4, VAR for subsequent scales
        :param label_B: class labels
        :param x_BLCv_wo_first_l: teacher forcing input (already includes 4x4 tokens for RandAR)
        :param infer_patch_nums: patch numbers for this training step
        :param randar_mask_ratio: ratio of tokens to mask for RandAR training
        :param student_forcing: if True, use student forcing mode (no masking, VAR-only processing)
        :return: logits BLV and loss mask
        """
        B = x_BLCv_wo_first_l.shape[0]
        device = label_B.device
        
        # Update patch numbers
        self.patch_nums = infer_patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        
        # For student forcing, we don't need begin_ends
        if not student_forcing:
            self.first_l = self.patch_nums[0] ** 2
            self.begin_ends = []
            cur = 0
            for i, pn in enumerate(self.patch_nums):
                self.begin_ends.append((cur, cur+pn ** 2))
                cur += pn*pn
        
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=device) < self.cond_drop_rate, self.num_classes, label_B)
            cond_BD = self.class_emb(label_B)
            
            # Prepare position embeddings for all scales
            pos_1LC = []
            for hw in self.patch_nums:
                curr_pos_1LC = F.interpolate(self.pos_1LC, size=(hw, hw), mode='area')
                curr_pos_1LC = curr_pos_1LC.reshape(1, self.pos_1LC.shape[1], -1).permute(0,2,1)
                pos_1LC.append(curr_pos_1LC)
            pos_1LC = torch.cat(pos_1LC, dim = 1)
            
            if student_forcing:
                # Student forcing mode: process all tokens with VAR positions (no special 4x4 handling)
                x_embed = self.word_embed(x_BLCv_wo_first_l)
                x_embed += pos_1LC[:, :self.L]
                
                # Prepend class token
                x_BLC = torch.cat([cond_BD.unsqueeze(1), x_embed], dim=1)
                
                # Create VAR-only attention mask (no RandAR masking)
                total_len = 1 + self.L
                attn_bias = self.create_var_attention_mask(total_len, device)
                
                # Variables needed for later processing
                orders = None
                randar_end = None
                num_masked = None
            else:
                # Teacher forcing mode with RandAR
                # Split input: first randar_tokens are for RandAR at randar_scale, rest for VAR
                x_randar = x_BLCv_wo_first_l[:, :self.randar_tokens]  # B x randar_tokens x Cvae
                x_rest = x_BLCv_wo_first_l[:, self.randar_tokens:]  # Rest of the tokens
                
                # RandAR processing for randar_scale
                orders = self.generate_random_order(B, device)
                num_masked = int(self.randar_tokens * randar_mask_ratio)
                
                # Process randar_scale tokens using helper method
                x_randar_embed = self.process_randar_tokens(x_randar, orders, num_masked, cond_BD, B, device)
                
                # Process rest with VAR positions
                x_rest_embed = self.word_embed(x_rest)
                x_rest_embed += pos_1LC[:, self.randar_tokens:self.L]
                
                # Combine RandAR and VAR embeddings
                x_BLC = torch.cat([x_randar_embed, x_rest_embed], dim=1)
    
                # Calculate sequence lengths based on mode
                if self.randar_mode == 'randar':
                    total_len = 1 + self.randar_tokens * 2 + (self.L - self.randar_tokens)
                    randar_end = 1 + self.randar_tokens * 2
                else:
                    total_len = 1 + self.L
                    randar_end = 1 + self.randar_tokens
                
                # Create hybrid attention mask using helper method
                attn_bias = self.create_attention_mask(total_len, randar_end, device)
        
        # from test_scripts.visualize_training_masks import visualize_attention_mask
        # visualize_attention_mask(attn_bias, title="Attention Mask", figsize=(10, 8), save_path=f'attn_bias_seqlen_{attn_bias.shape[2]}.png')

        # Forward through transformer
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # Get dtype for mixed precision
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        
        x_BLC = self.get_logits(x_BLC, cond_BD)

        if student_forcing:
            # Student forcing: extract logits (skip class token)
            logits_BLV = x_BLC[:, 1:]
            
            # All tokens contribute to loss in SF
            loss_mask = torch.ones(B, self.L, dtype=torch.bool, device=device)
        else:
            # Teacher forcing: extract and reorder logits using helper method
            logits_BLV = self.extract_and_reorder_logits(x_BLC, orders, randar_end)
    
            # Build loss mask: only masked 4x4 tokens and all VAR tokens contribute to loss
            # For RandAR tokens, create mask based on actual masked positions from orders - batched
            mask_randar = orders < num_masked  # B x randar_tokens
            
            mask_var = torch.ones(B, self.L - self.randar_tokens, dtype=torch.bool, device=device)
            loss_mask = torch.cat((mask_randar, mask_var), dim=1)

        # Sanity-check: logits length should now equal self.L
        assert logits_BLV.shape[1] == self.L, f"Expected {self.L} tokens, got {logits_BLV.shape[1]}"

        return logits_BLV, loss_mask
    
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5
        
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}, randar_scale={self.randar_scale}'