import time
import random
import scipy.stats as stats
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import ScaleAR
from models.helpers import sample_with_top_k_top_p_
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger
from utils.logger import UnifiedLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class ScaleARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, scalear_wo_ddp: ScaleAR, scalear: DDP,
        scalear_opt: AmpOptimizer, label_smooth: float, mask_ratio_min: float = 0.7,
        training_mode: str = 'teacher_forcing',  # 'teacher_forcing' or 'student_forcing' or 'mixed' or 'alternating' or 'hybrid_tf_sf' or 'csfl' or 'tf_then_single_sf'
        sigma: float = 0.5,  # mixing weight for mixed training mode
        hybrid_tf_scales: int = 8,  # number of scales to use TF for in hybrid mode
        sf_cfg_scale: float = 1.2,  # CFG scale for SF sampling
        sf_top_k: int = 900,  # Top-k for SF sampling
        sf_top_p: float = 0.96,  # Top-p for SF sampling
        sf_use_sampling: bool = True,  # Whether to use sampling vs argmax for SF
    ):
        super(ScaleARTrainer, self).__init__()
        
        self.scalear, self.vae_local = scalear, vae_local
        self.scalear_wo_ddp: ScaleAR = scalear_wo_ddp  # after torch.compile
        self.scalear_opt = scalear_opt
        
        del self.scalear_wo_ddp.rng
        self.scalear_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.patch_nums, self.resos = patch_nums, resos
        self.vae_embedding = F.normalize(self.vae_local.quantize.embedding.weight, p=2, dim=-1).to(device)
        self.Ch = self.vae_embedding.shape[1]
        self.training_mode = training_mode
        self.sigma = sigma
        self.device = device
        self.hybrid_tf_scales = hybrid_tf_scales
        # SF sampling parameters
        self.sf_cfg_scale = sf_cfg_scale
        self.sf_top_k = sf_top_k
        self.sf_top_p = sf_top_p
        self.sf_use_sampling = sf_use_sampling
        # Default inference schedule for student forcing
        self.default_inference_schedule = [4, 5, 6, 8, 10, 13, 16]
        
    @property
    def var_opt(self):
        """Compatibility property for train.py which expects var_opt"""
        return self.scalear_opt
    
    @property
    def var_wo_ddp(self):
        """Compatibility property for train.py which expects var_wo_ddp"""
        return self.scalear_wo_ddp

    def train_step_teacher_forcing(
        self, inp_B3HW: FTen, label_B: Union[ITen, FTen], curr_patch_nums: List[int], 
        mask_ratio: float
    ) -> Tuple[Ten, dict]:
        """
        Teacher forcing training: use GT features at each scale
        Returns loss and log info
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size
        
        # Teacher forcing: use GT features at each scale
        quant_z, gt_BL = self.encode_scalear_unified(inp_B3HW, curr_patch_nums)
        
        with self.scalear_opt.amp_ctx:
            # Forward pass with unified RandAR + VAR
            logits_BLV, loss_mask = self.scalear(label_B, quant_z, curr_patch_nums, randar_mask_ratio=mask_ratio)
            
            # Compute loss
            loss_per_tok = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            loss_per_tok = loss_per_tok * loss_mask.to(loss_per_tok.dtype)
            tokens_per_sample = loss_mask.sum(dim=1).to(loss_per_tok.dtype)
            total_loss = (loss_per_tok.sum(dim=1) / tokens_per_sample).mean()
        
        # Prepare log info
        log_info = {
            'logits': logits_BLV,
            'gt_indices': gt_BL,
            'loss_mask': loss_mask,
            'mode': 'teacher_forcing'
        }
        
        return total_loss, log_info
    
    def encode_scalear_unified(self, x, curr_patch_nums):
        """
        Encode image at multiple scales for ScaleAR unified training
        Returns embeddings suitable for RandAR (randar_scale) + VAR (subsequent scales)
        """
        h = self.vae_local.encode_conti(x)
        
        all_indices = []
        all_quant = []
        
        for num in range(len(curr_patch_nums)):
            curr_hw = curr_patch_nums[num]
            _h = F.interpolate(h.clone(), size=(curr_hw, curr_hw), mode='area')
            quant, _, log = self.vae_local.quantize(_h)
            indices = log[-1].view(quant.shape[0], -1)
            all_indices.append(indices)
            
            # For ScaleAR mode, if this is randar_scale, we need the raw embeddings
            if curr_hw == self.scalear_wo_ddp.randar_scale:
                # For randar_scale, use the actual quantized embeddings
                quant_randar = quant.reshape(quant.shape[0], quant.shape[1], -1).permute(0,2,1)
                all_quant.append(quant_randar)
            else:
                # For other scales, if previous scale exists, use upsampled version
                if num > 0:
                    prev_hw = curr_patch_nums[num-1]
                    # Get previous scale's quantized output
                    _h_prev = F.interpolate(h.clone(), size=(prev_hw, prev_hw), mode='area')
                    quant_prev, _, _ = self.vae_local.quantize(_h_prev)
                    # Upsample to current scale
                    next_quant = F.interpolate(quant_prev, size=(curr_hw, curr_hw), mode='bicubic')
                    next_quant = next_quant.reshape(quant.shape[0], quant.shape[1], -1).permute(0,2,1)
                    all_quant.append(next_quant)
        
        # Concatenate all quantized embeddings
        all_quant = torch.cat(all_quant, dim=1)  # B x total_tokens x Ch
        all_indices = torch.cat(all_indices, dim=1)  # B x total_tokens
        
        return all_quant, all_indices
    
    def train_step_student_forcing(
        self, inp_B3HW: FTen, label_B: Union[ITen, FTen], curr_patch_nums: List[int], 
        mask_ratio: float, prog_si: int, prog_wp_it: float,
        sf_cfg_scale: float = 1.0, sf_top_k: int = 0, sf_top_p: float = 0.0,
        sf_use_sampling: bool = False
    ) -> Tuple[Ten, dict]:
        """
        Student-forcing training: Generate each scale using model's own predictions
        Returns logits, loss mask, and ground truth indices for all scales
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size
        
        # Prepare labels for CFG if needed
        use_cfg = sf_cfg_scale > 1.0 and sf_use_sampling
        if use_cfg:
            # Double the batch with unconditional labels
            num_classes = self.scalear_wo_ddp.num_classes
            uncond_label = torch.full_like(label_B, fill_value=num_classes)
            label_B_cfg = torch.cat([label_B, uncond_label], dim=0)
        else:
            label_B_cfg = label_B
        device = inp_B3HW.device
        
        # Get full-scale GT latent and downsample to all required scales for supervision
        h_full = self.vae_local.encode_conti(inp_B3HW)
        gt_indices_by_scale = {}
        gt_quant_by_scale = {}
        
        for i, hw in enumerate(curr_patch_nums):
            h_scale = F.interpolate(h_full.clone(), size=(hw, hw), mode='area')
            quant, _, log = self.vae_local.quantize(h_scale)
            indices = log[-1].view(B, -1)
            gt_indices_by_scale[hw] = indices
            gt_quant_by_scale[hw] = quant
        
        # Initialize lists to collect outputs
        all_logits = []
        all_loss_masks = []
        all_gt_indices = []
        all_generated_quants = []  # Store generated embeddings for each scale
        
        # Generate progressively through all scales
        for i, curr_hw in enumerate(curr_patch_nums):
            if i == 0 and curr_hw == 4:
                # Phase 1: RandAR for 4x4
                quant_4x4 = gt_quant_by_scale[4].reshape(B, self.Ch, -1).permute(0, 2, 1)
                if use_cfg:
                    # Double the input for CFG
                    quant_4x4_cfg = quant_4x4.repeat(2, 1, 1)
                else:
                    quant_4x4_cfg = quant_4x4
                
                # Forward pass through RandAR (with gradients for training)
                with self.scalear_opt.amp_ctx:
                    logits_4x4_full, loss_mask_4x4_full = self.scalear(
                        label_B_cfg, quant_4x4_cfg, [4], randar_mask_ratio=mask_ratio
                    )
                
                # Extract conditional part for loss
                if use_cfg:
                    logits_4x4 = logits_4x4_full[:B]
                    loss_mask_4x4 = loss_mask_4x4_full[:B]
                else:
                    logits_4x4 = logits_4x4_full
                    loss_mask_4x4 = loss_mask_4x4_full
                
                all_logits.append(logits_4x4)
                all_loss_masks.append(loss_mask_4x4)
                all_gt_indices.append(gt_indices_by_scale[4])
                
                # Generate indices for next scale
                with torch.no_grad():
                    if sf_use_sampling:
                        if use_cfg:
                            # Apply proper CFG with gradual scaling
                            ratio = 1  # First scale (4x4)
                            t = sf_cfg_scale * 0.5 * (1 + ratio)
                            cond_logits = logits_4x4_full[:B]
                            uncond_logits = logits_4x4_full[B:]
                            logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                        else:
                            logits_cfg = logits_4x4
                        
                        pred_indices_4x4 = sample_with_top_k_top_p_(
                            logits_cfg,
                            rng=self.scalear_wo_ddp.rng,
                            top_k=sf_top_k,
                            top_p=sf_top_p,
                            num_samples=1
                        )[:, :, 0]
                    else:
                        pred_indices_4x4 = logits_4x4.argmax(dim=-1)
                    generated_quant_4x4 = self.vae_embedding[pred_indices_4x4]
                
                all_generated_quants.append(generated_quant_4x4)
                
            else:
                # Phase 2: VAR for subsequent scales
                # Get previous scale's generated embeddings
                prev_idx = i - 1
                prev_hw = curr_patch_nums[prev_idx]
                prev_quant = all_generated_quants[prev_idx]
                
                # Reshape and upsample previous scale
                prev_quant_2d = prev_quant.permute(0, 2, 1).reshape(B, self.Ch, prev_hw, prev_hw)
                upsampled = F.interpolate(prev_quant_2d, size=(curr_hw, curr_hw), mode='bicubic')
                upsampled_quant = upsampled.reshape(B, self.Ch, -1).permute(0, 2, 1)
                
                # Build full sequence of all generated scales so far + current upsampled
                scale_inputs = []
                for j in range(i):
                    scale_inputs.append(all_generated_quants[j])
                scale_inputs.append(upsampled_quant)
                scale_inputs = torch.cat(scale_inputs, dim=1)
                
                # Double inputs for CFG if needed
                if use_cfg:
                    scale_inputs_cfg = scale_inputs.repeat(2, 1, 1)
                else:
                    scale_inputs_cfg = scale_inputs
                
                # Forward pass for all scales up to current
                with self.scalear_opt.amp_ctx:
                    # Use student forcing mode in unified forward for proper handling of model predictions
                    logits_full_all, loss_mask_full_all = self.scalear(
                        label_B_cfg, scale_inputs_cfg, curr_patch_nums[:i+1], 
                        randar_mask_ratio=0.5, student_forcing=True
                    )
                
                # Extract logits for current scale tokens only
                start_idx = sum(pn**2 for pn in curr_patch_nums[:i])
                end_idx = start_idx + curr_hw**2
                
                if use_cfg:
                    # Extract conditional part for loss
                    logits_curr = logits_full_all[:B, start_idx:end_idx]
                    loss_mask_curr = loss_mask_full_all[:B, start_idx:end_idx]
                    # Store full logits for CFG sampling
                    logits_curr_full = logits_full_all[:, start_idx:end_idx]
                else:
                    logits_curr = logits_full_all[:, start_idx:end_idx]
                    loss_mask_curr = loss_mask_full_all[:, start_idx:end_idx]
                    logits_curr_full = logits_curr
                
                all_logits.append(logits_curr)
                all_loss_masks.append(loss_mask_curr)
                all_gt_indices.append(gt_indices_by_scale[curr_hw])
                
                # Generate indices for next scale
                with torch.no_grad():
                    if sf_use_sampling:
                        if use_cfg:
                            # Apply proper CFG with gradual scaling
                            ratio = i / (len(curr_patch_nums) - 1) if len(curr_patch_nums) > 1 else 0
                            t = sf_cfg_scale * 0.5 * (1 + ratio)
                            cond_logits = logits_curr_full[:B]
                            uncond_logits = logits_curr_full[B:]
                            logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                        else:
                            logits_cfg = logits_curr
                        
                        pred_indices_curr = sample_with_top_k_top_p_(
                            logits_cfg,
                            rng=self.scalear_wo_ddp.rng,
                            top_k=sf_top_k,
                            top_p=sf_top_p,
                            num_samples=1
                        )[:, :, 0]
                    else:
                        pred_indices_curr = logits_curr.argmax(dim=-1)
                    generated_quant_curr = self.vae_embedding[pred_indices_curr]
                
                all_generated_quants.append(generated_quant_curr)
        
        # Concatenate all outputs
        logits_BLV = torch.cat(all_logits, dim=1)
        loss_mask = torch.cat(all_loss_masks, dim=1)
        gt_BL = torch.cat(all_gt_indices, dim=1)
        
        # Compute loss
        with self.scalear_opt.amp_ctx:
            loss_per_tok = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            loss_per_tok = loss_per_tok * loss_mask.to(loss_per_tok.dtype)
            tokens_per_sample = loss_mask.sum(dim=1).to(loss_per_tok.dtype)
            total_loss = (loss_per_tok.sum(dim=1) / tokens_per_sample).mean()
        
        # Prepare log info
        log_info = {
            'logits': logits_BLV,
            'gt_indices': gt_BL,
            'loss_mask': loss_mask,
            'mode': 'student_forcing'
        }
        
        return total_loss, log_info

    def train_step_alternating_tf_sf(
        self, inp_B3HW: FTen, label_B: Union[ITen, FTen], curr_patch_nums: List[int], 
        mask_ratio: float, prog_si: int, prog_wp_it: float,
        sf_cfg_scale: float = 1.0, sf_top_k: int = 0, sf_top_p: float = 0.0,
        sf_use_sampling: bool = False
    ) -> Tuple[Ten, dict]:
        """
        Alternating TF/SF training mode:
        - Step 1 (4x4): TF with mask modeling (GT input)
        - Step 2 (e.g., 5x5): SF (model's 4x4 output upsampled as input)
        - Step 3 (e.g., 6x6): TF (GT 5x5 upsampled as input)
        - Step 4 (e.g., 7x7): SF (model's 6x6 output upsampled as input)
        Pattern: TF, SF, TF, SF, ...
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size
        
        # Prepare labels for CFG if needed
        use_cfg = sf_cfg_scale > 1.0 and sf_use_sampling
        if use_cfg:
            # Double the batch with unconditional labels
            num_classes = self.scalear_wo_ddp.num_classes
            uncond_label = torch.full_like(label_B, fill_value=num_classes)
            label_B_cfg = torch.cat([label_B, uncond_label], dim=0)
        else:
            label_B_cfg = label_B
        
        # Get GT embeddings for all scales
        h_full = self.vae_local.encode_conti(inp_B3HW)
        gt_indices_by_scale = {}
        gt_quant_by_scale = {}
        
        for hw in curr_patch_nums:
            h_scale = F.interpolate(h_full.clone(), size=(hw, hw), mode='area')
            quant, _, log = self.vae_local.quantize(h_scale)
            indices = log[-1].view(B, -1)
            gt_indices_by_scale[hw] = indices
            gt_quant_by_scale[hw] = quant
        
        # Build TF input sequence (all teacher forcing steps including 4x4)
        tf_inputs = []
        tf_scales = []
        
        for i in range(0, len(curr_patch_nums), 2):  # 0, 2, 4, ...
            curr_hw = curr_patch_nums[i]
            if i == 0:
                # First scale (4x4) with GT
                quant_4x4 = gt_quant_by_scale[4].reshape(B, self.Ch, -1).permute(0, 2, 1)
                tf_inputs.append(quant_4x4)
            else:
                # Upsample GT from previous scale
                prev_hw = curr_patch_nums[i-1]
                prev_gt = gt_quant_by_scale[prev_hw]
                prev_gt_2d = prev_gt.reshape(B, self.Ch, prev_hw, prev_hw)
                upsampled_gt = F.interpolate(prev_gt_2d, size=(curr_hw, curr_hw), mode='bicubic')
                upsampled_gt_quant = upsampled_gt.reshape(B, self.Ch, -1).permute(0, 2, 1)
                tf_inputs.append(upsampled_gt_quant)
            tf_scales.append(curr_hw)
        
        # Double TF inputs for CFG if needed
        if use_cfg:
            tf_input_seq = torch.cat(tf_inputs, dim=1).repeat(2, 1, 1)
        else:
            tf_input_seq = torch.cat(tf_inputs, dim=1)
        
        # Single forward pass for all TF steps (uses regular forward with masking for 4x4)
        with self.scalear_opt.amp_ctx:
            # Regular forward handles 4x4 masking automatically
            logits_tf_full, loss_mask_tf_full = self.scalear(
                label_B_cfg, tf_input_seq, tf_scales, randar_mask_ratio=mask_ratio
            )
        
        # Extract conditional part for loss
        if use_cfg:
            logits_tf = logits_tf_full[:B]
            loss_mask_tf = loss_mask_tf_full[:B]
        else:
            logits_tf = logits_tf_full
            loss_mask_tf = loss_mask_tf_full
        
        # Extract TF outputs for SF steps
        tf_outputs = {}
        start_idx = 0
        for i, scale in enumerate(tf_scales):
            end_idx = start_idx + scale**2
            with torch.no_grad():
                if sf_use_sampling:
                    if use_cfg:
                        # Apply proper CFG with gradual scaling
                        ratio = i / (len(tf_scales) - 1) if len(tf_scales) > 1 else 0
                        t = sf_cfg_scale * 0.5 * (1 + ratio)
                        curr_logits_cond = logits_tf_full[:B, start_idx:end_idx]
                        curr_logits_uncond = logits_tf_full[B:, start_idx:end_idx]
                        curr_logits_cfg = (1 + t) * curr_logits_cond - t * curr_logits_uncond
                    else:
                        curr_logits_cfg = logits_tf[:, start_idx:end_idx]
                    
                    pred_indices = sample_with_top_k_top_p_(
                        curr_logits_cfg,
                        rng=self.scalear_wo_ddp.rng,
                        top_k=sf_top_k,
                        top_p=sf_top_p,
                        num_samples=1
                    )[:, :, 0]
                else:
                    pred_indices = logits_tf[:, start_idx:end_idx].argmax(dim=-1)
                tf_outputs[scale] = self.vae_embedding[pred_indices]
            start_idx = end_idx
        
        # Build SF input sequence (all student forcing steps)
        sf_inputs = []
        sf_scales = []
        sf_outputs = {}
        
        for i in range(1, len(curr_patch_nums), 2):  # 1, 3, 5, ...
            curr_hw = curr_patch_nums[i]
            prev_hw = curr_patch_nums[i-1]
            
            # Use model output from previous TF step
            prev_output = tf_outputs[prev_hw]
            prev_2d = prev_output.permute(0, 2, 1).reshape(B, self.Ch, prev_hw, prev_hw)
            upsampled = F.interpolate(prev_2d, size=(curr_hw, curr_hw), mode='bicubic')
            upsampled_quant = upsampled.reshape(B, self.Ch, -1).permute(0, 2, 1)
            
            sf_inputs.append(upsampled_quant)
            sf_scales.append(curr_hw)
        
        # Single forward pass for all SF steps (if any)
        if sf_inputs:
            if use_cfg:
                sf_input_seq = torch.cat(sf_inputs, dim=1).repeat(2, 1, 1)
            else:
                sf_input_seq = torch.cat(sf_inputs, dim=1)
            
            with self.scalear_opt.amp_ctx:
                # Use student forcing mode (no masking, no special 4x4 handling)
                logits_sf_full, loss_mask_sf_full = self.scalear(
                    label_B_cfg, sf_input_seq, sf_scales,
                    randar_mask_ratio=0.5, student_forcing=True
                )
            
            # Extract conditional part for loss
            if use_cfg:
                logits_sf = logits_sf_full[:B]
                loss_mask_sf = loss_mask_sf_full[:B]
            else:
                logits_sf = logits_sf_full
                loss_mask_sf = loss_mask_sf_full
            
            # Store SF outputs (for potential future use)
            start_idx = 0
            for i, scale in enumerate(sf_scales):
                end_idx = start_idx + scale**2
                with torch.no_grad():
                    if sf_use_sampling:
                        if use_cfg:
                            # Apply proper CFG with gradual scaling
                            ratio = i / (len(sf_scales) - 1) if len(sf_scales) > 1 else 0
                            t = sf_cfg_scale * 0.5 * (1 + ratio)
                            curr_logits_cond = logits_sf_full[:B, start_idx:end_idx]
                            curr_logits_uncond = logits_sf_full[B:, start_idx:end_idx]
                            curr_logits_cfg = (1 + t) * curr_logits_cond - t * curr_logits_uncond
                        else:
                            curr_logits_cfg = logits_sf[:, start_idx:end_idx]
                        
                        pred_indices = sample_with_top_k_top_p_(
                            curr_logits_cfg,
                            rng=self.scalear_wo_ddp.rng,
                            top_k=sf_top_k,
                            top_p=sf_top_p,
                            num_samples=1
                        )[:, :, 0]
                    else:
                        pred_indices = logits_sf[:, start_idx:end_idx].argmax(dim=-1)
                    sf_outputs[scale] = self.vae_embedding[pred_indices]
                start_idx = end_idx
        
        # Interleave outputs in correct order
        all_logits = []
        all_loss_masks = []
        all_gt_indices = []
        
        tf_idx = 0
        sf_idx = 0
        
        for i, curr_hw in enumerate(curr_patch_nums):
            if i % 2 == 0:
                # TF step (including first 4x4)
                start = sum(tf_scales[j]**2 for j in range(tf_idx))
                end = start + curr_hw**2
                all_logits.append(logits_tf[:, start:end])
                all_loss_masks.append(loss_mask_tf[:, start:end])
                all_gt_indices.append(gt_indices_by_scale[curr_hw])
                tf_idx += 1
            else:
                # SF step
                if sf_inputs:
                    start = sum(sf_scales[j]**2 for j in range(sf_idx))
                    end = start + curr_hw**2
                    all_logits.append(logits_sf[:, start:end])
                    all_loss_masks.append(loss_mask_sf[:, start:end])
                    all_gt_indices.append(gt_indices_by_scale[curr_hw])
                    sf_idx += 1
        
        # Concatenate all outputs
        logits_BLV = torch.cat(all_logits, dim=1)
        loss_mask = torch.cat(all_loss_masks, dim=1)
        gt_BL = torch.cat(all_gt_indices, dim=1)
        
        # Compute total loss and separate TF and SF losses for logging
        with self.scalear_opt.amp_ctx:
            loss_per_tok = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            loss_per_tok = loss_per_tok * loss_mask.to(loss_per_tok.dtype)
            tokens_per_sample = loss_mask.sum(dim=1).to(loss_per_tok.dtype)
            total_loss = (loss_per_tok.sum(dim=1) / tokens_per_sample).mean()
            
            # Teacher forcing indices: 0, 2, 4, ...
            tf_indices = [0] + [i for i in range(2, len(curr_patch_nums), 2) if i < len(curr_patch_nums)]
            tf_masks = []
            tf_losses = []
            for idx in tf_indices:
                start = sum(curr_patch_nums[j]**2 for j in range(idx))
                end = start + curr_patch_nums[idx]**2
                tf_masks.append(loss_mask[:, start:end])
                tf_losses.append(loss_per_tok[:, start:end])
            
            if tf_masks:
                tf_mask_cat = torch.cat(tf_masks, dim=1)
                tf_loss_cat = torch.cat(tf_losses, dim=1)
                tf_tokens = tf_mask_cat.sum(dim=1).to(tf_loss_cat.dtype)
                self.last_teacher_loss = (tf_loss_cat.sum(dim=1) / tf_tokens).mean().item()
            
            # Student forcing indices: 1, 3, 5, ...
            sf_indices = [i for i in range(1, len(curr_patch_nums), 2)]
            sf_masks = []
            sf_losses = []
            for idx in sf_indices:
                start = sum(curr_patch_nums[j]**2 for j in range(idx))
                end = start + curr_patch_nums[idx]**2
                sf_masks.append(loss_mask[:, start:end])
                sf_losses.append(loss_per_tok[:, start:end])
            
            if sf_masks:
                sf_mask_cat = torch.cat(sf_masks, dim=1)
                sf_loss_cat = torch.cat(sf_losses, dim=1)
                sf_tokens = sf_mask_cat.sum(dim=1).to(sf_loss_cat.dtype)
                self.last_student_loss = (sf_loss_cat.sum(dim=1) / sf_tokens).mean().item()
        
        # Prepare log info
        log_info = {
            'logits': logits_BLV,
            'gt_indices': gt_BL,
            'loss_mask': loss_mask,
            'mode': 'alternating',
            'teacher_loss': getattr(self, 'last_teacher_loss', 0.0),
            'student_loss': getattr(self, 'last_student_loss', 0.0)
        }
        
        return total_loss, log_info

    def train_step_csfl(
        self, inp_B3HW: FTen, label_B: Union[ITen, FTen], curr_patch_nums: List[int], 
        mask_ratio: float, prog_si: int, prog_wp_it: float,
        sf_cfg_scale: float = 1.0, sf_top_k: int = 0, sf_top_p: float = 0.0,
        sf_use_sampling: bool = False
    ) -> Tuple[Ten, dict]:
        """
        Two-pass training mode:
        Pass 1: Teacher forcing with all scales using GT inputs -> compute TF loss
        Pass 2: Student forcing with sampled outputs from Pass 1 -> compute SF loss
        Returns: combined_loss, scale_log2 (for compatibility)
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size
        
        # Prepare labels for CFG if needed
        use_cfg = sf_cfg_scale > 1.0 and sf_use_sampling
        if use_cfg:
            # Double the batch with unconditional labels
            num_classes = self.scalear_wo_ddp.num_classes
            uncond_label = torch.full_like(label_B, fill_value=num_classes)
            label_B_cfg = torch.cat([label_B, uncond_label], dim=0)
        else:
            label_B_cfg = label_B
        
        # Get GT embeddings and indices for all scales
        h_full = self.vae_local.encode_conti(inp_B3HW)
        gt_indices_by_scale = {}
        gt_quant_by_scale = {}
        
        for hw in curr_patch_nums:
            h_scale = F.interpolate(h_full.clone(), size=(hw, hw), mode='area')
            quant, _, log = self.vae_local.quantize(h_scale)
            indices = log[-1].view(B, -1)
            gt_indices_by_scale[hw] = indices
            gt_quant_by_scale[hw] = quant
        
        # ======= Pass 1: Teacher Forcing with all scales =======
        tf_inputs = []
        for i, curr_hw in enumerate(curr_patch_nums):
            if i == 0 and curr_hw == 4:
                # First scale (4x4) with GT
                quant_4x4 = gt_quant_by_scale[4].reshape(B, self.Ch, -1).permute(0, 2, 1)
                if use_cfg:
                    # Double the input for CFG
                    quant_4x4 = quant_4x4.repeat(2, 1, 1)
                tf_inputs.append(quant_4x4)
            else:
                # Upsample GT from previous scale
                prev_hw = curr_patch_nums[i-1]
                prev_gt = gt_quant_by_scale[prev_hw]
                prev_gt_2d = prev_gt.reshape(B, self.Ch, prev_hw, prev_hw)
                upsampled_gt = F.interpolate(prev_gt_2d, size=(curr_hw, curr_hw), mode='bicubic')
                upsampled_gt_quant = upsampled_gt.reshape(B, self.Ch, -1).permute(0, 2, 1)
                if use_cfg:
                    # Double the input for CFG
                    upsampled_gt_quant = upsampled_gt_quant.repeat(2, 1, 1)
                tf_inputs.append(upsampled_gt_quant)
        
        # Single forward pass for all TF scales
        tf_input_seq = torch.cat(tf_inputs, dim=1)
        with self.scalear_opt.amp_ctx:
            logits_tf_full, loss_mask_tf_full = self.scalear(
                label_B_cfg, tf_input_seq, curr_patch_nums, randar_mask_ratio=mask_ratio
            )
        
        # Extract conditional part for loss computation
        if use_cfg:
            logits_tf = logits_tf_full[:B]
            loss_mask_tf = loss_mask_tf_full[:B]
        else:
            logits_tf = logits_tf_full
            loss_mask_tf = loss_mask_tf_full
        
        # Store TF data for scale-wise loss calculation
        gt_indices_tf = torch.cat([gt_indices_by_scale[scale] for scale in curr_patch_nums], dim=1)
        
        # Compute TF loss (only on conditional part)
        with self.scalear_opt.amp_ctx:
            loss_per_tok_tf = self.train_loss(logits_tf.reshape(-1, V), gt_indices_tf.reshape(-1)).reshape(B, -1)
            loss_per_tok_tf = loss_per_tok_tf * loss_mask_tf.to(loss_per_tok_tf.dtype)
            tokens_tf = loss_mask_tf.sum(dim=1).to(loss_per_tok_tf.dtype)
            tf_loss = (loss_per_tok_tf.sum(dim=1) / tokens_tf).mean()
            self.last_teacher_loss = tf_loss.item()
        
        # Sample from each scale's output to construct SF inputs
        sampled_quants = []
        start_idx = 0
        for i, scale in enumerate(curr_patch_nums):
            end_idx = start_idx + scale**2
            
            with torch.no_grad():
                if sf_use_sampling:
                    if use_cfg:
                        # Apply CFG with gradual scaling
                        ratio = i / (len(curr_patch_nums) - 1) if len(curr_patch_nums) > 1 else 0
                        t = sf_cfg_scale * 0.5 * (1 + ratio)
                        cond_logits = logits_tf_full[:B, start_idx:end_idx]
                        uncond_logits = logits_tf_full[B:, start_idx:end_idx]
                        scale_logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                    else:
                        scale_logits_cfg = logits_tf[:, start_idx:end_idx]
                    
                    # Sample using top-k/top-p
                    pred_indices = sample_with_top_k_top_p_(
                        scale_logits_cfg,
                        rng=self.scalear_wo_ddp.rng,
                        top_k=sf_top_k,
                        top_p=sf_top_p,
                        num_samples=1
                    )[:, :, 0]
                else:
                    # Use argmax (no CFG for deterministic)
                    pred_indices = logits_tf[:, start_idx:end_idx].argmax(dim=-1)
                
                sampled_quant = self.vae_embedding[pred_indices]
                sampled_quants.append(sampled_quant)
            
            start_idx = end_idx
        
        # ======= Pass 2: Student Forcing with sampled inputs =======
        # Build SF input sequence using sampled outputs from Pass 1
        sf_inputs = []
        sf_scales_for_loss = []  # Scales where we compute loss
        
        for i in range(len(curr_patch_nums) - 1):  # Skip last scale as it has no "next"
            curr_hw = curr_patch_nums[i]
            next_hw = curr_patch_nums[i + 1]
            
            # Use sampled output from current scale
            sampled_quant_curr = sampled_quants[i]
            
            # Reshape and upsample to next scale
            sampled_2d = sampled_quant_curr.permute(0, 2, 1).reshape(B, self.Ch, curr_hw, curr_hw)
            upsampled = F.interpolate(sampled_2d, size=(next_hw, next_hw), mode='bicubic')
            upsampled_quant = upsampled.reshape(B, self.Ch, -1).permute(0, 2, 1)
            
            sf_inputs.append(upsampled_quant)
            sf_scales_for_loss.append(next_hw)
        
        if sf_inputs:
            # Forward pass with SF inputs (no CFG needed for SF since we only compute loss, not sample)
            sf_input_seq = torch.cat(sf_inputs, dim=1)
            
            with self.scalear_opt.amp_ctx:
                logits_sf, loss_mask_sf = self.scalear(
                    label_B, sf_input_seq, sf_scales_for_loss, 
                    randar_mask_ratio=0.5, student_forcing=True
                )
            
            # Store SF data for scale-wise loss calculation
            gt_indices_sf = torch.cat([gt_indices_by_scale[scale] for scale in sf_scales_for_loss], dim=1)
            
            # Compute SF loss: SF predictions vs next scale's GT output (only on conditional part)
            with self.scalear_opt.amp_ctx:
                loss_per_tok_sf = self.train_loss(logits_sf.reshape(-1, V), gt_indices_sf.reshape(-1)).reshape(B, -1)
                loss_per_tok_sf = loss_per_tok_sf * loss_mask_sf.to(loss_per_tok_sf.dtype)
                tokens_sf = loss_mask_sf.sum(dim=1).to(loss_per_tok_sf.dtype)
                sf_loss = (loss_per_tok_sf.sum(dim=1) / tokens_sf).mean()
                self.last_student_loss = sf_loss.item()
            
            # Combine TF and SF losses
            total_loss = tf_loss + sf_loss
            self.last_csfl_total_loss = total_loss.item()
            
            # ============ VISUALIZATION CODE FOR TWO-PASS MODE ============
            # Visualize TF inputs/outputs and SF inputs/outputs at each scale
            Flag = False
            if Flag:
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec
                import os
                
                with torch.no_grad():
                    batch_size = min(B, 8)  # Visualize up to 4 samples
                    
                    for batch_idx in range(batch_size):
                        # Create figure with subplots for each scale
                        num_scales = len(curr_patch_nums)
                        fig = plt.figure(figsize=(20, 4 * num_scales))
                        gs = gridspec.GridSpec(num_scales, 6, figure=fig, hspace=0.3, wspace=0.2)
                        
                        # Process each scale
                        for scale_idx, scale in enumerate(curr_patch_nums):
                            # Get GT at this scale
                            gt_quant = gt_quant_by_scale[scale][batch_idx:batch_idx+1]
                            gt_quant_2d = gt_quant.reshape(1, self.Ch, scale, scale)
                            gt_decoded = self.vae_local.decode(gt_quant_2d)
                            gt_img = gt_decoded[0].clamp(0, 1).cpu()
                            
                            # TF Input at this scale
                            tf_start = sum(s**2 for s in curr_patch_nums[:scale_idx])
                            tf_end = tf_start + scale**2
                            tf_input_emb = tf_input_seq[batch_idx:batch_idx+1, tf_start:tf_end]
                            tf_input_2d = tf_input_emb.permute(0, 2, 1).reshape(1, self.Ch, scale, scale)
                            tf_input_decoded = self.vae_local.decode(tf_input_2d)
                            tf_input_img = tf_input_decoded[0].clamp(0, 1).cpu()
                            
                            # TF Output (prediction) at this scale
                            # Use conditional logits for visualization
                            tf_pred_indices = logits_tf[batch_idx:batch_idx+1, tf_start:tf_end].argmax(dim=-1)
                            tf_pred_emb = self.vae_embedding[tf_pred_indices]
                            tf_pred_2d = tf_pred_emb.permute(0, 2, 1).reshape(1, self.Ch, scale, scale)
                            tf_pred_decoded = self.vae_local.decode(tf_pred_2d)
                            tf_pred_img = tf_pred_decoded[0].clamp(0, 1).cpu()
                            
                            # Compute TF accuracy at this scale
                            tf_gt_indices_scale = gt_indices_tf[batch_idx:batch_idx+1, tf_start:tf_end]
                            tf_acc_scale = (tf_pred_indices == tf_gt_indices_scale).float().mean().item() * 100
                            
                            # Plot TF components
                            ax1 = fig.add_subplot(gs[scale_idx, 0])
                            ax1.imshow(tf_input_img.permute(1, 2, 0))
                            ax1.set_title(f'TF Input {scale}x{scale}')
                            ax1.axis('off')
                            
                            ax2 = fig.add_subplot(gs[scale_idx, 1])
                            ax2.imshow(tf_pred_img.permute(1, 2, 0))
                            ax2.set_title(f'TF Output\n(Acc: {tf_acc_scale:.1f}%)')
                            ax2.axis('off')
                            
                            ax3 = fig.add_subplot(gs[scale_idx, 2])
                            ax3.imshow(gt_img.permute(1, 2, 0))
                            ax3.set_title(f'GT {scale}x{scale}')
                            ax3.axis('off')
                            
                            # SF components (if this scale has SF prediction)
                            if scale in sf_scales_for_loss:
                                sf_scale_idx = sf_scales_for_loss.index(scale)
                                sf_start = sum(s**2 for s in sf_scales_for_loss[:sf_scale_idx])
                                sf_end = sf_start + scale**2
                                
                                # SF Input (upsampled from previous scale's TF output)
                                sf_input_emb = sf_input_seq[batch_idx:batch_idx+1, sf_start:sf_end]
                                sf_input_2d = sf_input_emb.permute(0, 2, 1).reshape(1, self.Ch, scale, scale)
                                sf_input_decoded = self.vae_local.decode(sf_input_2d)
                                sf_input_img = sf_input_decoded[0].clamp(0, 1).cpu()
                                
                                # SF Output (prediction)
                                sf_pred_indices = logits_sf[batch_idx:batch_idx+1, sf_start:sf_end].argmax(dim=-1)
                                sf_pred_emb = self.vae_embedding[sf_pred_indices]
                                sf_pred_2d = sf_pred_emb.permute(0, 2, 1).reshape(1, self.Ch, scale, scale)
                                sf_pred_decoded = self.vae_local.decode(sf_pred_2d)
                                sf_pred_img = sf_pred_decoded[0].clamp(0, 1).cpu()
                                
                                # Compute SF accuracy at this scale
                                sf_gt_indices_scale = gt_indices_sf[batch_idx:batch_idx+1, sf_start:sf_end]
                                sf_acc_scale = (sf_pred_indices == sf_gt_indices_scale).float().mean().item() * 100
                                
                                # Plot SF components
                                ax4 = fig.add_subplot(gs[scale_idx, 3])
                                ax4.imshow(sf_input_img.permute(1, 2, 0))
                                ax4.set_title(f'SF Input {scale}x{scale}')
                                ax4.axis('off')
                                
                                ax5 = fig.add_subplot(gs[scale_idx, 4])
                                ax5.imshow(sf_pred_img.permute(1, 2, 0))
                                ax5.set_title(f'SF Output\n(Acc: {sf_acc_scale:.1f}%)')
                                ax5.axis('off')
                                
                                # Difference map between SF output and GT
                                diff_sf_gt = torch.abs(sf_pred_img - gt_img).mean(dim=0)
                                ax6 = fig.add_subplot(gs[scale_idx, 5])
                                im = ax6.imshow(diff_sf_gt, cmap='hot', vmin=0, vmax=0.5)
                                ax6.set_title(f'|SF-GT| Diff\n(Mean: {diff_sf_gt.mean():.3f})')
                                ax6.axis('off')
                                plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
                            else:
                                # No SF at this scale (first scale)
                                ax4 = fig.add_subplot(gs[scale_idx, 3])
                                ax4.text(0.5, 0.5, 'No SF\n(First Scale)', ha='center', va='center', fontsize=12)
                                ax4.axis('off')
                                
                                ax5 = fig.add_subplot(gs[scale_idx, 4])
                                ax5.text(0.5, 0.5, 'No SF\n(First Scale)', ha='center', va='center', fontsize=12)
                                ax5.axis('off')
                                
                                # TF-GT difference instead
                                diff_tf_gt = torch.abs(tf_pred_img - gt_img).mean(dim=0)
                                ax6 = fig.add_subplot(gs[scale_idx, 5])
                                im = ax6.imshow(diff_tf_gt, cmap='hot', vmin=0, vmax=0.5)
                                ax6.set_title(f'|TF-GT| Diff\n(Mean: {diff_tf_gt.mean():.3f})')
                                ax6.axis('off')
                                plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
                        
                        # Add overall title
                        sampling_info = f"Sampling: {'ON' if sf_use_sampling else 'OFF'}"
                        if sf_use_sampling:
                            sampling_info += f" (top_k={sf_top_k}, top_p={sf_top_p:.2f})"
                        fig.suptitle(
                            f'Two-Pass Visualization - Step {prog_si} - Sample {batch_idx+1}/{batch_size}\n'
                            f'Scales: {curr_patch_nums}, SF Scales: {sf_scales_for_loss}\n'
                            f'TF Loss: {self.last_teacher_loss:.4f}, SF Loss: {self.last_student_loss:.4f}, Total: {self.last_csfl_total_loss:.4f}\n'
                            f'{sampling_info}',
                            fontsize=14
                        )
                        
                        # Save figure
                        vis_dir = 'csfl_vis'
                        os.makedirs(vis_dir, exist_ok=True)
                        plt.savefig(f'{vis_dir}/step_{prog_si:06d}_sample_{batch_idx:02d}.png', 
                                   dpi=100, bbox_inches='tight')
                        plt.close(fig)
                    
                    print(f"[TWO-PASS VIS] Saved visualization for {batch_size} samples to {vis_dir}/")
            # ============ END VISUALIZATION CODE ============
            
            # Prepare log info with both TF and SF data for scale-wise calculation
            log_info = {
                'mode': 'csfl',
                'teacher_loss': self.last_teacher_loss,
                'student_loss': self.last_student_loss,
                'total_loss': self.last_csfl_total_loss,
                # TF data for scale-wise logging
                'tf_logits': logits_tf,
                'tf_gt_indices': gt_indices_tf,
                'tf_loss_mask': loss_mask_tf,
                'tf_scales': curr_patch_nums,
                # SF data for scale-wise logging
                'sf_logits': logits_sf,
                'sf_gt_indices': gt_indices_sf,
                'sf_loss_mask': loss_mask_sf,
                'sf_scales': sf_scales_for_loss,
            }
        else:
            # No SF pass needed (only one scale)
            total_loss = tf_loss
            self.last_student_loss = 0.0
            self.last_csfl_total_loss = tf_loss.item()
            
            # Prepare log info with only TF data
            log_info = {
                'mode': 'csfl',
                'teacher_loss': self.last_teacher_loss,
                'student_loss': 0.0,
                'total_loss': self.last_csfl_total_loss,
                # TF data for scale-wise logging
                'tf_logits': logits_tf,
                'tf_gt_indices': gt_indices_tf,
                'tf_loss_mask': loss_mask_tf,
                'tf_scales': curr_patch_nums,
                # No SF data
                'sf_logits': None,
                'sf_gt_indices': None,
                'sf_loss_mask': None,
                'sf_scales': [],
            }
        
        return total_loss, log_info

    def train_step_tf_then_single_sf(
        self, inp_B3HW: FTen, label_B: Union[ITen, FTen], curr_patch_nums: List[int],
        mask_ratio: float, prog_si: int, prog_wp_it: float,
        sf_cfg_scale: float = 1.0, sf_top_k: int = 0, sf_top_p: float = 0.0,
        sf_use_sampling: bool = False
    ) -> Tuple[Ten, dict]:
        """
        TF-then-Single-SF training mode:
        Pass 1: Teacher forcing with all scales using GT inputs -> compute TF loss
        Pass 2: Student forcing on ONLY ONE randomly selected scale -> compute SF loss
        Returns: combined_loss (TF + SF for one scale)
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size

        # Prepare labels for CFG if needed
        use_cfg = sf_cfg_scale > 1.0 and sf_use_sampling
        if use_cfg:
            # Double the batch with unconditional labels
            num_classes = self.scalear_wo_ddp.num_classes
            uncond_label = torch.full_like(label_B, fill_value=num_classes)
            label_B_cfg = torch.cat([label_B, uncond_label], dim=0)
        else:
            label_B_cfg = label_B

        # Get GT embeddings and indices for all scales
        h_full = self.vae_local.encode_conti(inp_B3HW)
        gt_indices_by_scale = {}
        gt_quant_by_scale = {}

        for hw in curr_patch_nums:
            h_scale = F.interpolate(h_full.clone(), size=(hw, hw), mode='area')
            quant, _, log = self.vae_local.quantize(h_scale)
            indices = log[-1].view(B, -1)
            gt_indices_by_scale[hw] = indices
            gt_quant_by_scale[hw] = quant

        # ======= Pass 1: Teacher Forcing with all scales =======
        tf_inputs = []
        for i, curr_hw in enumerate(curr_patch_nums):
            if i == 0 and curr_hw == 4:
                # First scale (4x4) with GT
                quant_4x4 = gt_quant_by_scale[4].reshape(B, self.Ch, -1).permute(0, 2, 1)
                if use_cfg:
                    # Double the input for CFG
                    quant_4x4 = quant_4x4.repeat(2, 1, 1)
                tf_inputs.append(quant_4x4)
            else:
                # Upsample GT from previous scale
                prev_hw = curr_patch_nums[i-1]
                prev_gt = gt_quant_by_scale[prev_hw]
                prev_gt_2d = prev_gt.reshape(B, self.Ch, prev_hw, prev_hw)
                upsampled_gt = F.interpolate(prev_gt_2d, size=(curr_hw, curr_hw), mode='bicubic')
                upsampled_gt_quant = upsampled_gt.reshape(B, self.Ch, -1).permute(0, 2, 1)
                if use_cfg:
                    # Double the input for CFG
                    upsampled_gt_quant = upsampled_gt_quant.repeat(2, 1, 1)
                tf_inputs.append(upsampled_gt_quant)

        # Single forward pass for all TF scales
        tf_input_seq = torch.cat(tf_inputs, dim=1)
        with self.scalear_opt.amp_ctx:
            logits_tf_full, loss_mask_tf_full = self.scalear(
                label_B_cfg, tf_input_seq, curr_patch_nums, randar_mask_ratio=mask_ratio
            )

        # Extract conditional part for loss computation
        if use_cfg:
            logits_tf = logits_tf_full[:B]
            loss_mask_tf = loss_mask_tf_full[:B]
        else:
            logits_tf = logits_tf_full
            loss_mask_tf = loss_mask_tf_full

        # Store TF data for scale-wise loss calculation
        gt_indices_tf = torch.cat([gt_indices_by_scale[scale] for scale in curr_patch_nums], dim=1)

        # Compute TF loss (only on conditional part)
        with self.scalear_opt.amp_ctx:
            loss_per_tok_tf = self.train_loss(logits_tf.reshape(-1, V), gt_indices_tf.reshape(-1)).reshape(B, -1)
            loss_per_tok_tf = loss_per_tok_tf * loss_mask_tf.to(loss_per_tok_tf.dtype)
            tokens_tf = loss_mask_tf.sum(dim=1).to(loss_per_tok_tf.dtype)
            tf_loss = (loss_per_tok_tf.sum(dim=1) / tokens_tf).mean()
            self.last_teacher_loss = tf_loss.item()

        # Sample from each scale's output to construct SF inputs
        sampled_quants = []
        start_idx = 0
        for i, scale in enumerate(curr_patch_nums):
            end_idx = start_idx + scale**2

            with torch.no_grad():
                if sf_use_sampling:
                    if use_cfg:
                        # Apply CFG with gradual scaling
                        ratio = i / (len(curr_patch_nums) - 1) if len(curr_patch_nums) > 1 else 0
                        t = sf_cfg_scale * 0.5 * (1 + ratio)
                        cond_logits = logits_tf_full[:B, start_idx:end_idx]
                        uncond_logits = logits_tf_full[B:, start_idx:end_idx]
                        scale_logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                    else:
                        scale_logits_cfg = logits_tf[:, start_idx:end_idx]

                    # Sample using top-k/top-p
                    pred_indices = sample_with_top_k_top_p_(
                        scale_logits_cfg,
                        rng=self.scalear_wo_ddp.rng,
                        top_k=sf_top_k,
                        top_p=sf_top_p,
                        num_samples=1
                    )[:, :, 0]
                else:
                    # Use argmax (no CFG for deterministic)
                    pred_indices = logits_tf[:, start_idx:end_idx].argmax(dim=-1)

                sampled_quant = self.vae_embedding[pred_indices]
                sampled_quants.append(sampled_quant)

            start_idx = end_idx

        # ======= Pass 2: Student Forcing on ONE randomly selected scale =======
        # Check if we have more than one scale (need at least 2 scales for SF)
        if len(curr_patch_nums) > 1:
            # Randomly select one scale index (excluding the last scale since it has no "next")
            import random
            selected_idx = random.randint(0, len(curr_patch_nums) - 2)

            curr_hw = curr_patch_nums[selected_idx]
            next_hw = curr_patch_nums[selected_idx + 1]

            # Use sampled output from selected scale
            sampled_quant_curr = sampled_quants[selected_idx]

            # Reshape and upsample to next scale
            sampled_2d = sampled_quant_curr.permute(0, 2, 1).reshape(B, self.Ch, curr_hw, curr_hw)
            upsampled = F.interpolate(sampled_2d, size=(next_hw, next_hw), mode='bicubic')
            upsampled_quant = upsampled.reshape(B, self.Ch, -1).permute(0, 2, 1)

            # Forward pass with SF input (no CFG needed for SF since we only compute loss)
            with self.scalear_opt.amp_ctx:
                logits_sf, loss_mask_sf = self.scalear(
                    label_B, upsampled_quant, [next_hw],
                    randar_mask_ratio=0.5, student_forcing=True
                )

            # Get GT indices for the selected next scale
            gt_indices_sf = gt_indices_by_scale[next_hw]

            # Compute SF loss
            with self.scalear_opt.amp_ctx:
                loss_per_tok_sf = self.train_loss(logits_sf.reshape(-1, V), gt_indices_sf.reshape(-1)).reshape(B, -1)
                loss_per_tok_sf = loss_per_tok_sf * loss_mask_sf.to(loss_per_tok_sf.dtype)
                tokens_sf = loss_mask_sf.sum(dim=1).to(loss_per_tok_sf.dtype)
                sf_loss = (loss_per_tok_sf.sum(dim=1) / tokens_sf).mean()
                self.last_student_loss = sf_loss.item()

            # Combine TF and SF losses
            total_loss = tf_loss + self.sigma * sf_loss
            self.last_tf_single_sf_total_loss = total_loss.item()

            # Prepare log info with both TF and SF data for scale-wise calculation
            log_info = {
                'mode': 'tf_then_single_sf',
                'teacher_loss': self.last_teacher_loss,
                'student_loss': self.last_student_loss,
                'total_loss': self.last_tf_single_sf_total_loss,
                'selected_scale_idx': selected_idx,
                'selected_scale': next_hw,
                # TF data for scale-wise logging
                'tf_logits': logits_tf,
                'tf_gt_indices': gt_indices_tf,
                'tf_loss_mask': loss_mask_tf,
                'tf_scales': curr_patch_nums,
                # SF data for scale-wise logging
                'sf_logits': logits_sf,
                'sf_gt_indices': gt_indices_sf,
                'sf_loss_mask': loss_mask_sf,
                'sf_scales': [next_hw],
            }
        else:
            # No SF pass needed (only one scale)
            total_loss = tf_loss
            self.last_student_loss = 0.0
            self.last_tf_single_sf_total_loss = tf_loss.item()

            # Prepare log info with only TF data
            log_info = {
                'mode': 'tf_then_single_sf',
                'teacher_loss': self.last_teacher_loss,
                'student_loss': 0.0,
                'total_loss': self.last_tf_single_sf_total_loss,
                'selected_scale_idx': -1,
                'selected_scale': -1,
                # TF data for scale-wise logging
                'tf_logits': logits_tf,
                'tf_gt_indices': gt_indices_tf,
                'tf_loss_mask': loss_mask_tf,
                'tf_scales': curr_patch_nums,
                # No SF data
                'sf_logits': None,
                'sf_gt_indices': None,
                'sf_loss_mask': None,
                'sf_scales': [],
            }

        return total_loss, log_info

    def train_step_hybrid_tf_sf(
        self, inp_B3HW: FTen, label_B: Union[ITen, FTen], curr_patch_nums: List[int], 
        mask_ratio: float, prog_si: int, prog_wp_it: float,
        sf_cfg_scale: float = 1.0, sf_top_k: int = 0, sf_top_p: float = 0.0,
        sf_use_sampling: bool = False
    ) -> Tuple[Ten, dict]:
        """
        Hybrid TF/SF training mode:
        - Use teacher forcing for the first n scales (default n=8)
        - Use student forcing for the later scales
        - When sf_cfg_scale > 1, includes unconditional samples for proper CFG
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size
        
        # Prepare labels for CFG if needed
        use_cfg = sf_cfg_scale > 1.0 and sf_use_sampling
        if use_cfg:
            # Double the batch with unconditional labels
            num_classes = self.scalear_wo_ddp.num_classes
            uncond_label = torch.full_like(label_B, fill_value=num_classes)
            label_B_cfg = torch.cat([label_B, uncond_label], dim=0)
        else:
            label_B_cfg = label_B
        
        # Get GT embeddings for all scales
        h_full = self.vae_local.encode_conti(inp_B3HW)
        gt_indices_by_scale = {}
        gt_quant_by_scale = {}
        
        for hw in curr_patch_nums:
            h_scale = F.interpolate(h_full.clone(), size=(hw, hw), mode='area')
            quant, _, log = self.vae_local.quantize(h_scale)
            indices = log[-1].view(B, -1)
            gt_indices_by_scale[hw] = indices
            gt_quant_by_scale[hw] = quant
        
        # Determine which scales use TF vs SF
        tf_scale_idx = self.hybrid_tf_scales
        tf_scales = [scale for scale in curr_patch_nums if scale <= tf_scale_idx]
        sf_scales = [scale for scale in curr_patch_nums if scale > tf_scale_idx]
        
        # Phase 1: Teacher Forcing for first n scales
        tf_inputs = []
        for i, curr_hw in enumerate(tf_scales):
            if i == 0 and curr_hw == 4:
                # First scale (4x4) with GT
                quant_4x4 = gt_quant_by_scale[4].reshape(B, self.Ch, -1).permute(0, 2, 1)
                if use_cfg:
                    # Double the input for CFG
                    quant_4x4 = quant_4x4.repeat(2, 1, 1)
                tf_inputs.append(quant_4x4)
            else:
                # Upsample GT from previous scale
                prev_hw = tf_scales[i-1]
                prev_gt = gt_quant_by_scale[prev_hw]
                prev_gt_2d = prev_gt.reshape(B, self.Ch, prev_hw, prev_hw)
                upsampled_gt = F.interpolate(prev_gt_2d, size=(curr_hw, curr_hw), mode='bicubic')
                upsampled_gt_quant = upsampled_gt.reshape(B, self.Ch, -1).permute(0, 2, 1)
                if use_cfg:
                    # Double the input for CFG
                    upsampled_gt_quant = upsampled_gt_quant.repeat(2, 1, 1)
                tf_inputs.append(upsampled_gt_quant)
        
        # Single forward pass for all TF scales
        tf_input_seq = torch.cat(tf_inputs, dim=1)
        with self.scalear_opt.amp_ctx:
            logits_tf, loss_mask_tf = self.scalear(
                label_B_cfg, tf_input_seq, tf_scales, randar_mask_ratio=mask_ratio
            )
        
        # Collect TF outputs
        all_logits = []
        all_loss_masks = []
        all_gt_indices = []
        
        # Add TF logits (only conditional part for loss)
        start_idx = 0
        for scale in tf_scales:
            end_idx = start_idx + scale**2
            if use_cfg:
                # Only use conditional part for loss
                all_logits.append(logits_tf[:B, start_idx:end_idx])
                all_loss_masks.append(loss_mask_tf[:B, start_idx:end_idx])
            else:
                all_logits.append(logits_tf[:, start_idx:end_idx])
                all_loss_masks.append(loss_mask_tf[:, start_idx:end_idx])
            all_gt_indices.append(gt_indices_by_scale[scale])
            start_idx = end_idx
        
        # Phase 2: Student Forcing for later scales (if any)
        if sf_scales:
            # Get the last TF scale's output to use as starting point for SF
            last_tf_scale = tf_scales[-1]
            last_tf_start = sum(s**2 for s in tf_scales[:-1])
            last_tf_end = last_tf_start + last_tf_scale**2
            
            with torch.no_grad():
                if sf_use_sampling:
                    # Use sampling for the last TF prediction that feeds into SF
                    last_tf_logits = logits_tf[:, last_tf_start:last_tf_end]
                    
                    if use_cfg:
                        # Apply CFG with gradual scaling (this is the last TF scale before SF starts)
                        ratio = (len(tf_scales) - 1) / (len(tf_scales) + len(sf_scales) - 1) if len(tf_scales) + len(sf_scales) > 1 else 0
                        t = sf_cfg_scale * 0.5 * (1 + ratio)
                        cond_logits = last_tf_logits[:B]
                        uncond_logits = last_tf_logits[B:]
                        last_tf_logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                    else:
                        last_tf_logits_cfg = last_tf_logits
                    
                    last_tf_pred = sample_with_top_k_top_p_(
                        last_tf_logits_cfg,
                        rng=self.scalear_wo_ddp.rng,
                        top_k=sf_top_k,
                        top_p=sf_top_p,
                        num_samples=1
                    )[:, :, 0]
                else:
                    if use_cfg:
                        # Take only conditional part for argmax
                        last_tf_pred = logits_tf[:B, last_tf_start:last_tf_end].argmax(dim=-1)
                    else:
                        last_tf_pred = logits_tf[:, last_tf_start:last_tf_end].argmax(dim=-1)
                last_tf_quant = self.vae_embedding[last_tf_pred]
            
            # Generate SF scales progressively
            sf_generated_quants = []
            
            for i, curr_hw in enumerate(sf_scales):
                if i == 0:
                    # First SF scale: upsample from last TF scale
                    prev_hw = last_tf_scale
                    prev_quant = last_tf_quant
                else:
                    # Later SF scales: upsample from previous SF scale
                    prev_hw = sf_scales[i-1]
                    prev_quant = sf_generated_quants[i-1]
                
                # Upsample previous scale
                prev_quant_2d = prev_quant.permute(0, 2, 1).reshape(B, self.Ch, prev_hw, prev_hw)
                upsampled = F.interpolate(prev_quant_2d, size=(curr_hw, curr_hw), mode='bicubic')
                upsampled_quant = upsampled.reshape(B, self.Ch, -1).permute(0, 2, 1)
                
                # Double the input for CFG if needed
                if use_cfg:
                    upsampled_quant_cfg = upsampled_quant.repeat(2, 1, 1)
                else:
                    upsampled_quant_cfg = upsampled_quant
                
                # For SF with blockwise causal mask, we only need the upsampled input
                # The model will handle causal attention internally
                with self.scalear_opt.amp_ctx:
                    # Access through module for DDP compatibility
                    logits_curr_full, loss_mask_curr_full = self.scalear(
                        label_B_cfg, upsampled_quant_cfg, [curr_hw],
                        randar_mask_ratio=0.5, student_forcing=True
                    )
                
                # Extract conditional part for loss calculation
                if use_cfg:
                    logits_curr = logits_curr_full[:B]
                    loss_mask_curr = loss_mask_curr_full[:B]
                else:
                    logits_curr = logits_curr_full
                    loss_mask_curr = loss_mask_curr_full
                
                all_logits.append(logits_curr)
                all_loss_masks.append(loss_mask_curr)
                all_gt_indices.append(gt_indices_by_scale[curr_hw])
                
                # Generate prediction for next scale
                with torch.no_grad():
                    if sf_use_sampling:
                        # Use sampling with top-k/top-p like in inference
                        if use_cfg:
                            # Apply proper CFG with gradual scaling
                            sf_scale_idx = len(tf_scales) + i  # Current position in total scale sequence
                            ratio = sf_scale_idx / (len(tf_scales) + len(sf_scales) - 1) if len(tf_scales) + len(sf_scales) > 1 else 0
                            t = sf_cfg_scale * 0.5 * (1 + ratio)
                            cond_logits = logits_curr_full[:B]
                            uncond_logits = logits_curr_full[B:]
                            logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                        else:
                            logits_cfg = logits_curr
                        
                        # Sample with top-k/top-p
                        pred_curr = sample_with_top_k_top_p_(
                            logits_cfg, 
                            rng=self.scalear_wo_ddp.rng,
                            top_k=sf_top_k, 
                            top_p=sf_top_p, 
                            num_samples=1
                        )[:, :, 0]  # Get the sampled indices
                    else:
                        # Use argmax (deterministic)
                        pred_curr = logits_curr.argmax(dim=-1)
                    
                    quant_curr = self.vae_embedding[pred_curr]
                    sf_generated_quants.append(quant_curr)
        
        # ============ VISUALIZATION CODE FOR HYBRID MODE (TEMPORARY) ============
        # Compare GT input (what TF would use) vs SF input at SF scales
        Flag = False
        # Only visualize occasionally to avoid overhead
        if Flag==True and sf_scales:  # Only if there are SF scales
            import matplotlib.pyplot as plt
            import os
            with torch.no_grad():
                # Visualize all samples in batch
                batch_size = min(B, 8)  # Limit to 8 samples max for memory
                
                # Visualize only SF scales
                num_vis_scales = min(4, len(sf_scales))  # Visualize up to 4 SF scales
                if num_vis_scales > 0:
                    # Create figure for each sample in batch
                    for batch_idx in range(batch_size):
                        # 5 rows: GT output, TF input (what GT would provide), SF input (upsampled), SF output, Difference
                        fig, axes = plt.subplots(5, num_vis_scales, figsize=(num_vis_scales * 3, 15))
                        
                        # Ensure axes is 2D
                        if num_vis_scales == 1:
                            axes = axes.reshape(-1, 1)
                        
                        # Select SF scales to visualize (evenly spaced)
                        vis_sf_indices = np.linspace(0, len(sf_scales) - 1, num_vis_scales, dtype=int)
                        
                        for col_idx, sf_idx in enumerate(vis_sf_indices):
                            scale = sf_scales[sf_idx]
                            
                            # 1. GT direct output at this scale (ground truth)
                            gt_indices = gt_indices_by_scale[scale][batch_idx:batch_idx+1]
                            gt_quant_vis = self.vae_embedding[gt_indices]
                            gt_quant_2d = gt_quant_vis.permute(0, 2, 1).reshape(1, self.Ch, scale, scale)
                            
                            # Decode GT
                            gt_decoded = self.vae_local.decode(gt_quant_2d)
                            gt_img = gt_decoded[0].clamp(0, 1).cpu()
                            
                            # 2. Create TF input at this scale (what TF would have used)
                            # Find the previous scale in the overall sequence
                            all_scale_idx = curr_patch_nums.index(scale)
                            if all_scale_idx > 0:
                                prev_scale_in_sequence = curr_patch_nums[all_scale_idx - 1]
                                # Get GT at previous scale and upsample
                                prev_gt_quant = gt_quant_by_scale[prev_scale_in_sequence][batch_idx:batch_idx+1]
                                prev_gt_2d = prev_gt_quant.reshape(1, self.Ch, prev_scale_in_sequence, prev_scale_in_sequence)
                                tf_input_quant_2d = F.interpolate(prev_gt_2d, size=(scale, scale), mode='bicubic')
                                
                                # Decode TF input
                                tf_input_decoded = self.vae_local.decode(tf_input_quant_2d)
                                tf_input_img = tf_input_decoded[0].clamp(0, 1).cpu()
                            else:
                                # This shouldn't happen for SF scales, but handle it
                                tf_input_img = gt_img
                            
                            # 3. Get the SF input (upsampled from model's previous prediction)
                            if sf_idx == 0:
                                # First SF scale: upsampled from last TF scale
                                prev_scale = last_tf_scale
                                # Get the last TF prediction
                                last_tf_start = sum(s**2 for s in tf_scales[:-1])
                                last_tf_end = last_tf_start + prev_scale**2
                                prev_indices = logits_tf[batch_idx:batch_idx+1, last_tf_start:last_tf_end].argmax(dim=-1)
                                prev_quant = self.vae_embedding[prev_indices]
                            else:
                                # Later SF scales: upsampled from previous SF scale
                                prev_scale = sf_scales[sf_idx-1]
                                prev_quant = sf_generated_quants[sf_idx-1][batch_idx:batch_idx+1]
                            
                            # Upsample previous scale to current scale (SF input)
                            prev_quant_2d = prev_quant.permute(0, 2, 1).reshape(1, self.Ch, prev_scale, prev_scale)
                            sf_input_quant_2d = F.interpolate(prev_quant_2d, size=(scale, scale), mode='bicubic')
                            
                            # Decode SF input
                            sf_input_decoded = self.vae_local.decode(sf_input_quant_2d)
                            sf_input_img = sf_input_decoded[0].clamp(0, 1).cpu()
                            
                            # 4. SF output at this scale
                            sf_quant_vis = sf_generated_quants[sf_idx][batch_idx:batch_idx+1]
                            sf_quant_2d = sf_quant_vis.permute(0, 2, 1).reshape(1, self.Ch, scale, scale)
                            
                            # Decode SF output
                            sf_decoded = self.vae_local.decode(sf_quant_2d)
                            sf_img = sf_decoded[0].clamp(0, 1).cpu()
                            
                            # 5. Difference maps
                            diff_gt_sf = torch.abs(gt_img - sf_img).mean(dim=0)  # GT vs SF output
                            diff_tf_sf_input = torch.abs(tf_input_img - sf_input_img).mean(dim=0)  # TF input vs SF input
                            
                            # Plot
                            axes[0, col_idx].imshow(gt_img.permute(1, 2, 0))
                            axes[0, col_idx].set_title(f'GT Output {scale}x{scale}')
                            axes[0, col_idx].axis('off')
                            
                            axes[1, col_idx].imshow(tf_input_img.permute(1, 2, 0))
                            axes[1, col_idx].set_title(f'TF Input (GT upsampled)')
                            axes[1, col_idx].axis('off')
                            
                            axes[2, col_idx].imshow(sf_input_img.permute(1, 2, 0))
                            axes[2, col_idx].set_title(f'SF Input (from {prev_scale}x{prev_scale})')
                            axes[2, col_idx].axis('off')
                            
                            axes[3, col_idx].imshow(sf_img.permute(1, 2, 0))
                            axes[3, col_idx].set_title(f'SF Output {scale}x{scale}')
                            axes[3, col_idx].axis('off')
                            
                            # Difference visualization
                            im = axes[4, col_idx].imshow(diff_tf_sf_input, cmap='hot', vmin=0, vmax=0.5)
                            axes[4, col_idx].set_title(f'|TF-SF input| (max={diff_tf_sf_input.max():.3f})')
                            axes[4, col_idx].axis('off')
                            plt.colorbar(im, ax=axes[4, col_idx], fraction=0.046, pad=0.04)
                    
                        # Add step info for this batch sample
                        cfg_info = f"CFG={sf_cfg_scale:.1f}, top_k={sf_top_k}, top_p={sf_top_p:.2f}, sampling={'ON' if sf_use_sampling else 'OFF'}"
                        fig.suptitle(f'SF Scales Visualization - Step {prog_si} - Sample {batch_idx+1}/{batch_size}\nTF scales: {tf_scales}, SF scales: {sf_scales}\n{cfg_info}')
                        plt.tight_layout()
                        
                        # Save figure
                        vis_dir = 'hybrid_vis_sf'
                        os.makedirs(vis_dir, exist_ok=True)
                        plt.savefig(f'{vis_dir}/sf_step_{prog_si:06d}_sample_{batch_idx:02d}.png', dpi=100, bbox_inches='tight')
                        plt.close(fig)
                    
                    print(f"[VISUALIZATION] Saved SF scales comparison for {batch_size} samples to {vis_dir}/sf_step_{prog_si:06d}_sample_*.png")
        # ============ END VISUALIZATION CODE ============
        
        # Concatenate all outputs
        logits_BLV = torch.cat(all_logits, dim=1)
        loss_mask = torch.cat(all_loss_masks, dim=1)
        gt_BL = torch.cat(all_gt_indices, dim=1)
        
        # Compute total loss and separate TF and SF losses for logging
        with self.scalear_opt.amp_ctx:
            loss_per_tok = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            loss_per_tok = loss_per_tok * loss_mask.to(loss_per_tok.dtype)
            tokens_per_sample = loss_mask.sum(dim=1).to(loss_per_tok.dtype)
            total_loss = (loss_per_tok.sum(dim=1) / tokens_per_sample).mean()
            
            # Teacher forcing tokens
            tf_token_count = sum(s**2 for s in tf_scales)
            if tf_token_count > 0:
                tf_loss_masked = loss_per_tok[:, :tf_token_count]
                tf_tokens = loss_mask[:, :tf_token_count].sum(dim=1).to(loss_per_tok.dtype)
                self.last_teacher_loss = (tf_loss_masked.sum(dim=1) / tf_tokens).mean().item()
            else:
                self.last_teacher_loss = 0.0
            
            # Student forcing tokens
            if sf_scales:
                sf_loss_masked = loss_per_tok[:, tf_token_count:]
                sf_tokens = loss_mask[:, tf_token_count:].sum(dim=1).to(loss_per_tok.dtype)
                self.last_student_loss = (sf_loss_masked.sum(dim=1) / sf_tokens).mean().item()
            else:
                self.last_student_loss = 0.0
        
        # Prepare log info
        log_info = {
            'logits': logits_BLV,
            'gt_indices': gt_BL,
            'loss_mask': loss_mask,
            'mode': 'hybrid_tf_sf',
            'teacher_loss': self.last_teacher_loss,
            'student_loss': self.last_student_loss
        }
        
        return total_loss, log_info

    def gen_curr_patch_nums(self):
        """Generate patch numbers for ScaleAR training with randar_scale-based schedule"""
        # Get randar_scale from the model (the starting scale for RandAR)
        randar_scale = self.scalear_wo_ddp.randar_scale

        if random.random() < 0.05:
            # 5% chance: Use a standard set of scales
            if randar_scale <= 6:
                # If randar_scale <= 6, keep scales from randar_scale to 6
                base_scales = list(range(randar_scale, 7))  # e.g., if randar_scale=2: [2,3,4,5,6]
                curr_patch_nums = base_scales + [8, 10, 13, 16]
            else:
                # If randar_scale > 6, start from randar_scale
                curr_patch_nums = [randar_scale, 10, 13, 16] if randar_scale < 10 else [randar_scale, 13, 16]
        else:
            # 95% chance: New schedule
            if randar_scale <= 6:
                # Keep scales from randar_scale to 6
                base_scales = list(range(randar_scale, 7))  # e.g., if randar_scale=4: [4,5,6]
            else:
                # Start from randar_scale
                base_scales = [randar_scale]

            # Sample additional higher scales
            random_scales = []

            # Sample 1 from [7, 8, 9] if applicable
            if randar_scale < 7:
                scale_7_9 = random.sample([7, 8, 9], 1)
                random_scales.extend(scale_7_9)
            # elif randar_scale <= 9:
            #     # If randar_scale is 7, 8, or 9, sample from remaining ones
            #     remaining = [s for s in [7, 8, 9] if s > randar_scale]
            #     if remaining:
            #         random_scales.extend(random.sample(remaining, 1))

            # Sample 1 from [10, 11, 12] if applicable
            if randar_scale < 10:
                scale_10_12 = random.sample([10, 11, 12], 1)
                random_scales.extend(scale_10_12)
            elif randar_scale <= 12:
                remaining = [s for s in [10, 11, 12] if s > randar_scale]
                if remaining:
                    random_scales.extend(random.sample(remaining, 1))

            # Sample 1 from [13, 14, 15] if applicable
            if randar_scale < 13:
                scale_13_15 = random.sample([13, 14, 15], 1)
                random_scales.extend(scale_13_15)
            elif randar_scale <= 15:
                remaining = [s for s in [13, 14, 15] if s > randar_scale]
                if remaining:
                    random_scales.extend(random.sample(remaining, 1))

            curr_patch_nums = base_scales + sorted(random_scales) + [16]
        
        # Progressive scale dropping (never drop randar_scale or 16)
        x = random.random()

        # Only drop intermediate scales, keeping at least randar_scale and 16
        if x > 0.9 and len(curr_patch_nums) > 2:
            drop_index = random.choice(range(1, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.95 and len(curr_patch_nums) > 2:
            drop_index = random.choice(range(1, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.98 and len(curr_patch_nums) > 2:
            drop_index = random.choice(range(1, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.99 and len(curr_patch_nums) > 2:
            drop_index = random.choice(range(1, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        
        # Ensure total tokens don't exceed limit
        # ScaleAR is more memory efficient, but still enforce a reasonable limit
        total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        while total_lens > 800 and len(curr_patch_nums) > 2:
            # 16 + 25 + 36 + 64 + 100 + 169 + 256 = 666, :), zgz 2025.7.7
            # 16 + 25 + 36 + 81 + 12*12 + 15*15 + 16*16 = 783, zgz 2025.9.9
            # 1 + 4 + 9 + 16 + 25 + 36 + 81 + 12*12 + 15*15 + 16*16 = 797, zgz 2025.9.26
            # Drop scales from near the end, but keep the final scale (16)
            drop_index = random.choice(range(max(1, len(curr_patch_nums)-3), len(curr_patch_nums)-1))
            curr_patch_nums.pop(drop_index)
            total_lens = sum(pn ** 2 for pn in curr_patch_nums)
            
        return curr_patch_nums

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: UnifiedLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
        force_log: bool = False,  # Add parameter to force logging (e.g., for first 100 steps)
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.scalear.require_backward_grad_sync = stepping
        
        # Generate current patch numbers
        curr_patch_nums = self.gen_curr_patch_nums()
        
        # Sample mask ratio from truncated normal distribution
        mask_ratio = self.mask_ratio_generator.rvs(1)[0]

        # Determine effective training mode for mixed training
        if self.training_mode == 'mixed':
            # Alternate between teacher and student forcing based on global iteration
            effective_mode = 'teacher_forcing' if g_it % 2 == 0 else 'student_forcing'
        else:
            effective_mode = self.training_mode
        

        # Call the appropriate training function based on mode
        if effective_mode == 'teacher_forcing':
            total_loss, log_info = self.train_step_teacher_forcing(
                inp_B3HW, label_B, curr_patch_nums, mask_ratio
            )
        elif effective_mode == 'student_forcing':
            total_loss, log_info = self.train_step_student_forcing(
                inp_B3HW, label_B, curr_patch_nums, mask_ratio, prog_si, prog_wp_it,
                sf_cfg_scale=self.sf_cfg_scale,
                sf_top_k=self.sf_top_k,
                sf_top_p=self.sf_top_p,
                sf_use_sampling=self.sf_use_sampling
            )
        elif effective_mode == 'alternating':
            total_loss, log_info = self.train_step_alternating_tf_sf(
                inp_B3HW, label_B, curr_patch_nums, mask_ratio, prog_si, prog_wp_it,
                sf_cfg_scale=self.sf_cfg_scale,
                sf_top_k=self.sf_top_k,
                sf_top_p=self.sf_top_p,
                sf_use_sampling=self.sf_use_sampling
            )
        elif effective_mode == 'hybrid_tf_sf':
            total_loss, log_info = self.train_step_hybrid_tf_sf(
                inp_B3HW, label_B, self.default_inference_schedule, mask_ratio, prog_si, prog_wp_it,
                sf_cfg_scale=self.sf_cfg_scale,
                sf_top_k=self.sf_top_k,
                sf_top_p=self.sf_top_p,
                sf_use_sampling=self.sf_use_sampling
            )
        elif effective_mode == 'csfl':
            total_loss, log_info = self.train_step_csfl(
                inp_B3HW, label_B, curr_patch_nums, mask_ratio, prog_si, prog_wp_it,
                sf_cfg_scale=self.sf_cfg_scale,
                sf_top_k=self.sf_top_k,
                sf_top_p=self.sf_top_p,
                sf_use_sampling=self.sf_use_sampling
            )
        elif effective_mode == 'tf_then_single_sf':
            total_loss, log_info = self.train_step_tf_then_single_sf(
                inp_B3HW, label_B, curr_patch_nums, mask_ratio, prog_si, prog_wp_it,
                sf_cfg_scale=self.sf_cfg_scale,
                sf_top_k=self.sf_top_k,
                sf_top_p=self.sf_top_p,
                sf_use_sampling=self.sf_use_sampling
            )
        else:
            raise ValueError(f"Unknown effective training mode: {effective_mode} (original mode: {self.training_mode})")

        # backward
        grad_norm, scale_log2 = self.scalear_opt.backward_clip_step(loss=total_loss, stepping=stepping)
        
        # log
        log_interval = int(getattr(metric_lg, 'log_iters', 1))
        # Log if forced (first 100 steps) or at regular interval
        if force_log or it == 0 or it % log_interval == 0:
            # Extract logits and gt from log_info if available
            logits_BLV = log_info.get('logits', None)
            gt_BL = log_info.get('gt_indices', None)
            loss_mask = log_info.get('loss_mask', None)
            
            # Compute metrics based on available data
            mode = log_info.get('mode', 'unknown')
            
            if mode == 'csfl':
                # Special handling for csfl mode with separate TF and SF metrics
                tf_logits = log_info.get('tf_logits')
                tf_gt = log_info.get('tf_gt_indices')
                sf_logits = log_info.get('sf_logits')
                sf_gt = log_info.get('sf_gt_indices')
                tf_scales = log_info.get('tf_scales', [])
                sf_scales = log_info.get('sf_scales', [])
                
                # Overall metrics
                Lmean = total_loss.item()
                
                # TF scale-specific metrics
                scale_loss = {}
                scale_acc = {}
                
                if tf_logits is not None and tf_gt is not None:
                    tf_pred = tf_logits.data.argmax(dim=-1)
                    tf_acc_mean = (tf_pred == tf_gt).float().mean().item() * 100
                    
                    cur_idx = 0
                    for i, cur_scale in enumerate(tf_scales):
                        curr_l = cur_scale ** 2
                        curr_gt = tf_gt[:, cur_idx:cur_idx+curr_l]
                        curr_logit = tf_logits.data[:, cur_idx:cur_idx+curr_l]
                        curr_pred = tf_pred[:, cur_idx:cur_idx+curr_l]
                        
                        if cur_scale == 4:
                            # RandAR metrics for TF
                            scale_loss[f"randar_4x4_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                            scale_acc[f"randar_4x4_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                        else:
                            # VAR metrics for TF
                            scale_loss[f"var_scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                            scale_acc[f"var_scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                        
                        cur_idx += curr_l
                
                acc_mean = tf_acc_mean

                # SF scale-specific metrics
                if sf_logits is not None and sf_gt is not None and len(sf_scales) > 0:
                    sf_pred = sf_logits.data.argmax(dim=-1)
                    sf_acc_mean = (sf_pred == sf_gt).float().mean().item() * 100
                    
                    cur_idx = 0
                    for i, cur_scale in enumerate(sf_scales):
                        curr_l = cur_scale ** 2
                        curr_gt = sf_gt[:, cur_idx:cur_idx+curr_l]
                        curr_logit = sf_logits.data[:, cur_idx:cur_idx+curr_l]
                        curr_pred = sf_pred[:, cur_idx:cur_idx+curr_l]
                        
                        # SF VAR metrics (SF doesn't have 4x4)
                        scale_loss[f"sf_var_scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                        scale_acc[f"sf_var_scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                        
                        cur_idx += curr_l
                
                # Set tail metrics (use largest scale available)
                if len(tf_scales) > 0 and tf_scales[-1] == 16:
                    Ltail = scale_loss.get(f"var_scale_16_loss", -1)
                    acc_tail = scale_acc.get(f"var_scale_16_acc", -1)
                elif len(sf_scales) > 0 and sf_scales[-1] == 16:
                    Ltail = scale_loss.get(f"sf_var_scale_16_loss", -1)
                    acc_tail = scale_acc.get(f"sf_var_scale_16_acc", -1)
                else:
                    Ltail = -1
                    acc_tail = -1

            elif mode == 'tf_then_single_sf':
                # Special handling for tf_then_single_sf mode with separate TF and SF metrics
                tf_logits = log_info.get('tf_logits')
                tf_gt = log_info.get('tf_gt_indices')
                sf_logits = log_info.get('sf_logits')
                sf_gt = log_info.get('sf_gt_indices')
                tf_scales = log_info.get('tf_scales', [])
                sf_scales = log_info.get('sf_scales', [])

                # Overall metrics
                Lmean = total_loss.item()

                # TF scale-specific metrics
                scale_loss = {}
                scale_acc = {}

                if tf_logits is not None and tf_gt is not None:
                    tf_pred = tf_logits.data.argmax(dim=-1)
                    tf_acc_mean = (tf_pred == tf_gt).float().mean().item() * 100

                    cur_idx = 0
                    for i, cur_scale in enumerate(tf_scales):
                        curr_l = cur_scale ** 2
                        curr_gt = tf_gt[:, cur_idx:cur_idx+curr_l]
                        curr_logit = tf_logits.data[:, cur_idx:cur_idx+curr_l]
                        curr_pred = tf_pred[:, cur_idx:cur_idx+curr_l]

                        if cur_scale == 4:
                            # RandAR metrics for TF
                            scale_loss[f"randar_4x4_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                            scale_acc[f"randar_4x4_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                        else:
                            # VAR metrics for TF
                            scale_loss[f"var_scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                            scale_acc[f"var_scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100

                        cur_idx += curr_l

                acc_mean = tf_acc_mean

                # SF scale-specific metrics (only one scale)
                if sf_logits is not None and sf_gt is not None and len(sf_scales) > 0:
                    sf_pred = sf_logits.data.argmax(dim=-1)

                    cur_scale = sf_scales[0]  # Only one scale for SF
                    curr_gt = sf_gt
                    curr_logit = sf_logits.data
                    curr_pred = sf_pred

                    # SF VAR metrics
                    scale_loss[f"sf_var_scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                    scale_acc[f"sf_var_scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100

                # Set tail metrics (use largest scale available)
                if len(tf_scales) > 0 and tf_scales[-1] == 16:
                    Ltail = scale_loss.get(f"var_scale_16_loss", -1)
                    acc_tail = scale_acc.get(f"var_scale_16_acc", -1)
                elif len(sf_scales) > 0 and sf_scales[-1] == 16:
                    Ltail = scale_loss.get(f"sf_var_scale_16_loss", -1)
                    acc_tail = scale_acc.get(f"sf_var_scale_16_acc", -1)
                else:
                    Ltail = -1
                    acc_tail = -1

            elif logits_BLV is not None and gt_BL is not None:
                # Regular logging for modes that provide logits
                pred_BL = logits_BLV.data.argmax(dim=-1)
                Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
                acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
                
                # Log scale-specific metrics
                scale_loss = {}
                scale_acc = {}
                cur_idx = 0
                
                for i in range(len(curr_patch_nums)):
                    cur_scale = curr_patch_nums[i]
                    curr_l = cur_scale ** 2
                    curr_gt = gt_BL[:, cur_idx:cur_idx+curr_l]
                    curr_logit = logits_BLV.data[:, cur_idx:cur_idx+curr_l]
                    curr_pred = pred_BL[:, cur_idx:cur_idx+curr_l]
                    
                    if cur_scale == 4:
                        # RandAR metrics
                        scale_loss[f"randar_4x4_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                        scale_acc[f"randar_4x4_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                    else:
                        # VAR metrics
                        scale_loss[f"var_scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                        scale_acc[f"var_scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                    
                    cur_idx += curr_l
                
                Ltail = scale_loss.get(f"var_scale_16_loss", -1)
                acc_tail = scale_acc.get(f"var_scale_16_acc", -1)
            else:
                # Fallback for modes without detailed logits
                Lmean = total_loss.item()
                acc_mean = 0.0
                Ltail = -1
                acc_tail = -1
                scale_loss = {}
                scale_acc = {}
            
            if isinstance(grad_norm, torch.Tensor):
                grad_norm_val = grad_norm.item()
            else:
                grad_norm_val = float(grad_norm) if grad_norm is not None else 0.0
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_val)
            tb_lg.update(head='ScaleAR_loss', step=g_it, L_mean=Lmean, acc_mean=acc_mean)
            
            # Log mode-specific metrics based on log_info
            mode = log_info.get('mode', 'unknown')
            
            if mode == 'alternating' and 'teacher_loss' in log_info:
                tb_lg.update(head='ScaleAR_alternating', step=g_it, 
                            teacher_loss=log_info['teacher_loss'],
                            student_loss=log_info['student_loss'],
                            mixed_loss=total_loss.item())
            
            elif mode == 'hybrid_tf_sf' and 'teacher_loss' in log_info:
                tb_lg.update(head='ScaleAR_hybrid', step=g_it, 
                            teacher_loss=log_info['teacher_loss'],
                            student_loss=log_info['student_loss'],
                            mixed_loss=total_loss.item())
            
            elif mode == 'csfl' and 'teacher_loss' in log_info:
                tb_lg.update(head='ScaleAR_csfl', step=g_it,
                            teacher_loss=log_info['teacher_loss'],
                            student_loss=log_info['student_loss'],
                            total_loss=log_info['total_loss'])

            elif mode == 'tf_then_single_sf' and 'teacher_loss' in log_info:
                tb_lg.update(head='ScaleAR_tf_then_single_sf', step=g_it,
                            teacher_loss=log_info['teacher_loss'],
                            student_loss=log_info['student_loss'],
                            total_loss=log_info['total_loss'],
                            selected_scale=log_info.get('selected_scale', -1),
                            selected_scale_idx=log_info.get('selected_scale_idx', -1))

            tb_lg.update(head='ScaleAR_scale_loss', step=g_it, **scale_loss)
            tb_lg.update(head='ScaleAR_scale_acc', step=g_it, **scale_acc)
        
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'training_mode': self.training_mode,
            'sigma': self.sigma,
            'hybrid_tf_scales': self.hybrid_tf_scales,
            'sf_cfg_scale': self.sf_cfg_scale,
            'sf_top_k': self.sf_top_k,
            'sf_top_p': self.sf_top_p,
            'sf_use_sampling': self.sf_use_sampling,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('scalear_wo_ddp', 'vae_local', 'scalear_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('scalear_wo_ddp', 'vae_local', 'scalear_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[ScaleARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[ScaleARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[ScaleAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)