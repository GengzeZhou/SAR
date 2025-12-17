import time
from typing import List, Optional, Tuple, Union

import torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import FlexVAR
from models.helpers import sample_with_top_k_top_p_
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger
from utils.logger import UnifiedLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, var_wo_ddp: FlexVAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
        training_mode: str = 'teacher_forcing',
        sigma: float = 0.01,  # SF loss weight for tf_then_single_sf training mode
        sf_cfg_scale: float = 1.2,  # CFG scale for SF sampling
        sf_top_k: int = 900,  # Top-k for SF sampling
        sf_top_p: float = 0.96,  # Top-p for SF sampling
        sf_use_sampling: bool = True,  # Whether to use sampling vs argmax for SF
    ):
        super(VARTrainer, self).__init__()

        self.var, self.vae_local = var, vae_local
        self.var_wo_ddp: FlexVAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.vae_embedding = F.normalize(self.vae_local.quantize.embedding.weight, p=2, dim=-1).to(device)
        self.length = self.vae_embedding.shape[0] - 1
        self.Ch = self.vae_embedding.shape[1]

        # Training mode and loss tracking
        self.training_mode = training_mode
        self.sigma = sigma
        # SF sampling parameters
        self.sf_cfg_scale = sf_cfg_scale
        self.sf_top_k = sf_top_k
        self.sf_top_p = sf_top_p
        self.sf_use_sampling = sf_use_sampling
        self.last_teacher_loss = 0.0
        self.last_student_loss = 0.0
        self.last_csfl_total_loss = 0.0
        self.last_tf_single_sf_total_loss = 0.0

    def encode_var_wo_firstL(self, x, curr_patch_nums):
        h = self.vae_local.encode_conti(x)
        # return quant_z.reshape(quant_z.shape[0], quant_z.shape[1], -1), indices
        all_indices = []
        all_quant = []
        end = len(curr_patch_nums) -1
        for num in range(len(curr_patch_nums)):
            curr_hw = curr_patch_nums[num]
            _h = F.interpolate(h.clone(), size=(curr_hw, curr_hw), mode='area')
            quant, _, log = self.vae_local.quantize(_h)
            indices = log[-1].view(quant.shape[0], -1)
            all_indices.append(indices)
            if not num == end:
                next_hw = curr_patch_nums[num+1]
                
                next_quant = F.interpolate(quant, size=(next_hw, next_hw), mode='bicubic')
                next_quant = next_quant.reshape(quant.shape[0], quant.shape[1], -1)
                all_quant.append(next_quant)

        all_quant = torch.cat(all_quant, dim = 2).permute(0,2,1)
        all_indices = torch.cat(all_indices, dim = 1)
        # if random.random() < 0.1:
        #     bs, length = all_indices.shape
        #     random_ind = torch.randint(low=0, high=self.length, size=(bs, length//20), dtype=torch.int64)
        #     random_quant = self.vae_embedding[random_ind]
        #     random_indices = torch.randint(low=0, high=all_quant.shape[1]-1, size=(bs, length//20)).to(random_quant.device)

        #     index = random_indices.unsqueeze(-1).expand(-1, -1, self.Ch)
        #     all_quant.scatter_(1, index, random_quant)
        return all_quant, all_indices


    def gen_curr_patch_nums(self, ):
        if random.random() < 0.05:
            curr_patch_nums = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        else:
            random_numbers = random.sample(range(3, 11), 5) + random.sample(range(11, 16), 2)
            # random_numbers = random.sample(range(2, 16), 8) 
            random_numbers.sort()
            curr_patch_nums = [1, 2] + random_numbers + [16]

        # drop scales    
        x = random.random()
        if x > 0.9:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.95:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.98:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)
        if x > 0.99:
            drop_index = random.choice(range(2, len(curr_patch_nums) - 1))
            curr_patch_nums.pop(drop_index)

        total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        while total_lens > 680:
            drop_index = random.choice(range(len(curr_patch_nums)-4, len(curr_patch_nums)))
            curr_patch_nums.pop(drop_index)
            total_lens = sum(pn ** 2 for pn in curr_patch_nums)
        return curr_patch_nums

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: UnifiedLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # Route to appropriate training method based on training_mode
        if self.training_mode == 'tf_then_single_sf':
            return self.train_step_tf_then_single_sf_gt(
                it, g_it, stepping, metric_lg, tb_lg, inp_B3HW, label_B, prog_si, prog_wp_it
            )
        elif self.training_mode == 'csfl':
            return self.train_step_tf_then_single_sf_tf(
                it, g_it, stepping, metric_lg, tb_lg, inp_B3HW, label_B, prog_si, prog_wp_it
            )
        # Default: teacher_forcing mode
        # if progressive training
        # self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        # if self.last_prog_si != prog_si:
        #     if self.last_prog_si != -1: self.first_prog = False
        #     self.last_prog_si = prog_si
        #     self.prog_it = 0
        # self.prog_it += 1
        # prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        # if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        # if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog

        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        curr_patch_nums = self.gen_curr_patch_nums()
        quant_z, gt_BL = self.encode_var_wo_firstL(inp_B3HW, curr_patch_nums)


        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(label_B, quant_z, curr_patch_nums)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            # if prog_si >= 0:    # in progressive training
            #     bg, ed = self.begin_ends[prog_si]
            #     assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
            #     lw = self.loss_weight[:, :ed].clone()
            #     lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            # else:               # not in progressive training
            #     lw = self.loss_weight
            # loss = loss.mul(lw).sum(dim=-1).mean()

            L = sum(pn * pn for pn in curr_patch_nums)
            lw = torch.ones(1, L, device=self.loss_weight.device) / L

            loss = loss.mul(lw).sum(dim=-1).mean()

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it % metric_lg.log_iters == 0:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                cur_idx = 0
                scale_loss = {}
                scale_acc = {}
                for i in range(len(curr_patch_nums)):
                    cur_scale = curr_patch_nums[i]
                    curr_l = curr_patch_nums[i] ** 2
                    curr_gt = gt_BL[:, cur_idx:cur_idx+curr_l]
                    curr_logit = logits_BLV.data[:, cur_idx:cur_idx+curr_l]
                    curr_pred = pred_BL[:, cur_idx:cur_idx+curr_l]
                    scale_loss[f"scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                    scale_acc[f"scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                    cur_idx += curr_l
                Ltail = scale_loss[f"scale_{cur_scale}_loss"]
                acc_tail = scale_acc[f"scale_{cur_scale}_acc"]
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
            tb_lg.update(head='AR_loss', step=g_it, L_mean=Lmean, acc_mean=acc_mean)
            tb_lg.update(head='AR_scale_loss', step=g_it, **scale_loss)
            tb_lg.update(head='AR_scale_acc', step=g_it, **scale_acc)
        
        return grad_norm, scale_log2

    def train_step_tf_then_single_sf_gt(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: UnifiedLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """
        TF-then-Single-SF training mode (SF loss vs GT):
        Pass 1: Teacher forcing with all scales using GT inputs -> compute TF loss
        Pass 2: Student forcing on ONE randomly selected scale -> compute SF loss vs GT
        Returns: combined_loss
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        curr_patch_nums = self.gen_curr_patch_nums()

        # Prepare labels for CFG if needed
        use_cfg = self.sf_cfg_scale > 1.0 and self.sf_use_sampling
        if use_cfg:
            num_classes = self.var_wo_ddp.num_classes
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
        # Match encode_var_wo_firstL logic: upsample from scale i→i+1, excluding last scale
        # First scale is handled by SOS tokens in forward(), not included in input
        tf_inputs = []
        for i in range(len(curr_patch_nums) - 1):  # Exclude last scale
            curr_hw = curr_patch_nums[i]
            next_hw = curr_patch_nums[i + 1]

            # Upsample GT from current scale to next scale
            curr_gt = gt_quant_by_scale[curr_hw]
            curr_gt_2d = curr_gt.reshape(B, self.Ch, curr_hw, curr_hw)
            upsampled_gt = F.interpolate(curr_gt_2d, size=(next_hw, next_hw), mode='bicubic')
            upsampled_gt_quant = upsampled_gt.reshape(B, self.Ch, -1).permute(0, 2, 1)
            tf_inputs.append(upsampled_gt_quant)

        # Single forward pass for all TF scales
        tf_input_seq = torch.cat(tf_inputs, dim=1)
        # Double inputs for CFG if needed
        if use_cfg:
            tf_input_seq_cfg = tf_input_seq.repeat(2, 1, 1)
        else:
            tf_input_seq_cfg = tf_input_seq

        with self.var_opt.amp_ctx:
            logits_tf_full = self.var(label_B_cfg, tf_input_seq_cfg, curr_patch_nums)

        # Extract conditional part for loss
        if use_cfg:
            logits_tf = logits_tf_full[:B]
        else:
            logits_tf = logits_tf_full

        # Store TF data for scale-wise loss calculation (all scales for supervision)
        gt_indices_tf = torch.cat([gt_indices_by_scale[scale] for scale in curr_patch_nums], dim=1)

        # Compute TF loss
        with self.var_opt.amp_ctx:
            loss_tf = self.train_loss(logits_tf.view(-1, V), gt_indices_tf.view(-1)).view(B, -1)
            L = sum(pn * pn for pn in curr_patch_nums)
            lw = torch.ones(1, L, device=self.loss_weight.device) / L
            tf_loss = loss_tf.mul(lw).sum(dim=-1).mean()
            self.last_teacher_loss = tf_loss.item()

        # Sample from each scale's output to construct SF inputs
        sampled_quants = []
        start_idx = 0
        for i, scale in enumerate(curr_patch_nums):
            end_idx = start_idx + scale**2

            with torch.no_grad():
                if self.sf_use_sampling:
                    if use_cfg:
                        # Apply CFG with gradual scaling
                        ratio = i / (len(curr_patch_nums) - 1) if len(curr_patch_nums) > 1 else 0
                        t = self.sf_cfg_scale * 0.5 * (1 + ratio)
                        cond_logits = logits_tf_full[:B, start_idx:end_idx]
                        uncond_logits = logits_tf_full[B:, start_idx:end_idx]
                        logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                    else:
                        logits_cfg = logits_tf[:, start_idx:end_idx]

                    pred_indices = sample_with_top_k_top_p_(
                        logits_cfg,
                        rng=self.var_wo_ddp.rng,
                        top_k=self.sf_top_k,
                        top_p=self.sf_top_p,
                        num_samples=1
                    )[:, :, 0]
                else:
                    pred_indices = logits_tf[:, start_idx:end_idx].argmax(dim=-1)

                sampled_quant = self.vae_embedding[pred_indices]
                sampled_quants.append(sampled_quant)

            start_idx = end_idx

        # ======= Pass 2: Student Forcing on ONE randomly selected scale =======
        if len(curr_patch_nums) > 1:
            # Randomly select one scale index (excluding the last scale since it has no "next")
            # selected_idx = random.randint(0, len(curr_patch_nums) - 2)
            selected_idx = len(curr_patch_nums) - 2

            curr_hw = curr_patch_nums[selected_idx]
            next_hw = curr_patch_nums[selected_idx + 1]

            # Use sampled output from selected scale
            sampled_quant_curr = sampled_quants[selected_idx]

            # Reshape and upsample to next scale
            sampled_2d = sampled_quant_curr.permute(0, 2, 1).reshape(B, self.Ch, curr_hw, curr_hw)
            upsampled = F.interpolate(sampled_2d, size=(next_hw, next_hw), mode='bicubic')
            upsampled_quant = upsampled.reshape(B, self.Ch, -1).permute(0, 2, 1)

            # Forward pass with SF input - use student_forcing mode
            with self.var_opt.amp_ctx:
                logits_sf = self.var(label_B, upsampled_quant, [1, next_hw], student_forcing=True)

            # Get GT indices for the selected next scale
            gt_indices_sf = gt_indices_by_scale[next_hw]

            # Compute SF loss vs GT
            with self.var_opt.amp_ctx:
                loss_sf = self.train_loss(logits_sf.view(-1, V), gt_indices_sf.view(-1)).view(B, -1)
                sf_loss = loss_sf.mean()
                self.last_student_loss = sf_loss.item()

            # Combine TF and SF losses
            total_loss = tf_loss + self.sigma * sf_loss
            self.last_tf_single_sf_total_loss = total_loss.item()
        else:
            # No SF pass needed (only one scale)
            total_loss = tf_loss
            self.last_student_loss = 0.0
            self.last_tf_single_sf_total_loss = tf_loss.item()

        # Backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=total_loss, stepping=stepping)

        # Log
        pred_BL = logits_tf.data.argmax(dim=-1)
        if it == 0 or it % metric_lg.log_iters == 0:
            Lmean = self.val_loss(logits_tf.data.view(-1, V), gt_indices_tf.view(-1)).item()
            acc_mean = (pred_BL == gt_indices_tf).float().mean().item() * 100

            cur_idx = 0
            scale_loss = {}
            scale_acc = {}
            for i in range(len(curr_patch_nums)):
                cur_scale = curr_patch_nums[i]
                curr_l = curr_patch_nums[i] ** 2
                curr_gt = gt_indices_tf[:, cur_idx:cur_idx+curr_l]
                curr_logit = logits_tf.data[:, cur_idx:cur_idx+curr_l]
                curr_pred = pred_BL[:, cur_idx:cur_idx+curr_l]
                scale_loss[f"scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                scale_acc[f"scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                cur_idx += curr_l
            Ltail = scale_loss[f"scale_{cur_scale}_loss"]
            acc_tail = scale_acc[f"scale_{cur_scale}_acc"]

            grad_norm_val = grad_norm.item()
            metric_lg.update(
                Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_val,
                TF_loss=self.last_teacher_loss, SF_loss=self.last_student_loss, Total=self.last_tf_single_sf_total_loss
            )
            tb_lg.update(head='AR_loss', step=g_it, L_mean=Lmean, acc_mean=acc_mean)
            tb_lg.update(head='AR_scale_loss', step=g_it, **scale_loss)
            tb_lg.update(head='AR_scale_acc', step=g_it, **scale_acc)
            tb_lg.update(head='AR_tf_single_sf', step=g_it,
                        teacher_loss=self.last_teacher_loss,
                        student_loss=self.last_student_loss,
                        total_loss=self.last_tf_single_sf_total_loss)

        return grad_norm, scale_log2

    def train_step_tf_then_single_sf_tf(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: UnifiedLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """
        TF-then-Single-SF training mode (SF loss vs TF output):
        Pass 1: Teacher forcing with all scales using GT inputs -> compute TF loss
        Pass 2: Student forcing on ONE randomly selected scale -> compute SF loss vs TF predictions
        Returns: combined_loss
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        curr_patch_nums = self.gen_curr_patch_nums()

        # Prepare labels for CFG if needed
        use_cfg = self.sf_cfg_scale > 1.0 and self.sf_use_sampling
        if use_cfg:
            num_classes = self.var_wo_ddp.num_classes
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
        # Match encode_var_wo_firstL logic: upsample from scale i→i+1, excluding last scale
        # First scale is handled by SOS tokens in forward(), not included in input
        tf_inputs = []
        for i in range(len(curr_patch_nums) - 1):  # Exclude last scale
            curr_hw = curr_patch_nums[i]
            next_hw = curr_patch_nums[i + 1]

            # Upsample GT from current scale to next scale
            curr_gt = gt_quant_by_scale[curr_hw]
            curr_gt_2d = curr_gt.reshape(B, self.Ch, curr_hw, curr_hw)
            upsampled_gt = F.interpolate(curr_gt_2d, size=(next_hw, next_hw), mode='bicubic')
            upsampled_gt_quant = upsampled_gt.reshape(B, self.Ch, -1).permute(0, 2, 1)
            tf_inputs.append(upsampled_gt_quant)

        # Single forward pass for all TF scales
        tf_input_seq = torch.cat(tf_inputs, dim=1)
        # Double inputs for CFG if needed
        if use_cfg:
            tf_input_seq_cfg = tf_input_seq.repeat(2, 1, 1)
        else:
            tf_input_seq_cfg = tf_input_seq

        with self.var_opt.amp_ctx:
            logits_tf_full = self.var(label_B_cfg, tf_input_seq_cfg, curr_patch_nums)

        # Extract conditional part for loss
        if use_cfg:
            logits_tf = logits_tf_full[:B]
        else:
            logits_tf = logits_tf_full

        # Store TF data for scale-wise loss calculation (all scales for supervision)
        gt_indices_tf = torch.cat([gt_indices_by_scale[scale] for scale in curr_patch_nums], dim=1)

        # Compute TF loss
        with self.var_opt.amp_ctx:
            loss_tf = self.train_loss(logits_tf.view(-1, V), gt_indices_tf.view(-1)).view(B, -1)
            L = sum(pn * pn for pn in curr_patch_nums)
            lw = torch.ones(1, L, device=self.loss_weight.device) / L
            tf_loss = loss_tf.mul(lw).sum(dim=-1).mean()
            self.last_teacher_loss = tf_loss.item()

        # Sample from each scale's output to construct SF inputs
        sampled_quants = []
        tf_pred_indices_by_scale = {}
        start_idx = 0
        for i, scale in enumerate(curr_patch_nums):
            end_idx = start_idx + scale**2

            with torch.no_grad():
                if self.sf_use_sampling:
                    if use_cfg:
                        # Apply CFG with gradual scaling
                        ratio = i / (len(curr_patch_nums) - 1) if len(curr_patch_nums) > 1 else 0
                        t = self.sf_cfg_scale * 0.5 * (1 + ratio)
                        cond_logits = logits_tf_full[:B, start_idx:end_idx]
                        uncond_logits = logits_tf_full[B:, start_idx:end_idx]
                        logits_cfg = (1 + t) * cond_logits - t * uncond_logits
                    else:
                        logits_cfg = logits_tf[:, start_idx:end_idx]

                    pred_indices = sample_with_top_k_top_p_(
                        logits_cfg,
                        rng=self.var_wo_ddp.rng,
                        top_k=self.sf_top_k,
                        top_p=self.sf_top_p,
                        num_samples=1
                    )[:, :, 0]
                else:
                    pred_indices = logits_tf[:, start_idx:end_idx].argmax(dim=-1)

                tf_pred_indices_by_scale[scale] = pred_indices
                sampled_quant = self.vae_embedding[pred_indices]
                sampled_quants.append(sampled_quant)

            start_idx = end_idx

        # ======= Pass 2: Student Forcing on ONE randomly selected scale =======
        if len(curr_patch_nums) > 1:
            # Randomly select one scale index (excluding the last scale since it has no "next")
            # selected_idx = random.randint(0, len(curr_patch_nums) - 2)
            selected_idx = len(curr_patch_nums) - 2

            curr_hw = curr_patch_nums[selected_idx]
            next_hw = curr_patch_nums[selected_idx + 1]

            # Use sampled output from selected scale
            sampled_quant_curr = sampled_quants[selected_idx]

            # Reshape and upsample to next scale
            sampled_2d = sampled_quant_curr.permute(0, 2, 1).reshape(B, self.Ch, curr_hw, curr_hw)
            upsampled = F.interpolate(sampled_2d, size=(next_hw, next_hw), mode='bicubic')
            upsampled_quant = upsampled.reshape(B, self.Ch, -1).permute(0, 2, 1)

            # Forward pass with SF input - use student_forcing mode
            with self.var_opt.amp_ctx:
                logits_sf = self.var(label_B, upsampled_quant, [1, next_hw], student_forcing=True)

            # Get TF predictions for the selected next scale (as target instead of GT)
            tf_pred_indices_sf = tf_pred_indices_by_scale[next_hw]

            # Compute SF loss vs TF predictions
            with self.var_opt.amp_ctx:
                loss_sf = self.train_loss(logits_sf.view(-1, V), tf_pred_indices_sf.view(-1)).view(B, -1)
                sf_loss = loss_sf.mean()
                self.last_student_loss = sf_loss.item()

            # Combine TF and SF losses
            total_loss = tf_loss + self.sigma * sf_loss
            self.last_csfl_total_loss = total_loss.item()
        else:
            # No SF pass needed (only one scale)
            total_loss = tf_loss
            self.last_student_loss = 0.0
            self.last_csfl_total_loss = tf_loss.item()

        # Backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=total_loss, stepping=stepping)

        # Log
        pred_BL = logits_tf.data.argmax(dim=-1)
        if it == 0 or it % metric_lg.log_iters == 0:
            Lmean = self.val_loss(logits_tf.data.view(-1, V), gt_indices_tf.view(-1)).item()
            acc_mean = (pred_BL == gt_indices_tf).float().mean().item() * 100

            cur_idx = 0
            scale_loss = {}
            scale_acc = {}
            for i in range(len(curr_patch_nums)):
                cur_scale = curr_patch_nums[i]
                curr_l = curr_patch_nums[i] ** 2
                curr_gt = gt_indices_tf[:, cur_idx:cur_idx+curr_l]
                curr_logit = logits_tf.data[:, cur_idx:cur_idx+curr_l]
                curr_pred = pred_BL[:, cur_idx:cur_idx+curr_l]
                scale_loss[f"scale_{cur_scale}_loss"] = self.val_loss(curr_logit.reshape(-1, V), curr_gt.reshape(-1)).item()
                scale_acc[f"scale_{cur_scale}_acc"] = (curr_pred == curr_gt).float().mean().item() * 100
                cur_idx += curr_l
            Ltail = scale_loss[f"scale_{cur_scale}_loss"]
            acc_tail = scale_acc[f"scale_{cur_scale}_acc"]

            grad_norm_val = grad_norm.item()
            metric_lg.update(
                Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_val,
                TF_loss=self.last_teacher_loss, SF_loss=self.last_student_loss, Total=self.last_csfl_total_loss
            )
            tb_lg.update(head='AR_loss', step=g_it, L_mean=Lmean, acc_mean=acc_mean)
            tb_lg.update(head='AR_scale_loss', step=g_it, **scale_loss)
            tb_lg.update(head='AR_scale_acc', step=g_it, **scale_acc)
            tb_lg.update(head='AR_csfl', step=g_it,
                        teacher_loss=self.last_teacher_loss,
                        student_loss=self.last_student_loss,
                        total_loss=self.last_csfl_total_loss)

        return grad_norm, scale_log2

    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
            'training_mode': self.training_mode,
            'sf_cfg_scale': self.sf_cfg_scale,
            'sf_top_k': self.sf_top_k,
            'sf_top_p': self.sf_top_p,
            'sf_use_sampling': self.sf_use_sampling,
            'last_teacher_loss': self.last_teacher_loss,
            'last_student_loss': self.last_student_loss,
            'last_csfl_total_loss': self.last_csfl_total_loss,
            'last_tf_single_sf_total_loss': self.last_tf_single_sf_total_loss,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                # Skip attn_bias_for_masking when loading var_wo_ddp to avoid size mismatch
                if k == 'var_wo_ddp' and 'attn_bias_for_masking' in state[k]:
                    state[k].pop('attn_bias_for_masking')
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        self.training_mode = config.get('training_mode', 'teacher_forcing')
        self.sf_cfg_scale = config.get('sf_cfg_scale', 1.2)
        self.sf_top_k = config.get('sf_top_k', 900)
        self.sf_top_p = config.get('sf_top_p', 0.96)
        self.sf_use_sampling = config.get('sf_use_sampling', True)
        self.last_teacher_loss = config.get('last_teacher_loss', 0.0)
        self.last_student_loss = config.get('last_student_loss', 0.0)
        self.last_csfl_total_loss = config.get('last_csfl_total_loss', 0.0)
        self.last_tf_single_sf_total_loss = config.get('last_tf_single_sf_total_loss', 0.0)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
