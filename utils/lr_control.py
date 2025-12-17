import math
from pprint import pformat
from typing import Tuple, List, Dict, Union

import torch.nn

import dist


def lr_wd_annealing(sche_type: str, optimizer, peak_lr, wd, wd_end, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
    """
    Learning rate and weight decay annealing (scheduling) with various strategies.
    
    The learning rate schedule consists of two phases:
    1. Warmup phase: Linear increase from wp0*peak_lr to peak_lr
    2. Annealing phase: Decay from peak_lr to wpe*peak_lr using various strategies
    
    Args:
        sche_type: Schedule type for annealing phase
            - 'cos': Cosine annealing
            - 'lin': Linear decay with 15% constant phase at peak_lr
            - 'lin0': Linear decay with 5% constant phase at peak_lr  
            - 'lin00': Pure linear decay from start to end
            - 'linX': Linear decay with X% constant phase (e.g., 'lin0.2' = 20%)
            - 'exp': Exponential decay after 15% constant phase
        optimizer: PyTorch optimizer instance
        peak_lr: Peak learning rate (reached after warmup)
        wd: Initial weight decay
        wd_end: Final weight decay
        cur_it: Current iteration
        wp_it: Number of warmup iterations
        max_it: Total number of iterations
        wp0: Initial lr ratio at iteration 0 (default: 0.005 = 0.5% of peak_lr)
        wpe: Final lr ratio at max_it (default: 0.001 = 0.1% of peak_lr)
    
    Returns:
        Tuple of (min_lr, max_lr, min_wd, max_wd) across all param groups
    """
    wp_it = round(wp_it)
    
    # Phase 1: Linear warmup from wp0 to 1.0 (relative to peak_lr)
    if cur_it < wp_it:
        # Linear interpolation: starts at wp0, reaches 1.0 at wp_it
        cur_lr = wp0 + (1-wp0) * cur_it / wp_it
    else:
        # Phase 2: Annealing phase after warmup
        pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # Progress after warmup: [0, 1]
        rest = 1 - pasd     # Remaining progress: [1, 0]
        
        if sche_type == 'cos':
            # Cosine annealing: smooth decay using cosine curve
            # cos(0) = 1 -> cos(π) = -1, scaled to [1, wpe]
            cur_lr = wpe + (1-wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
            
        elif sche_type == 'lin':
            # Linear with 15% constant phase at peak
            T = 0.15; max_rest = 1-T
            if pasd < T: cur_lr = 1  # Stay at peak_lr for first 15%
            else: cur_lr = wpe + (1-wpe) * rest / max_rest  # Linear decay: 1 to wpe
            
        elif sche_type == 'lin0':
            # Linear with 5% constant phase at peak (default schedule)
            T = 0.05; max_rest = 1-T
            if pasd < T: cur_lr = 1  # Stay at peak_lr for first 5%
            else: cur_lr = wpe + (1-wpe) * rest / max_rest
            
        elif sche_type == 'lin00':
            # Pure linear decay from 1 to wpe immediately after warmup
            cur_lr = wpe + (1-wpe) * rest
            
        elif sche_type.startswith('lin'):
            # Linear with custom constant phase (e.g., 'lin0.8' = 80% constant)
            # Used for progressive training schedules
            T = float(sche_type[3:]); max_rest = 1-T
            # Compute midpoint lr for smooth transition
            wpe_mid = wpe + (1-wpe) * max_rest
            wpe_mid = (1 + wpe_mid) / 2
            if pasd < T: 
                # First T phase: interpolate from 1 to wpe_mid
                cur_lr = 1 + (wpe_mid-1) * pasd / T
            else: 
                # Remaining phase: interpolate from wpe_mid to wpe
                cur_lr = wpe + (wpe_mid-wpe) * rest / max_rest
                
        elif sche_type == 'exp':
            # Exponential decay after 15% constant phase
            T = 0.15; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else:
                # Exponential decay: e^(x*log(wpe)) where x goes from 0 to 1
                expo = (pasd-T) / max_rest * math.log(wpe)
                cur_lr = math.exp(expo)
        else:
            raise NotImplementedError(f'unknown sche_type {sche_type}')
    
    # Convert relative lr to absolute lr
    cur_lr *= peak_lr
    
    # Weight decay scheduling: cosine annealing from wd to wd_end
    pasd = cur_it / (max_it-1)
    cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))
    
    # Apply lr and wd to all parameter groups
    inf = 1e6
    min_lr, max_lr = inf, -1
    min_wd, max_wd = inf, -1
    for param_group in optimizer.param_groups:
        # Apply lr with optional per-group scaling factor
        param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' = lr scale
        max_lr = max(max_lr, param_group['lr'])
        min_lr = min(min_lr, param_group['lr'])
        
        # Apply wd with optional per-group scaling factor
        param_group['weight_decay'] = cur_wd * param_group.get('wd_sc', 1)  # 'wd_sc' = wd scale
        max_wd = max(max_wd, param_group['weight_decay'])
        if param_group['weight_decay'] > 0:
            min_wd = min(min_wd, param_group['weight_decay'])

    if min_lr == inf: min_lr = -1
    if min_wd == inf: min_wd = -1
    return min_lr, max_lr, min_wd, max_wd


def filter_params(model, nowd_keys=()) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name = name.replace('_fsdp_wrapped_module.', '')
        if not para.requires_grad:
            names_no_grad.append(name)
            continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)
        
        if para.ndim == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0., 'ND'
        else:
            cur_wd_sc, group_name = 1., 'D'
        cur_lr_sc = 1.
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    print(f'[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n')
    
    for rk in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == rk:
            print(f'[get_param_groups][rank{dist.get_rank()}] {type(model).__name__=} {count=}, {numel=}', flush=True, force=True)
    print('')
    
    assert len(names_no_grad) == 0, f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n'
    return names, paras, list(para_groups.values())
