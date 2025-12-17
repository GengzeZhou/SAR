import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
# from utils.data import build_dataset
from dataset.build import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from utils.logger import UnifiedLogger, DistLogger

# Add visualization imports
import numpy as np
from PIL import Image
from torchvision.utils import make_grid



class DebugOverfitDataset(torch.utils.data.Dataset):
    """Debug dataset that only uses first N samples and repeats them"""
    def __init__(self, base_dataset, num_samples=4):
        self.base_dataset = base_dataset
        self.num_samples = min(num_samples, len(base_dataset))
        # Cache the first N samples
        self.cached_samples = []
        self.debug_classes = []  # Track the classes for debug visualization
        for i in range(self.num_samples):
            sample = base_dataset[i]
            self.cached_samples.append(sample)
            # Extract class label (assuming it's the second element)
            if isinstance(sample, tuple) and len(sample) >= 2:
                self.debug_classes.append(int(sample[1]))
        print(f'[DebugOverfitDataset] Created with {self.num_samples} samples')
        print(f'[DebugOverfitDataset] Classes: {self.debug_classes}')
    
    def __len__(self):
        return len(self.cached_samples)
    
    def __getitem__(self, idx):
        # Always return one of the first N samples
        return self.cached_samples[idx % self.num_samples]


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args)
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # Create UnifiedLogger based on logger_type
        tb_lg = DistLogger(UnifiedLogger(args.logger_type, args.tb_log_dir_path, args), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # build data
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        num_classes, dataset_train, _ = build_dataset(
            args,
            # args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        )
        
        # Wrap dataset for overfitting debug if requested
        debug_dataset = None
        if args.overfit_debug:
            debug_dataset = DebugOverfitDataset(dataset_train, num_samples=4)
            dataset_train = debug_dataset
            print(f'[OVERFIT DEBUG MODE] Training will overfit on first 4 samples')
            # Adjust settings for faster feedback in debug mode
            args.log_every = min(args.log_every, 100)  # Log more frequently
            args.save_epo = 200
            print(f'[OVERFIT DEBUG MODE] Adjusted log_every={args.log_every}, save_epo={args.save_epo}')
        
        types = str((type(dataset_train).__name__, ))
        
        ld_val = None
        
        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=not args.overfit_debug, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        
        # [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import FlexVAR, ScaleAR, build_vae_var, build_vae_scalear
    from trainer import VARTrainer
    from scalear_trainer import ScaleARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    
    # Check if using ScaleAR model
    if hasattr(args, 'model_type') and args.model_type == 'scalear':
        vae_local, var_wo_ddp = build_vae_scalear(
            V=args.vae_v,
            Cvae=args.vae_c,
            device=dist.get_device(), patch_nums=args.patch_nums,
            num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm, token_dropout_p=args.token_dropout_p,
            randar_scale=getattr(args, 'randar_scale', 4),
            randar_mode=getattr(args, 'randar_mode', 'maskgit'),
            enable_var_kv=getattr(args, 'enable_var_kv', False),
            share_randar_var_pos_embed=getattr(args, 'share_randar_var_pos_embed', False),
            flash_if_available=args.fuse, fused_if_available=args.fuse,
            init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini, vae_ckpt=args.vae_ckpt,
        )
    else:
        vae_local, var_wo_ddp = build_vae_var(
            V=args.vae_v,
            Cvae=args.vae_c,
            device=dist.get_device(), patch_nums=args.patch_nums,
            num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm, token_dropout_p=args.token_dropout_p,
            flash_if_available=args.fuse, fused_if_available=args.fuse,
            init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini, vae_ckpt=args.vae_ckpt,
        )
    
    vae_local = args.compile_model(vae_local, args.vfast)
    var_wo_ddp = args.compile_model(var_wo_ddp, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    # build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # build trainer
    if hasattr(args, 'model_type') and args.model_type == 'scalear':
        trainer = ScaleARTrainer(
            device=args.device, patch_nums=args.patch_nums, resos=args.resos,
            vae_local=vae_local, scalear_wo_ddp=var_wo_ddp, scalear=var,
            scalear_opt=var_optim, label_smooth=args.ls, mask_ratio_min=args.mask_ratio_min,
            training_mode=getattr(args, 'training_mode', 'teacher_forcing'),
            sigma=getattr(args, 'sigma', 0.5),
            hybrid_tf_scales=getattr(args, 'hybrid_tf_scales', 8),
            sf_cfg_scale=getattr(args, 'sf_cfg_scale', 1.2),
            sf_top_k=getattr(args, 'sf_top_k', 900),
            sf_top_p=getattr(args, 'sf_top_p', 0.96),
            sf_use_sampling=getattr(args, 'sf_use_sampling', True),
        )
    else:
        trainer = VARTrainer(
            device=args.device, patch_nums=args.patch_nums, resos=args.resos,
            vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
            var_opt=var_optim, label_smooth=args.ls,
            training_mode=args.training_mode,
            sigma=args.sigma,
        )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    elif args.continue_training_ckpt is not None:
        ckpt = torch.load(args.continue_training_ckpt, map_location='cpu')
        old_params = trainer.var_wo_ddp.state_dict()
        ckpt["attn_bias_for_masking"] = old_params["attn_bias_for_masking"]
        trainer.var_wo_ddp.load_state_dict(ckpt, strict=True)
    
    # Store debug classes and dataset in trainer if in debug mode
    if args.overfit_debug and 'debug_dataset' in locals() and debug_dataset is not None:
        trainer.debug_classes = debug_dataset.debug_classes
        trainer.debug_dataset = debug_dataset
    
    del vae_local, var_wo_ddp, var, var_optim
    
    # Watch model with wandb if using wandb logger
    if with_tb_lg and args.logger_type == 'wandb':
        if hasattr(trainer, 'scalear_wo_ddp'):
            tb_lg.watch_model(trainer.scalear_wo_ddp, log_freq=args.log_every)
        else:
            tb_lg.watch_model(trainer.var_wo_ddp, log_freq=args.log_every)
    
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        label = torch.ones(B, dtype=torch.long)
        
        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True)
        tb_lg.set_step(ep * iters_train)
        
        # visualize before training
        if ep == start_ep:
            visualize_inference(trainer, args, ep, num_samples=args.vis_num_samples, cfg_scale=args.vis_cfg_scale, tb_lg=tb_lg)
            # Visualize VAE reconstruction for first 4 images
            visualize_vae_reconstruction_first4(trainer, args, ld_train, tb_lg=tb_lg)
        
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer, session_start_ep=(ep == start_ep)
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail, epoch=ep+1)
        is_val_and_also_saving = (ep + 1) % args.save_epo == 0 or (ep + 1) == args.ep
        print('x'*10, dist.get_rank(), is_val_and_also_saving)
        if is_val_and_also_saving:
            print(f'is_val_and_also_saving: {is_val_and_also_saving}')
            
            # Add inference visualization before saving checkpoint
            if args.vis_save:
                try:
                    visualize_inference(trainer, args, ep, num_samples=args.vis_num_samples, cfg_scale=args.vis_cfg_scale, tb_lg=tb_lg)
                except Exception as e:
                    print(f'[inference visualization] failed with error: {e}', flush=True)

            if dist.get_rank()==0 and not args.overfit_debug:
            # if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, f'ar-ckpt-epo{ep}.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True)
            dist.barrier()

        print(    f'     [ep{ep}]  (training)  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', rest_hours=round(sec / 60 / 60, 2), **AR_ep_loss)
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: UnifiedLogger, ld_or_itrt, iters_train: int, trainer, session_start_ep: bool = False):
    # import heavy packages after Dataloader object creation
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, args.log_every, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        # Determine if we should force logging (first 100 steps of session)
        force_log = session_start_ep and it < 100
        
        # Pass force_log parameter if this is a ScaleAR model
        if hasattr(args, 'model_type') and args.model_type == 'scalear':
            grad_norm, scale_log2 = trainer.train_step(
                it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
                inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=args.pgwp * iters_train,
                force_log=force_log,
            )
        else:
            grad_norm, scale_log2 = trainer.train_step(
                it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
                inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=args.pgwp * iters_train,
            )
        
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_lr', lr_min=min_tlr, lr_max=max_tlr)
        tb_lg.update(head='AR_weight_decay', weight_decay_min=min_twd, weight_decay_max=max_twd)
        tb_lg.update(head='AR_grad', scale_log2=scale_log2, grad_norm=grad_norm)
        if args.tclip > 0:
            tb_lg.update(head='AR_grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


def visualize_inference(trainer, args, ep, num_samples=8, cfg_scale=4.0, tb_lg: UnifiedLogger = None):
    """Generate and save inference samples for visualization with parallel generation support"""
    import os
    
    # Set model to eval mode
    if hasattr(trainer, 'scalear_wo_ddp'):
        trainer.scalear_wo_ddp.eval()
    else:
        trainer.var_wo_ddp.eval()
    
    # Check if we're in debug mode and should use specific classes
    debug_classes = None
    if args.overfit_debug and hasattr(trainer, 'debug_classes'):
        debug_classes = trainer.debug_classes
        # Use only the debug classes without cfg
        cfg_scale = 2.0
        num_samples = min(num_samples, len(debug_classes))
        print(f'[inference visualization] Debug mode: using classes {debug_classes} with cfg=2.0', flush=True)
    
    # Calculate samples per GPU
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    samples_per_gpu = (num_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, num_samples)
    local_num_samples = end_idx - start_idx
    
    if local_num_samples <= 0:
        print(f'[inference visualization] rank {rank} has no samples to generate', flush=True)
        return
    
    print(f'[inference visualization] rank {rank} generating {local_num_samples} samples (indices {start_idx}-{end_idx-1})...', flush=True)
    
    # Create visualization directory (all ranks create it to avoid race conditions)
    vis_dir = os.path.join(args.local_out_dir_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate samples with different class labels
    if debug_classes is not None:
        # Use debug classes repeated to match num_samples
        class_labels = debug_classes * ((num_samples // len(debug_classes)) + 1)
        class_labels = class_labels[:num_samples]
    else:
        # Use default diverse class labels
        class_labels = [
            # Animals
            88,   # African grey parrot
            130,  # Flamingo  
            207,  # Golden retriever
            360,  # Otter
            387,  # African elephant
            33,   # Loggerhead turtle
            1,    # Goldfish
            292,  # Tiger
            340,  # Zebra
            281,  # Tabby cat
            151,  # Chihuahua
            273,  # Dingo
            
            # Nature/Landscapes
            974,  # Cliff
            980,  # Volcano
            972,  # Mountain
            979,  # Valley
            975,  # Lakeside
            973,  # Coral reef
            
            # Objects/Food
            928,  # Ice cream
            950,  # Orange
            417,  # Balloon
            933,  # Cheeseburger
            954,  # Banana
            953,  # Pineapple
            960,  # Pizza
            967,  # Espresso
            
            # Insects/Small creatures
            323,  # Monarch butterfly
            321,  # Admiral butterfly
            319,  # Cabbage butterfly
            311,  # Grasshopper
            300,  # Ladybug
            301,  # Leaf beetle

            # Structures & Indoor Objects
            985,  # Palace
            987,  # Castle
            988,  # Monastery
            992,  # Mosque
            975,  # Lakeside (you had lakeside, but adding indoor/outdoor structures)
            741,  # Park bench
            706,  # Dining table
            647,  # Parking meter

            # Vehicles / Transportation
            609,  # Jetliner
            586,  # Helicopter
            817,  # Sports car
            510,  # Fire truck
            864,  # Tank
            581,  # School bus
            436,  # Tandem bicycle
            656,  # Passenger ship

            # Sports / Human Activities / Tools
            870,  # Tennis ball
            852,  # Baseball
            876,  # Volleyball
            889,  # Hockey puck
            898,  # Basketball
            765,  # Snowmobile
            516,  # Lawn mower
            566,  # Chainsaw
            
            # Musical Instruments
            402,  # Acoustic guitar
            420,  # Violin
            402,  # Banjo
            402,  # Drum
            486,  # French horn
            558,  # Saxophone
            579,  # Piano
            567,  # Marimba
        ]
        class_labels = class_labels * ((num_samples // len(class_labels)) + 1)
        class_labels = class_labels[:num_samples]
    
    # Get this GPU's subset of class labels
    local_class_labels = class_labels[start_idx:end_idx]
    
    with torch.no_grad():
        # Generate samples
        B = local_num_samples
        label_B = torch.tensor(local_class_labels, dtype=torch.long, device=args.device)
        
        # Use the VAR/ScaleAR model's built-in sampling
        if hasattr(trainer, 'scalear_wo_ddp'):
            var_wo_ddp = trainer.scalear_wo_ddp
        else:
            var_wo_ddp = trainer.var_wo_ddp
        # Use the same patch_nums as training
        infer_patch_nums = args.patch_nums
        
        # Visualize intermediate scales
        if args.vis_intermediate_scales:
            # ScaleAR with intermediate scales
            result = var_wo_ddp.autoregressive_infer_cfg(
                vqvae=trainer.vae_local,
                B=B, 
                label_B=label_B,
                infer_patch_nums=infer_patch_nums,
                cfg=cfg_scale, 
                top_k=900, 
                top_p=0.95,
                more_smooth=False,
                return_intermediate_scales=True,
                cosine_steps=args.vis_cosine_steps,
            )
            samples, intermediate_scales = result
            images = samples
        else:
            # Regular inference without intermediate scales
            samples = var_wo_ddp.autoregressive_infer_cfg(
                vqvae=trainer.vae_local,
                B=B, 
                label_B=label_B,
                infer_patch_nums=infer_patch_nums,
                cfg=cfg_scale, 
                top_k=900, 
                top_p=0.95,
                more_smooth=False,
                cosine_steps=args.vis_cosine_steps
            )
            images = samples
            intermediate_scales = None
    
    # Gather all generated images to master using allgather
    gathered_images = dist.allgather(images, cat=False)
    
    # Gather intermediate scales if available
    if intermediate_scales is not None:
        gathered_intermediate = {}
        for scale, scale_images in intermediate_scales.items():
            gathered_scale_images = dist.allgather(scale_images, cat=False)
            gathered_intermediate[scale] = gathered_scale_images
    else:
        gathered_intermediate = None
    
    if rank == 0:
        # Filter out empty tensors and concatenate
        all_images = torch.cat([img for img in gathered_images if img.shape[0] > 0], dim=0)
        
        # Create and save grid for final scale
        # Use 8 columns for 32 images (4 rows x 8 columns)
        nrow = 8 if num_samples >= 32 else 4
        # Use normalize=True to properly handle [-1, 1] range images
        grid = make_grid(all_images[:num_samples], nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))
        # Grid is now in [0, 1] range after normalization
        grid_img = Image.fromarray((grid.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        grid_path = os.path.join(vis_dir, f'ep{ep:03d}_grid.png')
        grid_img.save(grid_path)
        
        # Save intermediate scale grids if available
        intermediate_grids = {}  # Store grids for logging later
        if gathered_intermediate is not None:
            for scale, scale_images_list in gathered_intermediate.items():
                # Filter and concatenate
                scale_all_images = torch.cat([img for img in scale_images_list if img.shape[0] > 0], dim=0)
                # Create grid for this scale
                scale_grid = make_grid(scale_all_images[:num_samples], nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))
                scale_grid_img = Image.fromarray((scale_grid.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                scale_grid_path = os.path.join(vis_dir, f'ep{ep:03d}_grid_{scale}x{scale}.png')
                scale_grid_img.save(scale_grid_path)
                # Store grid tensor for logging
                intermediate_grids[scale] = scale_grid
        
        # Log to wandb/tensorboard if available
        if tb_lg is not None:
            try:
                # Use the grid tensor directly (already in [0, 1] range after normalization)
                # Explicitly pass the current step to ensure logging happens for all epochs
                # Get the underlying logger's step if using DistLogger wrapper
                if hasattr(tb_lg, '_logger') and tb_lg._logger is not None:
                    current_step = tb_lg._logger.step
                else:
                    current_step = getattr(tb_lg, 'step', None)
                # Use consistent tag names (without epoch) so wandb/tensorboard creates sliders
                tb_lg.log_image(f"samples/grid", grid, step=current_step, caption=f"Epoch {ep} samples (cfg={cfg_scale})")
                
                # Log intermediate scale grids if available
                if intermediate_grids:
                    for scale in sorted(intermediate_grids.keys()):
                        tb_lg.log_image(f"samples/grid_{scale}x{scale}", intermediate_grids[scale], step=current_step, 
                                      caption=f"Epoch {ep} {scale}x{scale} samples (cfg={cfg_scale})")
            except Exception as e:
                print(f'[wandb/tensorboard logging] failed with error: {e}', flush=True)
        
        print(f'[inference visualization] Completed! Generated {num_samples} samples across {world_size} GPUs', flush=True)
        print(f'[inference visualization] Saved to {vis_dir}', flush=True)
    
    # Set model back to train mode
    if hasattr(trainer, 'scalear_wo_ddp'):
        trainer.scalear_wo_ddp.train()
    else:
        trainer.var_wo_ddp.train()
    
    # Synchronize all ranks before returning
    dist.barrier()


def visualize_vae_reconstruction_first4(trainer, args, dataloader_iter, tb_lg: UnifiedLogger = None):
    """Visualize VAE reconstruction at different scales for the first 4 images"""
    import os
    import torch.nn.functional as F
    
    print(f'[VAE reconstruction visualization] Starting on rank {dist.get_rank()}...', flush=True)
    
    # Set model to eval mode
    trainer.vae_local.eval()
    
    # Get first 4 images from dataloader
    # Each rank gets its own batch, but we'll gather to rank 0
    if args.overfit_debug and hasattr(trainer, 'debug_dataset'):
        # In debug mode, use the cached samples from trainer
        images = []
        labels = []
        debug_dataset = trainer.debug_dataset
        for i in range(min(4, len(debug_dataset.cached_samples))):
            img, label = debug_dataset.cached_samples[i]
            images.append(img)
            labels.append(label)
        
        if len(images) > 0:
            images = torch.stack(images).to(args.device)
            labels = torch.tensor(labels).to(args.device)
        else:
            # Fallback to empty tensors
            images = torch.zeros(0, 3, args.data_load_reso, args.data_load_reso).to(args.device)
            labels = torch.zeros(0, dtype=torch.long).to(args.device)
    else:
        # Get first batch from dataloader iterator
        try:
            images, labels = next(dataloader_iter)
            images = images.to(args.device)
            labels = labels.to(args.device)
        except StopIteration:
            # Empty dataloader on this rank
            images = torch.zeros(0, 3, args.data_load_reso, args.data_load_reso).to(args.device)
            labels = torch.zeros(0, dtype=torch.long).to(args.device)
    
    # Gather first 4 images across all ranks to rank 0
    all_images = dist.allgather(images, cat=True)
    all_labels = dist.allgather(labels, cat=True)
    
    # Take only first 4 samples
    if all_images.shape[0] >= 4:
        images = all_images[:4]
        labels = all_labels[:4]
    else:
        print(f'[VAE reconstruction visualization] Warning: Only {all_images.shape[0]} images available', flush=True)
        images = all_images
        labels = all_labels
    
    # Only rank 0 does the visualization
    if dist.get_rank() == 0:
        vis_dir = os.path.join(args.local_out_dir_path, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Only rank 0 performs visualization
    if dist.get_rank() == 0 and images.shape[0] > 0:
        with torch.no_grad():
            # Get patch sizes from args
            patch_sizes = [pn * args.patch_size for pn in args.patch_nums]
            
            all_images = []
            
            # Add original images first
            for i in range(images.shape[0]):
                all_images.append(images[i])
            
            # First, encode at full resolution to get 16x16 latents
            h_full = trainer.vae_local.encode_conti(images)  # Continuous latent at full resolution
            
            # For each scale in patch_nums
            for scale_idx, pn in enumerate(args.patch_nums):
                # Calculate the latent size for this patch number
                latent_size = pn  # patch_num directly corresponds to latent dimensions
                
                # Downsample the continuous latent to target size
                if latent_size != 16:  # 16 is the full resolution
                    h_scaled = F.interpolate(h_full.clone(), size=(latent_size, latent_size), mode='area')
                else:
                    h_scaled = h_full.clone()
                
                # Quantize at this scale
                quant, _, _ = trainer.vae_local.quantize(h_scaled)
                
                # Decode back to image space
                reconstructed = trainer.vae_local.decode(quant)
                
                # Always resize to match original image size for consistent visualization
                original_size = images.shape[-1]  # Get size from original images
                if reconstructed.shape[-1] != original_size:
                    reconstructed = F.interpolate(reconstructed, size=(original_size, original_size), mode='bilinear', align_corners=False)
                
                # Add reconstructions
                for i in range(reconstructed.shape[0]):
                    all_images.append(reconstructed[i])
        
            # Create grid: first row is originals, then each row is reconstructions at different scales
            # Total images: 4 originals + 4 * num_scales reconstructions
            vis_tensor = torch.stack(all_images)
            nrow = min(4, images.shape[0])  # 4 images per row or less if we have fewer
            grid = make_grid(vis_tensor, nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))
            
            # Save grid image
            grid_img = Image.fromarray((grid.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            grid_path = os.path.join(vis_dir, f'vae_reconstruction_first4.png')
            grid_img.save(grid_path)
            
            # Log to wandb/tensorboard if available
            if tb_lg is not None:
                try:
                    if hasattr(tb_lg, '_logger') and tb_lg._logger is not None:
                        current_step = tb_lg._logger.step
                    else:
                        current_step = getattr(tb_lg, 'step', None)
                    
                    # Create caption with scale info
                    latent_sizes = ', '.join([f'{pn}x{pn}' for pn in args.patch_nums])
                    caption = f"VAE Reconstruction of first {images.shape[0]} images. Row 1: Original. Rows 2-{len(args.patch_nums)+1}: Reconstructions from {latent_sizes} latents"
                    
                    tb_lg.log_image(f"vae_recon/first4_images", grid, step=current_step, caption=caption)
                    
                    # Also log individual scale comparisons
                    for scale_idx, target_size in enumerate(patch_sizes):
                        scale_images = []
                        # Original images
                        for i in range(images.shape[0]):
                            scale_images.append(all_images[i])
                        # Reconstructions at this scale
                        for i in range(images.shape[0]):
                            scale_images.append(all_images[images.shape[0] + scale_idx * images.shape[0] + i])
                        
                        scale_grid = make_grid(torch.stack(scale_images), nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))
                        tb_lg.log_image(f"vae_recon/scale_{target_size}x{target_size}", scale_grid, step=current_step, 
                                      caption=f"Top: Original, Bottom: VAE reconstruction at {target_size}x{target_size}")
                    
                except Exception as e:
                    print(f'[VAE recon logging] failed with error: {e}', flush=True)
            
            print(f'[VAE reconstruction visualization] Completed! Saved to {grid_path}', flush=True)
            print(f'[VAE reconstruction visualization] Visualized scales: {", ".join([f"{ps}x{ps}" for ps in patch_sizes])}', flush=True)
    
    # Synchronize before continuing
    dist.barrier()


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
