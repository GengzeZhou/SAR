import os
import os.path as osp
import torch
import random
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import build_vae_var, build_vae_scalear
from torchvision.utils import save_image
import time
import torch.distributed as dist
import argparse
import json
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="scalear", choices=["var", "scalear"],
                        help="Model type: 'var' for original VAR or 'scalear' for ScaleAR")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--var_ckpt", type=str, required=True, help="Path to VAR/ScaleAR checkpoint")
    parser.add_argument("--vae_ckpt", type=str, default="pretrained/FlexVAE.pth")
    parser.add_argument("--cfg", type=float, default=1.5, help="Classifier-free guidance scale")
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--maxpn", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16, help="Model depth")
    parser.add_argument("--infer_patch_nums", type=str, help="Patch numbers for inference")
    parser.add_argument("--output_path", type=str, default="/mnt/localssd", help="Base output path")
    parser.add_argument("--results_path", type=str, default="/mnt/localssd", help="Results path")
    parser.add_argument("--cosine_steps", type=int, default=8, help="Number of cosine steps (ScaleAR only)")
    parser.add_argument("--randar_scale", type=int, default=4, help="RandAR starting scale (e.g., 4 for 4x4, 8 for 8x8)")
    parser.add_argument("--randar_mode", type=str, default="maskgit", choices=["maskgit", "randar"], help="RandAR mode")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per class")
    parser.add_argument("--vis_samples", type=int, default=100, help="Number of samples for visualization")

    args = parser.parse_args()

    # Set default infer_patch_nums based on model type and randar_scale
    if args.infer_patch_nums is None or args.infer_patch_nums == "4_5_6_8_10_13_16":
        if args.model_type == "var":
            args.infer_patch_nums = "1_2_3_4_5_6_8_10_13_16"
        else:  # scalear - start from randar_scale
            randar_scale = args.randar_scale
            # Build patch numbers starting from randar_scale
            all_scales = [4, 5, 6, 8, 10, 13, 16]
            valid_scales = [s for s in all_scales if s >= randar_scale]
            args.infer_patch_nums = "_".join(map(str, valid_scales))

    # Auto-parse exp_name from checkpoint if not provided
    if args.exp_name is None:
        try:
            exp_dir = osp.basename(osp.dirname(osp.dirname(args.var_ckpt)))
            args.exp_name = exp_dir.replace('_', '-')
        except:
            args.exp_name = f"{args.model_type}_eval"

    MODEL_DEPTH = args.depth
    assert MODEL_DEPTH in {16, 20, 24, 30}
    infer_patch_nums = tuple(map(int, args.infer_patch_nums.replace('-', '_').split('_')))

    dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()

    if global_rank == 0:
        print(f"=== {args.model_type.upper()} Evaluation ===")
        print(f"Configuration:")
        print(f"  Model Type: {args.model_type}")
        print(f"  CFG Scale: {args.cfg}")
        if args.model_type == "scalear":
            print(f"  Cosine Steps: {args.cosine_steps}")
            print(f"  RandAR Scale: {args.randar_scale}")
            print(f"  RandAR Mode: {args.randar_mode}")
        print(f"  Output Path: {args.output_path}")
        print(f"  Results Path: {args.results_path}")
        print(f"  Experiment: {args.exp_name}")
        print(f"  Model Depth: {MODEL_DEPTH}")
        print(f"  Patch Nums: {infer_patch_nums}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Samples per class: {args.num_samples}")
        print(f"=" * 30)

    seed = 0 * dist.get_world_size() + global_rank
    torch.cuda.set_device(device)

    # Build paths
    if args.model_type == "scalear":
        config_name = f"cfg{args.cfg}_cosine{args.cosine_steps}"
    else:
        config_name = f"cfg{args.cfg}"

    results_root = osp.join(
        args.results_path,
        args.exp_name,
        f"d{MODEL_DEPTH}_{config_name}_shape{infer_patch_nums[-1]*16}_{len(infer_patch_nums)}step_maxpn{args.maxpn}"
    )

    save_root = osp.join(
        args.output_path,
        args.exp_name,
        f"d{MODEL_DEPTH}_{config_name}_shape{infer_patch_nums[-1]*16}_{len(infer_patch_nums)}step_maxpn{args.maxpn}",
    )

    # Only rank 0 creates directories and saves config
    if global_rank == 0:
        os.makedirs(results_root, exist_ok=True)
        os.makedirs(save_root, exist_ok=True)
        print(f"Inference results saved to: {save_root}")

        config_dict = vars(args)
        config_dict['save_root'] = save_root
        config_dict['results_root'] = results_root
        with open(osp.join(results_root, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

    # Build models based on model type
    vae_ckpt = args.vae_ckpt
    var_ckpt = args.var_ckpt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_type == "var":
        # Build original VAR model
        vae, model = build_vae_var(
            V=8912,
            Cvae=32, device=device, num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
            vae_ckpt=vae_ckpt,
        )

        # Load VAR checkpoint
        ckpt = torch.load(var_ckpt, map_location='cpu')
        if 'trainer' in ckpt.keys():
            ckpt = ckpt['trainer']['var_wo_ddp']
        old_params = model.state_dict()
        ckpt["attn_bias_for_masking"] = old_params["attn_bias_for_masking"]
        model.load_state_dict(ckpt, strict=True)
        model.update_patch_related(infer_patch_nums)

    else:  # scalear
        # Build ScaleAR model
        vae, model = build_vae_scalear(
            V=8912,
            Cvae=32, device=device, num_classes=1000, depth=MODEL_DEPTH,
            patch_nums=infer_patch_nums,
            vae_ckpt=vae_ckpt,
            randar_scale=getattr(args, 'randar_scale', 4),
            randar_mode=getattr(args, 'randar_mode', 'maskgit'),
            enable_var_kv=getattr(args, 'enable_var_kv', False),
            share_randar_var_pos_embed=getattr(args, 'share_randar_var_pos_embed', True),
        )

        # Load ScaleAR checkpoint
        ckpt = torch.load(var_ckpt, map_location='cpu')
        if 'trainer' in ckpt.keys():
            ckpt = ckpt['trainer'].get('scalear_wo_ddp', ckpt['trainer'].get('var_wo_ddp'))

        # Remove attn_bias_for_masking from checkpoint as it depends on patch_nums
        # The model will regenerate it based on the evaluation patch_nums
        if 'attn_bias_for_masking' in ckpt:
            del ckpt['attn_bias_for_masking']

        model.load_state_dict(ckpt, strict=False)

        # ScaleAR patch_nums are already set during model initialization via build_vae_scalear
        # No need to call update_patch_related as it doesn't exist for ScaleAR

    vae.eval(), model.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in model.parameters(): p.requires_grad_(False)

    if global_rank == 0:
        print(f'Model loading finished.')

    # Sampling configuration
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # Generate data list
    data_list = [i for i in range(1000)] * args.num_samples
    random.shuffle(data_list)
    length = len(data_list) // dist.get_world_size()
    data_sublist = data_list[global_rank::dist.get_world_size()]
    dataset = [data_sublist[i:i + args.batch_size] for i in range(0, len(data_sublist), args.batch_size)]

    # Track class mapping for all images
    image_class_mapping = {}

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):

            total_time = 0
            num_batches = 0

            for num, c in enumerate(dataset):
                label_B: torch.LongTensor = torch.tensor(c, device=device)

                t1 = time.time()
                B = len(c)

                # Model-specific inference
                if args.model_type == "var":
                    # Original VAR inference
                    recon_B3HW = model.autoregressive_infer_cfg(
                        vqvae=vae, B=B, label_B=label_B,
                        infer_patch_nums=infer_patch_nums,
                        cfg=args.cfg, top_k=args.top_k, top_p=0.95,
                        g_seed=None, more_smooth=False,
                        max_pn=args.maxpn
                    )
                else:  # scalear
                    # ScaleAR inference
                    recon_B3HW = model.autoregressive_infer_cfg(
                        vqvae=vae, B=B, label_B=label_B,
                        infer_patch_nums=infer_patch_nums,
                        cfg=args.cfg, top_k=args.top_k, top_p=0.95,
                        g_seed=None, more_smooth=False,
                        cosine_steps=args.cosine_steps
                    )

                # Save individual images
                for i in range(len(c)):
                    img_filename = f"rank{global_rank}-{num * args.batch_size + i}.png"
                    img_path = osp.join(save_root, img_filename)
                    save_image(recon_B3HW[i].unsqueeze(0), img_path, normalize=True, value_range=(-1, 1))

                    # Track class mapping
                    image_class_mapping[img_filename] = c[i]

                t2 = time.time()
                batch_time = t2 - t1
                total_time += batch_time
                num_batches += 1

                if num % 5 == 0:
                    avg_time = total_time / num_batches if num_batches > 0 else batch_time
                    etc = avg_time * (len(dataset) - num - 1)
                    if global_rank == 0:
                        print(f"Batch {num}/{len(dataset)}, Time: {batch_time:.2f}s, "
                              f"Avg: {avg_time:.2f}s, ETC: {etc:.1f}s")

    # Each rank saves its own class mapping
    rank_mapping_path = osp.join(save_root, f'class_mapping_rank{global_rank}.json')
    with open(rank_mapping_path, 'w') as f:
        json.dump(image_class_mapping, f, indent=2)

    # Save timing statistics
    if global_rank == 0:
        stats = {
            'total_time': total_time,
            'num_batches': num_batches,
            'avg_batch_time': total_time / num_batches if num_batches > 0 else 0,
            'total_samples': num_batches * args.batch_size,
            'model_type': args.model_type,
            'cfg': args.cfg,
            'model_depth': MODEL_DEPTH
        }
        if args.model_type == "scalear":
            stats['cosine_steps'] = args.cosine_steps

        with open(osp.join(results_root, 'generation_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    # Wait for all ranks to finish
    dist.barrier()

    # Rank 0 consolidates all class mappings
    if global_rank == 0:
        consolidated_mapping = {}
        for rank in range(dist.get_world_size()):
            rank_mapping_path = osp.join(save_root, f'class_mapping_rank{rank}.json')
            if osp.exists(rank_mapping_path):
                with open(rank_mapping_path, 'r') as f:
                    rank_mapping = json.load(f)
                    consolidated_mapping.update(rank_mapping)

        # Save consolidated mapping
        with open(osp.join(save_root, 'class_mapping.json'), 'w') as f:
            json.dump(consolidated_mapping, f, indent=2)

        print(f"Consolidated class mapping saved with {len(consolidated_mapping)} images")