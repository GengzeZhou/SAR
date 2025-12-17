

<div align="center">
<h1>Rethinking Training Dynamics in Scale-wise Autoregressive Generation</h1>

<a href="https://gengzezhou.github.io/" target="_blank">Gengze Zhou</a><sup>1*</sup>,
<a href="https://chongjiange.github.io/" target="_blank">Chongjian Ge</a><sup>2</sup>,
<a href="https://www.cs.unc.edu/~airsplay/" target="_blank">Hao Tan</a><sup>2</sup>,
<a href="https://pages.cs.wisc.edu/~fliu/" target="_blank">Feng Liu</a><sup>2</sup>,
<a href="https://yiconghong.me" target="_blank">Yicong Hong</a><sup>2</sup>

<sup>1</sup>Australian Institute for Machine Learning, Adelaide University &nbsp;&nbsp;&nbsp;
<sup>2</sup>Adobe Research

[![arXiv](https://img.shields.io/badge/arXiv-2512.06421-b31b1b.svg)](https://arxiv.org/abs/2512.06421)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-SAR--ckpts-yellow)](https://huggingface.co/ZGZzz/FlexVAR-ckpts)&nbsp;
[![project page](https://img.shields.io/badge/Project%20Page-SAR-blue)](https://gengzezhou.github.io/SAR)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

<div align="center">
<img src="resources/SAR Results.png" width="958%">
</div>

## Abstract

Recent advances in autoregressive (AR) generative models have produced increasingly powerful systems for media synthesis. Among them, **next-scale prediction** has emerged as a popular paradigm, where models generate images in a coarse-to-fine manner. However, scale-wise AR models suffer from **exposure bias**, which undermines generation quality.
We identify two primary causes of this issue:
1. **Train–test mismatch**: The model relies on imperfect predictions during inference but ground truth during training.
2. **Imbalanced learning difficulty**: Coarse scales must generate global structure from scratch, while fine scales only perform easier reconstruction.

To address this, we propose **Self-Autoregressive Refinement (SAR)**. SAR introduces a **Stagger-Scale Rollout (SSR)** mechanism to expose the model to its own intermediate predictions and a **Contrastive Student-Forcing Loss (CSFL)** to ensure stable training. Experimental results show that applying SAR to pretrained AR models consistently improves generation quality with minimal computational overhead (e.g., **5.2% FID reduction** on FlexVAR-d16 within 10 epochs).


## Self-Autoregressive Refinement (SAR)

<div align="center">
<img src="resources/SSR.jpg" width="958%">
</div>

SAR is a lightweight post-training algorithm that bridges the train-test gap. It consists of two key components:

### 1. Stagger-Scale Rollout (SSR)

SSR is a two-step rollout strategy that is computationally efficient (requiring only one extra forward pass):

- **Step 1 (Teacher Forcing):** The model performs teacher forcing and predicts at all scales using ground-truth conditioning.
- **Step 2 (Student Forcing):** These predictions are upsampled to form scale-shifted inputs, enabling a second forward pass that produces student-forced predictions.

Teacher-forcing loss provides ground-truth supervision, while the contrastive student-forcing loss aligns student-forced outputs with their teacher-forced counterparts.

### 2. Contrastive Student-Forcing Loss (CSFL)

Naive student forcing often causes the model to drift away from the ground truth. To fix this, CSFL:

- Aligns the **student prediction** with the **stable teacher prediction** instead of forcing the student prediction to match the ground truth (which causes conflicts).
- Teaches the model to remain consistent with the "expert" trajectory even when conditioned on imperfect inputs.

## Installation

### Option 1: Using uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

# (Optional) Install flash-attn and xformers for faster attention
uv pip install flash-attn xformers
```

### Option 2: Using conda

```bash
conda create -n sar python=3.10 -y
conda activate sar
pip install -r requirements.txt

# (Optional) Install flash-attn and xformers for faster attention
pip install flash-attn xformers
```

> **Note:** Always activate your environment (`source .venv/bin/activate` for uv or `conda activate sar` for conda) before running training/evaluation scripts.

### Data Preparation

Download the [ImageNet](http://image-net.org/) dataset.

Assume the ImageNet is in `/path/to/imagenet`. It should look like this:

```
/path/to/imagenet/:
    train/:
        n01440764:
            many_images.JPEG ...
        n01443537:
            many_images.JPEG ...
    val/:
        n01440764:
            ILSVRC2012_val_00000293.JPEG ...
        n01443537:
            ILSVRC2012_val_00000236.JPEG ...
```

> **Note:** The arg `--data_path=/path/to/imagenet` should be passed to the training script.




## Model Zoo

### VQVAE Tokenizer

Download [FlexVAE.pth](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVAE.pth) first and place it at `pretrained/FlexVAE.pth`.

### Pretrained Checkpoints (ImageNet 256×256)

| Model | Params | FID ↓ | IS ↑ | Weights |
|-------|--------|-------|------|---------|
| FlexVAR-d16 | 310M | 3.05 | 291.3 | [FlexVARd16-epo179.pth](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVARd16-epo179.pth) |
| **FlexVAR-d16 + SAR** | 310M | **2.89** | 266.6 | [SARd16-epo179.pth](https://huggingface.co/ZGZzz/SAR/resolve/main/SARd16-epo179.pth) |
| FlexVAR-d20 | 600M | 2.41 | 299.3 | [FlexVARd20-epo249.pth](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVARd20-epo249.pth) |
| **FlexVAR-d20 + SAR** | 600M | **2.35** | 293.3 | [SARd20-epo249.pth](https://huggingface.co/ZGZzz/SAR/resolve/main/SARd20-epo249.pth) |
| FlexVAR-d24 | 1.0B | 2.21 | 299.1 | [FlexVARd24-epo349.pth](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVARd24-epo349.pth) |
| **FlexVAR-d24 + SAR** | 1.0B | **2.14** | 315.5 | [SARd24-epo349.pth](https://huggingface.co/ZGZzz/SAR/resolve/main/SARd24-epo349.pth) |

## SAR Training (Self-Autoregressive Refinement)

Train SAR with Contrastive Student-Forcing Loss (CSFL):

```bash
# SAR d16 (depth 16)
bash scripts/train_SAR_d16.sh

# SAR d20 (depth 20)
bash scripts/train_SAR_d20.sh

# SAR d24 (depth 24)
bash scripts/train_SAR_d24.sh
```

> **Note:** (1) For simplicity, instead of training a FlexVAR model for 170 epochs and then applying SAR for 10 epochs, in the following examples we directly download the pretrained FlexVAR checkpoint (e.g., 180 epochs for d16) and train SAR for 10 additional epochs. (2) The [`scalear_trainer.py`](scalear_trainer.py) implements the hybrid modeling model described in **Section 3.3** of our paper, and is not used in the main experiment.

### Logging with Weights & Biases

To enable [Weights & Biases](https://wandb.ai/) logging during training:

```bash
# Option 1: Interactive login (one-time setup)
wandb login

# Option 2: Using API token (useful for servers/automation)
export WANDB_API_KEY=your-api-key-here
# Or pass it directly: --wandb_api_key=your-api-key-here

# Then add these flags to your training command
torchrun ... train.py \
    --logger_type=wandb \
    --wandb_project=sar \
    --wandb_entity=your-team-name \    # optional
    --wandb_run_name=my-experiment \   # optional, auto-generated if not provided
    --wandb_tags=d16,csfl \            # optional, comma-separated
    ...
```

Available wandb arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--logger_type` | `tensorboard` | Set to `wandb` to enable wandb logging |
| `--wandb_project` | `sar` | Wandb project name |
| `--wandb_entity` | `None` | Wandb team/entity name (optional) |
| `--wandb_run_name` | `None` | Custom run name (auto-generated if not set) |
| `--wandb_tags` | `None` | Comma-separated tags for the run |
| `--wandb_notes` | `None` | Notes for the run |

## Evaluation

First, setup the evaluation environment:
```bash
bash scripts/setup_eval.sh
```

For FID evaluation, images are sampled and saved as PNG files. Use the [OpenAI's FID evaluation toolkit](https://github.com/openai/guided-diffusion/tree/main/evaluations) with reference ground truth npz file of [256×256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) to evaluate FID, IS, precision, and recall. See [Evaluation](utils/evaluations/c2i/README.md) for details.

Run evaluation:
```bash
# Evaluate SAR d16
bash scripts/eval_SAR_d16.sh

# Evaluate SAR d20
bash scripts/eval_SAR_d20.sh

# Evaluate SAR d24
bash scripts/eval_SAR_d24.sh
```


## Inference Demo

Generate sample images with a trained model:

```python
import torch
from models import build_vae_var

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vae, model = build_vae_var(
    V=8912, Cvae=32, device=device,
    num_classes=1000, depth=16,
    vae_ckpt='pretrained/FlexVAE.pth'
)

# Load checkpoint
ckpt = torch.load('pretrained/SARd16-epo179.pth', map_location='cpu')
if 'trainer' in ckpt:
    ckpt = ckpt['trainer']['var_wo_ddp']
model.load_state_dict(ckpt, strict=False)
model.eval()

# Generate images
with torch.no_grad():
    # Class labels (e.g., 207=golden retriever, 88=parrot)
    labels = torch.tensor([207, 88, 360, 387], device=device)

    images = model.autoregressive_infer_cfg(
        vqvae=vae,
        B=4,
        label_B=labels,
        cfg=2.5,      # classifier-free guidance scale
        top_k=900,    # top-k sampling
        top_p=0.95,   # nucleus sampling
    )

# Save images
from torchvision.utils import save_image
save_image(images, 'samples.png', normalize=True, value_range=(-1, 1), nrow=4)
```

A complete inference demo is provided at [`demo_inference.ipynb`](demo_inference.ipynb).


## Acknowledgement

This codebase is built upon [VAR](https://github.com/FoundationVision/VAR), [FlexVAR](https://github.com/jiaosiyu1999/FlexVAR). Our thanks go out to the creators of these outstanding projects.

## Citation

```bibtex
@article{zhou2025rethinking,
  title={Rethinking Training Dynamics in Scale-wise Autoregressive Generation},
  author={Zhou, Gengze and Ge, Chongjian and Tan, Hao and Liu, Feng and Hong, Yicong},
  journal={arXiv preprint arXiv:2512.06421},
  year={2025}
}
```