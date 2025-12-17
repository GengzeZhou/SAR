import logging
import torch.distributed as dist
import os
import time
from typing import Optional, Dict, Any, Union
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from torch.utils.tensorboard import SummaryWriter

logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor.utils").setLevel(logging.CRITICAL)
logging.getLogger("torch._dynamo.symbolic_convert").setLevel(logging.CRITICAL)
logging.getLogger("torch._dynamo.output_graph").setLevel(logging.CRITICAL)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class UnifiedLogger:
    """Unified logger that can use either TensorBoard or Weights & Biases"""
    
    def __init__(self, logger_type: str, log_dir: str, args=None):
        self.logger_type = logger_type.lower()
        self.step = 0
        self.log_dir = log_dir
        
        if self.logger_type == 'wandb':
            if not WANDB_AVAILABLE:
                raise ImportError(
                    "wandb is not installed. Please install it with: pip install wandb"
                )
            
            # Set WANDB_API_KEY from args if provided
            if args and args.wandb_api_key:
                os.environ["WANDB_API_KEY"] = args.wandb_api_key
            
            # Check for WANDB_API_KEY
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
            if WANDB_API_KEY is None or len(WANDB_API_KEY) <= len("local-"):
                print("Warning: No valid WANDB_API_KEY environment variable found.")
                print("Please set it with: export WANDB_API_KEY=your_api_key")
                print("Or run: wandb login")
                print("Attempting to use wandb without API key (may require interactive login)...")
            
            # Login to wandb (will use WANDB_API_KEY if available)
            try:
                # Check if custom wandb host is set
                wandb_host = os.environ.get("WANDB_HOST", None)
                if args and args.wandb_host:
                    wandb_host = args.wandb_host
                
                if wandb_host:
                    wandb.login(host=wandb_host)
                else:
                    wandb.login()
            except Exception as e:
                print(f"Warning: wandb login failed: {e}")
                print("Continuing without explicit login...")
            
            # Initialize wandb
            wandb_config = {}
            if args is not None:
                # Convert args to dict for wandb config
                wandb_config = args.state_dict()
                
                # Parse tags if provided
                tags = []
                if args.wandb_tags:
                    tags = [tag.strip() for tag in args.wandb_tags.split(',')]
                
                # Generate run name if not provided
                run_name = args.wandb_run_name
                if not run_name:
                    run_name = f"VARd{args.depth}_ep{args.ep}_bs{args.bs}_{time.strftime('%Y%m%d_%H%M%S')}"
                
                # Handle run resumption
                run_id = None
                resume_mode = "allow"
                
                # Check if we should try to reuse a previous run
                run_uuid_file = os.path.join(log_dir, "wandb_run_uuid.txt")
                os.makedirs(log_dir, exist_ok=True)
                
                try:
                    if os.path.exists(run_uuid_file):
                        with open(run_uuid_file, 'r') as f:
                            run_id = f.read().strip()
                        if run_id:
                            print(f"Resuming wandb run with ID: {run_id}")
                            resume_mode = "must"  # Force resume if we have a run ID
                except Exception as e:
                    print(f"Error reading wandb run ID from {run_uuid_file}: {e}")
                
                # If no existing run ID, generate a new one
                if run_id is None:
                    run_id = wandb.util.generate_id()
                    try:
                        with open(run_uuid_file, 'w') as f:
                            f.write(run_id)
                        print(f"Starting new wandb run with ID: {run_id}")
                    except Exception as e:
                        print(f"Warning: Could not save wandb run ID to {run_uuid_file}: {e}")
                
                # Initialize wandb run
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    config=wandb_config,
                    tags=tags,
                    notes=args.wandb_notes,
                    dir=log_dir,
                    id=run_id,
                    resume=resume_mode
                )

            else:
                # Fallback initialization without args
                wandb.init(project='flexvar', dir=log_dir, resume='allow')
            
            self.writer = None
            
        elif self.logger_type == 'tensorboard':
            # Initialize TensorBoard
            filename_suffix = f'__{time.strftime("%m%d_%H%M")}'
            self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
            
        else:
            raise ValueError(f"Unknown logger type: {self.logger_type}. Choose 'tensorboard' or 'wandb'")
    
    def set_step(self, step: Optional[int] = None):
        """Set the global step for logging"""
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def update(self, head: str = 'scalar', step: Optional[int] = None, **kwargs):
        """Log scalar values"""
        for k, v in kwargs.items():
            if v is None:
                continue
            
            if hasattr(v, 'item'):
                v = v.item()
            
            if self.logger_type == 'wandb':
                log_step = step if step is not None else self.step
                wandb.log({f'{head}/{k}': v}, step=log_step)
            
            elif self.logger_type == 'tensorboard':
                if step is None:  # iter wise
                    it = self.step
                    if it == 0 or (it + 1) % 500 == 0:
                        self.writer.add_scalar(f'{head}/{k}', v, it)
                else:  # epoch wise
                    self.writer.add_scalar(f'{head}/{k}', v, step)
    
    def log_tensor_as_distri(self, tag: str, tensor1d: torch.Tensor, step: Optional[int] = None):
        """Log tensor as distribution/histogram"""
        if step is None:
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:
            loggable = True
            
        if loggable:
            if self.logger_type == 'wandb':
                wandb.log({tag: wandb.Histogram(tensor1d.cpu().numpy())}, step=step)
            elif self.logger_type == 'tensorboard':
                try:
                    self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
                except Exception as e:
                    print(f'[log_tensor_as_distri writer.add_histogram failed]: {e}')
    
    def log_image(self, tag: str, img_chw: torch.Tensor, step: Optional[int] = None, caption: Optional[str] = None):
        """Log image with optional caption"""
        if step is None:
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:
            loggable = True
            
        if loggable:
            if self.logger_type == 'wandb':
                # Convert CHW to HWC for wandb
                img_hwc = img_chw.permute(1, 2, 0).cpu().numpy()
                wandb.log({tag: wandb.Image(img_hwc, caption=caption)}, step=step)
            elif self.logger_type == 'tensorboard':
                self.writer.add_image(tag, img_chw, step, dataformats='CHW')
                # TensorBoard doesn't support captions directly, log as text if provided
                if caption:
                    self.writer.add_text(f"{tag}_caption", caption, step)
    
    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log a dictionary of metrics"""
        if self.logger_type == 'wandb':
            log_step = step if step is not None else self.step
            wandb.log(metrics, step=log_step)
        elif self.logger_type == 'tensorboard':
            for k, v in metrics.items():
                if v is not None:
                    if hasattr(v, 'item'):
                        v = v.item()
                    if step is None:
                        self.writer.add_scalar(k, v, self.step)
                    else:
                        self.writer.add_scalar(k, v, step)
    
    def flush(self):
        """Flush the logger"""
        if self.logger_type == 'tensorboard' and self.writer is not None:
            self.writer.flush()
        # wandb automatically handles flushing
    
    def close(self):
        """Close the logger"""
        if self.logger_type == 'wandb':
            wandb.finish()
        elif self.logger_type == 'tensorboard' and self.writer is not None:
            self.writer.close()
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Watch model gradients and parameters (wandb only)"""
        if self.logger_type == 'wandb':
            wandb.watch(model, log='all', log_freq=log_freq)


class DistLogger:
    """Distributed logger wrapper that only logs on the master process"""
    
    def __init__(self, logger: Optional[UnifiedLogger], verbose: bool):
        self._logger = logger
        self._verbose = verbose
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        if self._verbose and self._logger is not None:
            return getattr(self._logger, attr)
        else:
            return DistLogger.do_nothing