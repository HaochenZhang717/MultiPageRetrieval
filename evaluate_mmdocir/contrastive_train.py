import os
import argparse
import warnings
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from Cython.Compiler.Options import embed_modules
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
from transformers import get_cosine_schedule_with_warmup
from MMDocIRDataset import PageRetrievalDataset
from embed_models import DeepseekOCREmbeddingModel, MultiPageEmbedding


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _reduce_scalar(x: float, device: torch.device):
    t = torch.tensor(x, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()

def denorm_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)

def parse_betas(betas_str: str):
    a, b = betas_str.split(",")
    return float(a), float(b)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# =========================================================
# Args
# =========================================================
def get_args():
    parser = argparse.ArgumentParser(description="Finetune Deepseek OCR into an embedding model")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6.4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=str, default="0.9,0.99")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_percent", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--wandb_project", type=str, default="dc-ae-finetune")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="ckpts_deblur_unet")


    parser.add_argument("--data_domains", type=str, nargs="+",
                        default=["q_proj", "v_proj"],
                        help="List of domains to load data from")

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--image_input_size", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)

    # Lora parameters
    parser.add_argument("--init_weights", type=str, default="default", help="How to initialize LoRA weights")
    parser.add_argument("--target_modules", type=str, nargs="+", help="List of module names to apply LoRA to")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_dim", type=int, default=8,
                        help="LoRA rank dimension r")
    parser.add_argument("--use_dora", action="store_true",
                        help="Enable DoRA instead of LoRA")

    return parser.parse_args()



# =========================================================
# Main
# =========================================================
def main():
    args = get_args()

    # --------------- DDP init ----------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    set_seed(args.seed)

    # --------------- Dataset -----------------

    full_dataset = PageRetrievalDataset(
        data_path="./MMDocIRDataset",
        dataset_domain=args.data_domains,
        unified_size=(args.image_input_size, args.image_input_size)
    )

    n_total = len(full_dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    # Create a generator with a fixed seed
    g = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=g)

    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=2,
                            pin_memory=True, drop_last=False)

    # --------------- Model -------------------
    if args.model_version == "v1":
        embed_model = DeepseekOCREmbeddingModel(
            init_weights=args.init_weights,
            target_modules=args.target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_dim=args.lora_dim,
            use_dora=args.use_dora,
            debug= False,
            embed_dim=args.embed_dim,
            temperature=args.temperature
        ).to(device)
        embed_model.prepare_for_finetune()
    elif args.model_version == "v2":
        embed_model = MultiPageEmbedding(
            ?????
        )



    # --------------- Optim / Sched -----------
    betas = parse_betas(args.betas)

    opt = optim.AdamW(filter(lambda p: p.requires_grad, embed_model.parameters()),
                      lr=args.lr, betas=betas, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_percent)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    embed_model = DDP(embed_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    # --------------- W&B ---------------------
    if rank == 0:
        ensure_dir(args.ckpt_dir)
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    global_step = 0

    # ======================================================
    # Training
    # ======================================================
    for epoch in range(args.epochs):
        embed_model.train()
        train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        opt.zero_grad()

        for queries, pos_images, neg_images in pbar:
            pos_images = pos_images.to(dtype=torch.bfloat16, device=device)
            neg_images = neg_images.to(dtype=torch.bfloat16, device=device)

            total_loss = embed_model(queries, pos_images, neg_images, return_outputs=False)

            total_loss.backward()

            global_step += 1

            torch.nn.utils.clip_grad_norm_(embed_model.parameters(), args.grad_clip_norm)
            opt.step()
            sched.step()
            opt.zero_grad()
            if rank == 0:
                wandb.log({
                    "step": global_step,
                    "lr": sched.get_last_lr()[0],
                    "train/total_loss_step": total_loss.item(),
                })

        # ----------------------
        # Evaluation
        # ----------------------
        val_sampler.set_epoch(epoch)
        pbar = tqdm(val_loader, desc=f"Eval Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        embed_model.eval()
        va_seen = 0
        va_loss = 0
        for queries, pos_images, neg_images in pbar:
            pos_images = pos_images.to(dtype=torch.bfloat16, device=device)
            neg_images = neg_images.to(dtype=torch.bfloat16, device=device)
            with torch.no_grad():
                total_loss = embed_model(queries, pos_images, neg_images, return_outputs=False)

            va_seen += len(queries)
            va_loss += total_loss.item() * len(queries)

        seen_val = _reduce_scalar(va_seen, device)
        val_total = _reduce_scalar(va_loss, device) / max(1, seen_val)

        # -----------------------
        # Logging + ckpt
        # -----------------------
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "val/total_loss": val_total})
            embed_model.module.save_model(f"{args.ckpt_dir}-Epoch{epoch+1}")


    dist.destroy_process_group()


if __name__ == "__main__":
    main()