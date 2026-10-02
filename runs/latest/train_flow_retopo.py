#!/usr/bin/env python3
"""
GPU Training Engine & Scaled Model Recipe for Flow Retopology
==============================================================
Phase 3: Scaling, Kinematic Conditioning & Production Retopology
Project Code: MESH-FLOW-RETOPOLOGY

This module implements the complete PyTorch training harness:
1. Multi-Modal Conditional Flow Matching Backbone (FlowRetopoDiT)
2. Joint Objective: L = L_CFM + lambda_strain * L_strain (with dynamic schedule)
3. AdamW Optimizer with Cosine Annealing Learning Rate Decay & Linear Warmup
4. Mixed Precision Training (CUDA AMP / CPU autocast fallback)
5. Gradient Clipping (norm <= 1.0) & Numerical Stability Guards
6. Rig-Retopo-3K Ingestion & Synthetic Streaming
7. Checkpoint Management & Validation Loss Evaluation
"""

import os
import sys
import time
import math
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flow_retopo_model import FlowRetopoDiT, CombinedFlowLoss
from dataset_pipeline import RigRetopoDataset, rig_retopo_collate_fn


# ==============================================================================
# 1. Warmup + Cosine Decay Learning Rate Scheduler
# ==============================================================================

class CosineWarmupScheduler:
    """Linear warmup followed by cosine annealing learning rate schedule."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.05
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            alpha = float(self.current_step) / float(self.warmup_steps)
        else:
            progress = float(self.current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            alpha = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * alpha

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


# ==============================================================================
# 2. Complete Flow Retopology Trainer
# ==============================================================================

class FlowRetopoTrainer:
    """
    Production-grade training harness for Conditional Flow Matching Retopology.
    """
    def __init__(
        self,
        model: FlowRetopoDiT,
        loss_fn: CombinedFlowLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWarmupScheduler,
        device: torch.device,
        save_dir: str = "checkpoints",
        use_amp: bool = True,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.use_amp = use_amp and (device.type == "cuda")
        self.max_grad_norm = max_grad_norm
        
        # AMP Scaler for GPU mixed precision
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.history = {
            "step": [],
            "loss_total": [],
            "loss_cfm": [],
            "loss_strain": [],
            "lr": [],
            "grad_norm": [],
            "step_time_ms": []
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        t0 = time.perf_counter()
        
        s_1 = batch["s_1"].to(self.device) # (B, N, 14)
        high_points = batch["high_points"].to(self.device)
        high_normals = batch["high_normals"].to(self.device)
        joint_positions = batch["joint_positions"].to(self.device)
        skinning_weights = batch["skinning_weights"].to(self.device)
        skel_adj = batch["skel_adj"].to(self.device)
        quad_indices = batch["quad_indices"].to(self.device)
        v1_axes = batch["v1_axes"].to(self.device)
        v2_axes = batch["v2_axes"].to(self.device)
        
        B, N, state_dim = s_1.shape
        t = torch.rand(B, device=self.device)
        s_0 = torch.randn_like(s_1)
        
        t_expand = t[:, None, None]
        s_t = (1.0 - t_expand) * s_0 + t_expand * s_1
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            v_pred = self.model(
                s_t=s_t,
                t=t,
                high_points=high_points,
                high_normals=high_normals,
                joint_positions=joint_positions,
                skinning_weights=skinning_weights,
                skel_adj=skel_adj
            )
            
            total_loss, loss_dict = self.loss_fn(
                v_pred=v_pred,
                s_0=s_0,
                s_1=s_1,
                quad_indices=quad_indices,
                v1_axes=v1_axes,
                v2_axes=v2_axes
            )
            
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        self.scheduler.step()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_cfm": float(loss_dict["loss_cfm"]),
            "loss_strain": float(loss_dict["loss_strain"]),
            "lr": float(self.scheduler.get_last_lr()[0]),
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "step_time_ms": float(dt_ms)
        }
        return metrics

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss_accum = 0.0
        cfm_loss_accum = 0.0
        strain_loss_accum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                s_1 = batch["s_1"].to(self.device)
                high_points = batch["high_points"].to(self.device)
                high_normals = batch["high_normals"].to(self.device)
                joint_positions = batch["joint_positions"].to(self.device)
                skinning_weights = batch["skinning_weights"].to(self.device)
                skel_adj = batch["skel_adj"].to(self.device)
                quad_indices = batch["quad_indices"].to(self.device)
                v1_axes = batch["v1_axes"].to(self.device)
                v2_axes = batch["v2_axes"].to(self.device)
                
                B, N, _ = s_1.shape
                t = torch.rand(B, device=self.device)
                s_0 = torch.randn_like(s_1)
                t_expand = t[:, None, None]
                s_t = (1.0 - t_expand) * s_0 + t_expand * s_1
                
                v_pred = self.model(
                    s_t=s_t,
                    t=t,
                    high_points=high_points,
                    high_normals=high_normals,
                    joint_positions=joint_positions,
                    skinning_weights=skinning_weights,
                    skel_adj=skel_adj
                )
                
                tot, l_dict = self.loss_fn(
                    v_pred=v_pred,
                    s_0=s_0,
                    s_1=s_1,
                    quad_indices=quad_indices,
                    v1_axes=v1_axes,
                    v2_axes=v2_axes
                )
                
                total_loss_accum += tot.item()
                cfm_loss_accum += l_dict["loss_cfm"]
                strain_loss_accum += l_dict["loss_strain"]
                num_batches += 1
                
        num_batches = max(1, num_batches)
        return {
            "val_loss_total": total_loss_accum / num_batches,
            "val_loss_cfm": cfm_loss_accum / num_batches,
            "val_loss_strain": strain_loss_accum / num_batches
        }

    def save_checkpoint(self, step: int, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"flow_retopo_step_{step}.pt")
            
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "state_dim": self.model.state_dim,
                "hidden_dim": self.model.hidden_dim
            }
        }, filepath)
        return filepath

    def load_checkpoint(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt


# ==============================================================================
# 3. Scaled Training Execution Script
# ==============================================================================

def run_training_experiment(
    total_steps: int = 150,
    batch_size: int = 4,
    lr: float = 3e-4,
    lambda_strain: float = 0.25,
    hidden_dim: int = 64,
    num_layers: int = 3,
    save_dir: str = "checkpoints"
) -> Dict[str, Any]:
    print("=" * 85)
    print("EXECUTING WP 3.2: FLOW RETOPOLOGY TRAINING & OPTIMIZATION HARNESS")
    print(f"Total Steps: {total_steps} | Batch Size: {batch_size} | Base LR: {lr:.1e} | Lambda Strain: {lambda_strain}")
    print("=" * 85)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {device} (Type: {device.type})")
    
    dataset = RigRetopoDataset(num_samples=max(64, total_steps * batch_size), archetypes=["cylindrical_joint"])
    val_dataset = RigRetopoDataset(num_samples=16, archetypes=["cylindrical_joint"])
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=rig_retopo_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=rig_retopo_collate_fn)
    
    model = FlowRetopoDiT(
        state_dim=14,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=4,
        mlp_ratio=2.0
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized FlowRetopoDiT Backbone: {param_count:,} trainable parameters")
    
    loss_fn = CombinedFlowLoss(lambda_strain=lambda_strain)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=max(10, total_steps // 10), total_steps=total_steps)
    
    trainer = FlowRetopoTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir
    )
    
    print("\n[PROGRESS] Commencing Training Steps...")
    print(f"{'Step':<8} | {'Total Loss':<12} | {'CFM Loss':<12} | {'Strain Loss':<12} | {'Grad Norm':<10} | {'LR':<10} | {'Time (ms)':<10}")
    print("-" * 85)
    
    data_iter = iter(train_loader)
    initial_loss = None
    final_loss = None
    
    for step in range(1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
            
        metrics = trainer.train_step(batch)
        
        trainer.history["step"].append(step)
        trainer.history["loss_total"].append(metrics["loss_total"])
        trainer.history["loss_cfm"].append(metrics["loss_cfm"])
        trainer.history["loss_strain"].append(metrics["loss_strain"])
        trainer.history["lr"].append(metrics["lr"])
        trainer.history["grad_norm"].append(metrics["grad_norm"])
        trainer.history["step_time_ms"].append(metrics["step_time_ms"])
        
        if step == 1:
            initial_loss = metrics["loss_total"]
            
        if step == 1 or step % (total_steps // 10 or 1) == 0 or step == total_steps:
            final_loss = metrics["loss_total"]
            print(
                f"{step:<8} | {metrics['loss_total']:<12.5f} | {metrics['loss_cfm']:<12.5f} | "
                f"{metrics['loss_strain']:<12.5f} | {metrics['grad_norm']:<10.4f} | "
                f"{metrics['lr']:<10.2e} | {metrics['step_time_ms']:<10.1f}"
            )
            
    print("\n[VALIDATION] Evaluating Final Model on Hold-Out Validation Set...")
    val_metrics = trainer.evaluate(val_loader)
    print(f"  Validation Total Loss:  {val_metrics['val_loss_total']:.5f}")
    print(f"  Validation CFM Loss:    {val_metrics['val_loss_cfm']:.5f}")
    print(f"  Validation Strain Loss: {val_metrics['val_loss_strain']:.5f}")
    
    ckpt_path = trainer.save_checkpoint(total_steps)
    print(f"  Saved Final Checkpoint: {ckpt_path}")
    
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100.0
    print(f"\n[SUMMARY] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Reduction: {loss_reduction:.2f}%)")
    assert final_loss < initial_loss, "Model failed to minimize training loss!"
    assert all(not math.isnan(l) for l in trainer.history["loss_total"]), "NaN encountered in training loss!"
    
    print("\n" + "=" * 85)
    print("WP 3.2 TRAINING HARNESS VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 85)
    
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction,
        "val_metrics": val_metrics,
        "history": trainer.history,
        "checkpoint": ckpt_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Retopology Model")
    parser.add_argument("--steps", type=int, default=150, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--lambda-strain", type=float, default=0.25, help="Weight for 4-RoSy strain regularizer")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()
    
    run_training_experiment(
        total_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_strain=args.lambda_strain,
        save_dir=args.save_dir
    )