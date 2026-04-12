import torch
import time
import os
import numpy as np
from torch.amp import autocast, GradScaler
from engines.logger import Logger
from copy import deepcopy
from core.datasets.augmentation import GPUAugmentor

class UniversalTrainer:
    def __init__(self, algorithm, loaders, optimizer, scheduler, cfg, device, save_dir):
        self.cfg = cfg
        self.algorithm = algorithm
        self.device = device
        self.save_dir = save_dir
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.gpu_augmentor = GPUAugmentor(cfg, device)
        self.use_amp = cfg.get('use_amp', True)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        self.logger = Logger(save_dir)
        self.logger.save_config(cfg)
        self.epochs = cfg['epochs']
        self.num_classes = cfg.get('num_classes', 10)

        _SSL_ALGORITHMS = {'fixmatch', 'flexmatch', 'softmatch', 'ds3l'}
        self.is_ssl = cfg['algorithm'] in _SSL_ALGORITHMS
        self.num_it_per_epoch = cfg.get('num_it_per_epoch', 1024) if self.is_ssl else len(self.loaders['labeled'])

        self.best_acc = 0.0
        self.ema_decay = cfg.get('ema_decay', 0.999)
        self.ema_model = deepcopy(self.algorithm.model)
        for param in self.ema_model.parameters():
            param.detach_()

        self.start_epoch = 0
        self.resume_global_step = 0
        self._try_resume()

    def _try_resume(self):
        ckpt_path = os.path.join(self.save_dir, "checkpoint.pth")
        if not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.algorithm.model.load_state_dict(ckpt['model_state_dict'])
        self.ema_model.load_state_dict(ckpt['ema_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_acc = ckpt['best_acc']
        self.start_epoch = ckpt['epoch'] + 1
        self.resume_global_step = ckpt['global_step']
        self.logger.log(
            f"Resuming from epoch {self.start_epoch}/{self.epochs} "
            f"(best_acc={self.best_acc:.2f}%, global_step={self.resume_global_step})"
        )

    def _save_checkpoint(self, epoch, global_step):
        torch.save({
            'epoch':                epoch,
            'global_step':          global_step,
            'best_acc':             self.best_acc,
            'model_state_dict':     self.algorithm.model.state_dict(),
            'ema_state_dict':       self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict':    self.scaler.state_dict(),
        }, os.path.join(self.save_dir, "checkpoint.pth"))

    def _update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.algorithm.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.algorithm.model.buffers()):
                ema_buffer.copy_(buffer)

    def train_epoch(self, epoch, global_step):
        self.algorithm.model.train()
        epoch_stats = {}

        if self.is_ssl:
            if not hasattr(self, 'labeled_iter'): self.labeled_iter = iter(self.loaders['labeled'])
            if not hasattr(self, 'unlabeled_iter'): self.unlabeled_iter = iter(self.loaders['unlabeled'])

            for _ in range(self.num_it_per_epoch):
                try: x_raw, x_targets = next(self.labeled_iter)
                except StopIteration:
                    self.labeled_iter = iter(self.loaders['labeled'])
                    x_raw, x_targets = next(self.labeled_iter)

                try: u_raw, u_targets_gt = next(self.unlabeled_iter)
                except StopIteration:
                    self.unlabeled_iter = iter(self.loaders['unlabeled'])
                    u_raw, u_targets_gt = next(self.unlabeled_iter)

                with torch.no_grad():
                    # Transfer each separately with non_blocking to overlap PCIe copy and CPU work
                    x_gpu = x_raw.to(self.device, non_blocking=True)
                    u_gpu = u_raw.to(self.device, non_blocking=True)
                    inputs_x   = self.gpu_augmentor(x_gpu, mode='weak')
                    inputs_u_w = self.gpu_augmentor(u_gpu, mode='weak')
                    inputs_u_s = self.gpu_augmentor(u_gpu, mode='strong')

                data_batch = {
                    'labeled': (inputs_x, x_targets),
                    'unlabeled': ((inputs_u_w, inputs_u_s), u_targets_gt),
                    'ema_model': self.ema_model,
                }

                loss, step_stats = self._train_step(data_batch, global_step)
                self._update_ema()
                self.scheduler.step()  # per-iteration (USB standard)
                self._accumulate_stats(epoch_stats, step_stats)

                if global_step % 50 == 0:
                    self.logger.log_stats(step_stats, global_step, prefix="Train")
                global_step += 1
        else:
            for x_raw, x_targets in self.loaders['labeled']:
                with torch.no_grad():
                    inputs_x = self.gpu_augmentor(x_raw, mode='weak')
                data_batch = {'labeled': (inputs_x, x_targets)}
                loss, step_stats = self._train_step(data_batch, global_step)
                self.scheduler.step()  # per-iteration
                self._accumulate_stats(epoch_stats, step_stats)
                if global_step % 50 == 0:
                    self.logger.log_stats(step_stats, global_step, prefix="Train")
                global_step += 1

        return epoch_stats, global_step

    def _train_step(self, data_batch, global_step):
        self.optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=self.use_amp):
            loss, step_stats = self.algorithm.compute_loss(data_batch, global_step)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return loss, step_stats

    def _accumulate_stats(self, epoch_stats, step_stats):
        for k, v in step_stats.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                epoch_stats[k] = epoch_stats.get(k, 0) + v
            elif isinstance(v, np.ndarray):
                epoch_stats[k] = epoch_stats.get(k, np.zeros_like(v)) + v

    def run(self):
        self.logger.log(f"Starting experiment: {self.cfg['algorithm']} (C-Score diagnostic mode)")

        global_step = self.resume_global_step
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            stats, global_step = self.train_epoch(epoch, global_step)
            # scheduler.step() is called inside train_epoch (per-iteration)

            eval_model = self.ema_model if self.is_ssl else self.algorithm.model
            eval_res = self.algorithm.evaluate_with_model(eval_model, self.loaders['test'])
            test_acc = eval_res['acc']

            if test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(eval_model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))

            self.logger.log_stats({"accuracy": test_acc}, epoch, prefix="Eval")

            epoch_time = time.time() - start_time
            it = self.num_it_per_epoch

            log_msg = f"Epoch [{epoch+1}/{self.epochs}] {epoch_time:.1f}s Acc: {test_acc:.2f}% | Loss: {stats['loss']/it:.4f} "

            if 'id_mask_ratio' in stats: log_msg += f"ID-M: {stats['id_mask_ratio']/it:.1%} "
            if 'ood_mask_ratio' in stats: log_msg += f"OOD-M: {stats['ood_mask_ratio']/it:.1%} "
            if 'id_pseudo_acc' in stats: log_msg += f"ID-PA: {stats['id_pseudo_acc']/it:.1%} "

            if 'PLE' in stats: log_msg += f"PLE: {stats['PLE']/it:.3f} "
            if 'CCI' in stats: log_msg += f"CCI: {stats['CCI']/it:.3f} "
            if 'Sem_Drift' in stats: log_msg += f"S-Drift: {stats['Sem_Drift']/it:.3f} "
            if 'OOD_FF' in stats: log_msg += f"OOD-FF: {stats['OOD_FF']/it:.3f} "
            if 'tau_mean' in stats: log_msg += f"tau-mean: {stats['tau_mean']/it:.3f} tau-std: {stats['tau_std']/it:.3f} "
            if 'SoftW_ID' in stats: log_msg += f"SoftW-ID: {stats['SoftW_ID']/it:.3f} SoftW-OOD: {stats['SoftW_OOD']/it:.3f} "
            if 'DS3L_WID' in stats: log_msg += f"DS3L-WID: {stats['DS3L_WID']/it:.3f} DS3L-WOOD: {stats['DS3L_WOOD']/it:.3f} "
            if 'Grad_X_Norm' in stats: log_msg += f"G-x: {stats['Grad_X_Norm']/it:.4f} G-u: {stats['Grad_U_Norm']/it:.4f} G-Align: {stats['Grad_Align']/it:.3f} "
            if 'mask_ratio' in stats and 'id_mask_ratio' not in stats:
                log_msg += f"Mask: {stats['mask_ratio']/it:.1%} "

            self.logger.log(log_msg)

            if (epoch + 1) % 10 == 0:
                self.logger.log(f"   >> Per-Class Acc: {[round(float(x), 2) for x in eval_res['per_class_acc']]}")
                if 'class_mask_counts' in stats:
                    mask_dist = stats['class_mask_counts'] / it
                    self.logger.log(f"   >> Class Mask Dist: {[round(float(x), 1) for x in mask_dist]}")
                self._save_checkpoint(epoch, global_step)

        self.logger.log(f"Training complete. Best accuracy: {self.best_acc:.2f}%")
        self.logger.close()
