import torch, os, random, matplotlib.pyplot as plt, numpy as np
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from typing import Optional, Callable, Union
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd

"""
Training module for supervised and PINN models.

This module defines the `Training` class used to train, validate, and test
PyTorch-based PINN models with optional physics-residual and
boundary-residual constraints. The class supports:
- supervised data loss evaluation,
- optional physics and boundary collocation losses,
- adaptive loss-weight updates based on gradient norms,
- model checkpointing using validation performance, and
- storage of training, validation, and gradient histories.

The implementation is designed for PINN workflows in which the total
training objective may combine labeled-data loss with additional residual-based
penalty terms.
"""


class Training(object):
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]] = None,
        phys_coll_loader: Optional[torch.utils.data.DataLoader] = None,
        bnd_coll_loader: Optional[torch.utils.data.DataLoader] = None,
        phys_residual: Optional[Callable] = None,
        bnd_residual: Optional[Callable] = None,
        ckpt_path: str = "./logs/best_model.pth",
        phys_weight: float = 0.0,
        bnd_weight: float = 0.0,
        adapt_wts: bool = False,
        theta: Optional[torch.nn.Parameter] = None,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.phys_coll_loader = phys_coll_loader
        self.bnd_coll_loader = bnd_coll_loader
        self.phys_residual = phys_residual
        self.bnd_residual = bnd_residual
        self.ckpt_path = ckpt_path
        self.phys_weight = phys_weight
        self.bnd_weight = bnd_weight
        self.adapt_wts = adapt_wts
        self.theta = theta

        self.wt_update_every = 10
        self.wt_ema = 0.1

        self.grad_data_hist = []
        self.grad_phys_hist = []
        self.grad_bnd_hist = []
        self.all_phys_colloc_batches = (
            [b[0] for b in self.phys_coll_loader]
            if self.phys_coll_loader is not None
            else []
        )
        self.all_bnd_colloc_batches = (
            [b[0] for b in self.bnd_coll_loader]
            if self.bnd_coll_loader is not None
            else []
        )
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def grad_norm(self, loss):
        if loss is not None:
            g_ = torch.autograd.grad(
                loss, self.trainable_params, retain_graph=True, allow_unused=True
            )
            g_ = [g for g in g_ if g is not None]
            norm = (
                torch.norm(torch.stack([g.norm() for g in g_]))
                if g_
                else torch.tensor(0.0, device=self.device)
            )
        else:
            norm = torch.tensor(0.0, device=self.device)
        return norm

    def train_epoch(self):
        self.model.train()
        phys_coll_iter = cycle(
            self.phys_coll_loader if self.phys_coll_loader is not None else []
        )
        bnd_coll_iter = cycle(
            self.bnd_coll_loader if self.bnd_coll_loader is not None else []
        )

        tot_loss_sum, data_loss_sum, phys_loss_sum, bnd_loss_sum = 0.0, 0.0, 0.0, 0.0
        n_samples, n_phys_colloc, n_bnd_colloc = 0, 0, 0
        grad_d_sum, grad_p_sum, grad_b_sum = 0.0, 0.0, 0.0
        grad_count = 0

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            batch_size = x_batch.size(0)

            y_pred = self.model(x_batch)
            ld = self.loss_fn(y_pred, y_batch)

            if (
                self.phys_weight != 0
                and self.phys_residual is not None
                and self.phys_coll_loader is not None
            ):
                x_phys = next(phys_coll_iter)[0].to(self.device).requires_grad_(True)
                phys_coll_size = x_phys.size(0)
                y_phys_pred = self.model(x_phys)
                res_phys = self.phys_residual(x_phys, y_phys_pred)  # type: ignore
                lp = torch.mean(res_phys**2)
            else:
                phys_coll_size = 0
                lp = None

            if (
                self.bnd_weight != 0
                and self.bnd_residual is not None
                and self.bnd_coll_loader is not None
            ):
                x_bnd = next(bnd_coll_iter)[0].to(self.device).requires_grad_(True)
                bnd_coll_size = x_bnd.size(0)
                y_bnd_pred = self.model(x_bnd)
                res_bnd = self.bnd_residual(x_bnd, y_bnd_pred)
                lb = torch.mean(res_bnd**2)
            else:
                bnd_coll_size = 0
                lb = None

            norm_data_loss = self.grad_norm(ld)
            norm_phys_loss = self.grad_norm(lp)
            norm_bnd_loss = self.grad_norm(lb)

            grad_d_sum += norm_data_loss.item()
            grad_p_sum += norm_phys_loss.item()
            grad_b_sum += norm_bnd_loss.item()
            grad_count += 1

            loss = (
                ld
                + (self.phys_weight * lp if lp is not None else 0.0)
                + (self.bnd_weight * lb if lb is not None else 0.0)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accounting losses
            tot_loss_sum += loss.item() * batch_size
            data_loss_sum += ld.item() * batch_size
            phys_loss_sum += lp.item() * phys_coll_size if lp is not None else 0.0
            bnd_loss_sum += lb.item() * bnd_coll_size if lb is not None else 0.0

            n_samples += batch_size
            n_phys_colloc += phys_coll_size
            n_bnd_colloc += bnd_coll_size

        avg_total = tot_loss_sum / n_samples if n_samples else 0.0
        avg_data = data_loss_sum / n_samples if n_samples else 0.0
        avg_phys = phys_loss_sum / n_phys_colloc if n_phys_colloc else 0.0
        avg_bnd = bnd_loss_sum / n_bnd_colloc if n_bnd_colloc else 0.0

        avg_grad_d = grad_d_sum / grad_count if grad_count else 0.0
        avg_grad_p = grad_p_sum / grad_count if grad_count else 0.0
        avg_grad_b = grad_b_sum / grad_count if grad_count else 0.0

        return {
            "total_loss": avg_total,
            "data_loss": avg_data,
            "phys_loss": avg_phys,
            "bnd_loss": avg_bnd,
            "avg_grad_d": avg_grad_d,
            "avg_grad_p": avg_grad_p,
            "avg_grad_b": avg_grad_b,
        }

    def validate_test(
        self,
        loader: torch.utils.data.DataLoader,
        phys_colloc_frac: float = 0.25,
        bnd_colloc_frac: float = 0.25,
    ):
        self.model.eval()

        tot_data_loss, n_data = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                batch_size = x_batch.size(0)
                y_pred = self.model(x_batch)
                ld = self.loss_fn(y_pred, y_batch)
                tot_data_loss += ld.item() * batch_size
                n_data += batch_size
        avg_data = tot_data_loss / n_data if n_data else 0.0

        tot_phys_loss, n_phys = 0.0, 0
        n_phys_coll_total = len(self.all_phys_colloc_batches)
        phys_frac = max(0.0, min(1.0, float(phys_colloc_frac)))
        n_sample = min(n_phys_coll_total, int(round(n_phys_coll_total * phys_frac)))
        if (
            self.phys_weight != 0
            and self.phys_residual is not None
            and len(self.all_phys_colloc_batches) > 0
            and n_sample > 0
        ):
            sampled_phys_coll_batches = random.sample(
                self.all_phys_colloc_batches, n_sample
            )

            for x_phys_batch in sampled_phys_coll_batches:
                x_phys = x_phys_batch.to(self.device).requires_grad_(True)
                phys_coll_size = x_phys.size(0)
                y_phys = self.model(x_phys)
                res_phys = self.phys_residual(x_phys, y_phys)
                lp = torch.mean(res_phys**2)
                tot_phys_loss += lp.item() * phys_coll_size
                n_phys += phys_coll_size
        avg_phys = tot_phys_loss / n_phys if n_phys else 0.0

        tot_bnd_loss, n_bnd = 0.0, 0
        n_bnd_coll_total = len(self.all_bnd_colloc_batches)
        bnd_frac = max(0.0, min(1.0, float(bnd_colloc_frac)))
        n_sample = max(
            1, min(n_bnd_coll_total, int(round(n_bnd_coll_total * bnd_frac)))
        )
        if (
            self.bnd_weight != 0
            and self.bnd_residual is not None
            and len(self.all_bnd_colloc_batches) > 0
            and n_sample > 0
        ):
            sampled_bnd_coll_batches = random.sample(
                self.all_bnd_colloc_batches, n_sample
            )
            for x_bnd_batch in sampled_bnd_coll_batches:
                x_bnd = x_bnd_batch.to(self.device).requires_grad_(True)
                bnd_coll_size = x_bnd.size(0)
                y_bnd = self.model(x_bnd)
                res_bnd = self.bnd_residual(x_bnd, y_bnd)
                lb = torch.mean(res_bnd**2)
                tot_bnd_loss += lb.item() * bnd_coll_size
                n_bnd += bnd_coll_size
        avg_bnd = tot_bnd_loss / n_bnd if n_bnd else 0.0

        return avg_data, avg_phys, avg_bnd

    def adam_step(self, epochs: int, val_every: int, verbose: bool = True):

        best_total = float("inf")
        loss_t, loss_d, loss_p, loss_b = [], [], [], []
        val_loss_t, val_loss_d, val_loss_p, val_loss_b = [], [], [], []
        wt_phys, wt_bnd = [], []
        inv_param = []

        pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")

        for epoch in pbar:

            loss_stats = self.train_epoch()
            loss_t.append(loss_stats["total_loss"])
            loss_d.append(loss_stats["data_loss"])
            loss_p.append(loss_stats["phys_loss"])
            loss_b.append(loss_stats["bnd_loss"])

            self.grad_data_hist.append(loss_stats["avg_grad_d"])
            self.grad_phys_hist.append(loss_stats["avg_grad_p"])
            self.grad_bnd_hist.append(loss_stats["avg_grad_b"])

            postfix = {
                "L_T": f"{loss_stats['total_loss']:.2e}",
                "L_D": f"{loss_stats['data_loss']:.2e}",
                "L_P": f"{loss_stats['phys_loss']:.2e}",
                "L_B": f"{loss_stats['bnd_loss']:.2e}",
            }
            pbar.set_description(f"Epoch {epoch}/{epochs}")
            pbar.set_postfix(postfix)

            if epoch == 1 and verbose:
                tqdm.write(
                    f"\033[32m\nEpoch: {epoch} | Training  | "
                    f"Total Loss:    {loss_stats['total_loss']:.2e},"
                    f"\tData Loss:     {loss_stats['data_loss']:.2e},"
                    f"\tPhysics Loss:  {loss_stats['phys_loss']:.2e},"
                    f"\tBoundary Loss: {loss_stats['bnd_loss']:.2e}"
                    f"\033[0m]"
                )

            if (
                self.adapt_wts
                and (epoch % self.wt_update_every == 0)
                and self.phys_weight != 0
                and self.bnd_weight != 0
            ):
                grad_d = loss_stats["avg_grad_d"]
                grad_p = loss_stats["avg_grad_p"]
                grad_b = loss_stats["avg_grad_b"]

                if self.phys_weight and grad_p > 0.0:
                    target_wp = grad_d / (grad_p + 1e-12)
                    self.phys_weight = (
                        1.0 - self.wt_ema
                    ) * self.phys_weight + self.wt_ema * target_wp

                if self.bnd_weight and grad_b > 0.0:
                    target_wb = grad_d / (grad_b + 1e-12)
                    self.bnd_weight = (
                        1.0 - self.wt_ema
                    ) * self.bnd_weight + self.wt_ema * target_wb

            wt_phys.append(self.phys_weight)
            wt_bnd.append(self.bnd_weight)

            if epoch % val_every == 0:

                val_data, val_phys, val_bnd = self.validate_test(loader=self.val_loader)
                val_total = (
                    val_data + self.phys_weight * val_phys + self.bnd_weight * val_bnd
                )
                val_loss_t.append(val_total)
                val_loss_d.append(val_data)
                val_loss_p.append(val_phys)
                val_loss_b.append(val_bnd)
                if verbose:
                    tqdm.write(
                        f"\033[34m\nEpoch: {epoch} | Validation |"
                        f"Total Loss:    {val_total:.2e},"
                        f"\tData Loss:     {val_data:.2e},"
                        f"\tPhysics Loss:  {val_phys:.2e},"
                        f"\tBoundary Loss: {val_bnd:.2e}"
                        f"\033[0m"
                    )
                if val_total < best_total:
                    best_total = val_total
                    torch.save(
                        {"model_state_dict": self.model.state_dict()}, self.ckpt_path
                    )

                if self.theta is not None:
                    inv_param.append(self.theta.detach().cpu().numpy().copy())

                if self.scheduler is not None and isinstance(
                    self.scheduler, ReduceLROnPlateau
                ):
                    self.scheduler.step(val_total)

            if self.scheduler is not None and not isinstance(
                self.scheduler, ReduceLROnPlateau
            ):
                self.scheduler.step()
        pbar.close()

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        test_data, test_phys, test_bnd = self.validate_test(loader=self.test_loader)
        test_total = (
            test_data + self.phys_weight * test_phys + self.bnd_weight * test_bnd
        )
        if verbose:
            print(
                f"\033[33m\nTest | Total Loss: {test_total:.2e}, Data Loss: {test_data:.2e}, Physics Loss: {test_phys:.2e}, Boundary Loss: {test_bnd:.2e}\033[0m",
                flush=True,
            )

        return {
            "loss_t": loss_t,
            "loss_d": loss_d,
            "loss_p": loss_p,
            "loss_b": loss_b,
            "val_loss_t": val_loss_t,
            "val_loss_d": val_loss_d,
            "val_loss_p": val_loss_p,
            "val_loss_b": val_loss_b,
            "wt_phys": wt_phys,
            "wt_bnd": wt_bnd,
            "grad_data_hist": self.grad_data_hist,
            "grad_phys_hist": self.grad_phys_hist,
            "grad_bnd_hist": self.grad_bnd_hist,
            "inv_param": inv_param,
        }

    def lbfgs_step(self, mode: str, N_LBFGS: int = 50, verbose: bool = True):
        """
            Second-phase full-batch LBFGS optimization.

        Inputs
        ------
        mode : str
            Optimization mode:
            - "data"     : data loss
            - "physics"  : physics loss
            - "boundary" : boundary loss
            - "overall physics" : physics + boundary losses
            - "combined" : data + physics + boundary losses

        N_LBFGS : int, optional, default=50
            Number of LBFGS outer steps.

        verbose : bool, optional, default=True
            Whether to print optimization progress.

        Returns
        -------
        history_lbfgs : dict
            Dictionary containing LBFGS loss histories.
        """
        mode = mode.lower()
        valid_modes = ("data", "physics", "boundary", "combined", "overall physics")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of: {valid_modes}")

        ckpt = torch.load(self.ckpt_path, map_location=self.device)

        def gather_full_data(loader):
            X, Y = [], []
            for x_batch, y_batch in loader:
                X.append(x_batch)
                Y.append(y_batch)
            X = torch.cat(X, dim=0).to(self.device)
            Y = torch.cat(Y, dim=0).to(self.device)
            return X, Y

        def gather_full_colloc(batch_list):
            if len(batch_list) == 0:
                return None
            return torch.cat(batch_list, dim=0).to(self.device)

        self.X_all, self.Y_all = gather_full_data(self.train_loader)
        self.X_phys_all = gather_full_colloc(self.all_phys_colloc_batches)
        self.X_bnd_all = gather_full_colloc(self.all_bnd_colloc_batches)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.train()

        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=1,
            tolerance_grad=1e-9,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        history_lbfgs = {"loss_t": [], "loss_d": [], "loss_p": [], "loss_b": []}
        best_total = float("inf")

        def closure():
            self.lbfgs.zero_grad()
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            if mode in ("data", "combined"):
                y_pred = self.model(self.X_all)
                ld = self.loss_fn(y_pred, self.Y_all)
                loss = loss + ld

            if (
                mode in ("physics", "combined", "overall physics")
                and self.X_phys_all is not None
                and self.phys_residual is not None
                and self.phys_weight != 0
            ):
                x_phys = self.X_phys_all.detach().clone().requires_grad_(True)
                y_phys = self.model(x_phys)
                res_phys = self.phys_residual(x_phys, y_phys)
                lp = torch.mean(res_phys**2)
                loss = loss + self.phys_weight * lp

            if (
                mode in ("boundary", "combined", "overall physics")
                and self.X_bnd_all is not None
                and self.bnd_residual is not None
                and self.bnd_weight != 0
            ):
                x_bnd = self.X_bnd_all.detach().clone().requires_grad_(True)
                y_bnd = self.model(x_bnd)
                res_bnd = self.bnd_residual(x_bnd, y_bnd)
                lb = torch.mean(res_bnd**2)
                loss = loss + self.bnd_weight * lb

            loss.backward()
            return loss

        def evaluate_losses():
            self.model.train()

            ld = 0.0
            if mode in ("data", "combined"):
                ld = self.loss_fn(self.model(self.X_all), self.Y_all).item()

            lp = 0.0
            if (
                mode in ("physics", "combined", "overall physics")
                and self.X_phys_all is not None
                and self.phys_residual is not None
                and self.phys_weight != 0
            ):
                x_phys = self.X_phys_all.detach().clone().requires_grad_(True)
                y_phys = self.model(x_phys)
                res_phys = self.phys_residual(x_phys, y_phys)
                lp = torch.mean(res_phys**2).item()

            lb = 0.0
            if (
                mode in ("boundary", "combined", "overall physics")
                and self.X_bnd_all is not None
                and self.bnd_residual is not None
                and self.bnd_weight != 0
            ):
                x_bnd = self.X_bnd_all.detach().clone().requires_grad_(True)
                y_bnd = self.model(x_bnd)
                res_bnd = self.bnd_residual(x_bnd, y_bnd)
                lb = torch.mean(res_bnd**2).item()

            total = 0.0
            if mode in ("data", "combined"):
                total += ld
            if mode in ("physics", "combined", "overall physics"):
                total += self.phys_weight * lp
            if mode in ("boundary", "combined", "overall physics"):
                total += self.bnd_weight * lb

            return total, ld, lp, lb

        pbar = tqdm(range(1, N_LBFGS + 1), desc=f"LBFGS [{mode}]", unit="step")

        for step in pbar:
            self.lbfgs.step(closure)

            total, ld, lp, lb = evaluate_losses()

            history_lbfgs["loss_t"].append(total)
            history_lbfgs["loss_d"].append(ld)
            history_lbfgs["loss_p"].append(lp)
            history_lbfgs["loss_b"].append(lb)

            pbar.set_postfix(
                {
                    "L_T": f"{total:.2e}",
                    "L_D": f"{ld:.2e}",
                    "L_P": f"{lp:.2e}",
                    "L_B": f"{lb:.2e}",
                }
            )

            if verbose:
                tqdm.write(
                    f"LBFGS Step: {step} | "
                    f"Total Loss: {total:.2e}, "
                    f"Data Loss: {ld:.2e}, "
                    f"Physics Loss: {lp:.2e}, "
                    f"Boundary Loss: {lb:.2e}"
                )

            if total < best_total:
                best_total = total
                torch.save(
                    {"model_state_dict": self.model.state_dict()}, self.ckpt_path
                )

        pbar.close()

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        return history_lbfgs
