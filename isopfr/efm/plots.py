import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class Plotter:
    def __init__(
        self,
        history: dict,
        val_every: int = 1,
        fontsize: int = 20,
        fontfamily: str = "Arial",
        dpi: int = 300,
        figsize: tuple = (8, 6),
    ):
        """
        Plot training history stored in the `history` dictionary

        Inputs
        ------
        history : dict (dictionary containing training histories, e.g.)
                loss_t, loss_d, loss_p, loss_b, val_loss_t, ..., wt_phys, wt_bnd,
                grad_data_hist, grad_phys_hist, grad_bnd_hist, inv_param.

        val_every : int, optional, default=1 (validation frequency used during training)

        fontsize : int, optional, default=20

        fontfamily : str, optional, default="Arial"

        dpi : int, optional, default=300

        figsize : tuple, optional, default=(8, 6)
        """
        self.history = history
        self.val_every = val_every
        self.fontsize = fontsize
        self.fontfamily = fontfamily
        self.dpi = dpi
        self.figsize = figsize

    @staticmethod
    def _smooth(y, alpha=0.05):
        y = pd.Series(y)
        return y.ewm(alpha=alpha, adjust=False).mean().to_numpy()

    @staticmethod
    def _has_data(x):
        return x is not None and len(x) > 0

    @staticmethod
    def _has_nonzero_data(y):
        if y is None or len(y) == 0:
            return False
        y_arr = np.asarray(y, dtype=float)
        return not np.all(np.nan_to_num(y_arr, nan=0.0) == 0)

    def _train_epochs(self, n):
        return np.arange(1, n + 1)

    def _val_epochs(self, n):
        return np.arange(self.val_every, self.val_every * n + 1, self.val_every)

    def _format_axis(self, ax, xlabel="Epoch", ylabel=None, scale=None):
        if scale is not None:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: f"{x/scale:.0f}")
            )
        ax.set_xlabel(xlabel, fontsize=self.fontsize, fontfamily=self.fontfamily)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.fontsize, fontfamily=self.fontfamily)
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=self.fontsize,
            labelfontfamily=self.fontfamily,
        )
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=self.fontsize,
            labelfontfamily=self.fontfamily,
        )

    def plot_individual_loss(
        self,
        key: str,
        ylabel: str = "Loss",
        color: str = "black",
        alpha: float = 0.05,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """Plot one loss history"""
        y = self.history.get(key, [])
        if not self._has_nonzero_data(y):
            return

        x = (
            self._val_epochs(len(y))
            if key.startswith("val_")
            else self._train_epochs(len(y))
        )
        y_s = self._smooth(y, alpha=alpha)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.semilogy(x, y, linewidth=1.0, alpha=0.25, color=color)
        ax.semilogy(x, y_s, linewidth=2, color=color)

        self._format_axis(
            ax,
            xlabel="Epoch" if scale is None else f"Epoch (×{scale/1000}k)",
            ylabel=ylabel,
            scale=scale,
        )

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight")

    def plot_all_train_losses(
        self,
        alpha: float = 0.05,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot all training losses.
        """
        train_map = {
            "loss_t": ("Total Loss", "black"),
            "loss_d": ("Data Loss", "blue"),
            "loss_p": ("Physics Loss", "red"),
            "loss_b": ("Boundary Loss", "green"),
        }

        has_any = any(
            self._has_nonzero_data(self.history.get(key, [])) for key in train_map
        )
        if not has_any:
            return

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for key, (label, color) in train_map.items():
            y = self.history.get(key, [])
            if not self._has_nonzero_data(y):
                continue
            x = self._train_epochs(len(y))
            y_s = self._smooth(y, alpha=alpha)
            ax.semilogy(x, y, linewidth=1.0, alpha=0.25, color=color)
            ax.semilogy(x, y_s, linewidth=2, color=color, label=label)

        self._format_axis(
            ax,
            xlabel="Epoch" if scale is None else f"Epoch (×{scale/1000}k)",
            ylabel="Training Loss",
            scale=scale,
        )
        ax.legend(
            frameon=False, prop={"family": self.fontfamily, "size": self.fontsize - 2}
        )
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight")

    def plot_all_val_losses(
        self,
        alpha: float = 0.05,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot all validation losses.
        """
        val_map = {
            "val_loss_t": ("Total Loss", "black"),
            "val_loss_d": ("Data Loss", "blue"),
            "val_loss_p": ("Physics Loss", "red"),
            "val_loss_b": ("Boundary Loss", "green"),
        }

        has_any = any(
            self._has_nonzero_data(self.history.get(key, [])) for key in val_map
        )
        if not has_any:
            return

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for key, (label, color) in val_map.items():
            y = self.history.get(key, [])
            if not self._has_nonzero_data(y):
                continue
            x = self._val_epochs(len(y))
            y_s = self._smooth(y, alpha=alpha)
            ax.semilogy(x, y, linewidth=1.0, alpha=0.25, color=color)
            ax.semilogy(x, y_s, linewidth=2, color=color, label=label)

        self._format_axis(
            ax,
            xlabel="Epoch" if scale is None else f"Epoch (×{scale/1000}k)",
            ylabel="Validation Loss",
            scale=scale,
        )
        ax.legend(
            frameon=False, prop={"family": self.fontfamily, "size": self.fontsize - 2}
        )
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight")

    def plot_weights(
        self,
        alpha: float = 0.01,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot adaptive physics and boundary weights on dual y-axes.
        """
        wt_phys = self.history.get("wt_phys", [])
        wt_bnd = self.history.get("wt_bnd", [])

        if not self._has_nonzero_data(wt_phys) and not self._has_nonzero_data(wt_bnd):
            return

        fig, ax1 = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax2 = ax1.twinx()

        if self._has_nonzero_data(wt_phys):
            x1 = self._train_epochs(len(wt_phys))
            smooth_phys = self._smooth(wt_phys, alpha=alpha)
            ax1.plot(x1, wt_phys, alpha=0.5, linewidth=1.5, color="firebrick")
            ax1.plot(x1, smooth_phys, linewidth=2.0, color="firebrick")
            ax1.set_ylabel(
                r"Adaptive Physics Weight ($\lambda_P$)",
                fontsize=self.fontsize,
                fontfamily=self.fontfamily,
            )
            ax1.spines["left"].set_color("firebrick")
            ax1.yaxis.label.set_color("firebrick")
            ax1.tick_params(
                axis="y",
                colors="firebrick",
                labelsize=self.fontsize,
                labelfontfamily=self.fontfamily,
            )

        if self._has_nonzero_data(wt_bnd):
            x2 = self._train_epochs(len(wt_bnd))
            smooth_bnd = self._smooth(wt_bnd, alpha=alpha)
            ax2.plot(x2, wt_bnd, alpha=0.5, linewidth=1.5, color="teal")
            ax2.plot(x2, smooth_bnd, linewidth=2.0, color="teal")
            ax2.set_ylabel(
                r"Adaptive Boundary Weight ($\lambda_B$)",
                fontsize=self.fontsize,
                fontfamily=self.fontfamily,
            )
            ax2.spines["right"].set_color("teal")
            ax2.yaxis.label.set_color("teal")
            ax2.tick_params(
                axis="y",
                colors="teal",
                labelsize=self.fontsize,
                labelfontfamily=self.fontfamily,
            )

        self._format_axis(
            ax1,
            xlabel="Epoch" if scale is None else f"Epoch (×{scale/1000}k)",
            ylabel=None,
            scale=scale,
        )
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight")

    def plot_gradient_history(
        self,
        key: str,
        ylabel: str,
        color: str = "black",
        alpha: float = 0.05,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot gradient-norm history.
        """
        y = self.history.get(key, [])
        if not self._has_nonzero_data(y):
            return

        x = self._train_epochs(len(y))
        y_s = self._smooth(y, alpha=alpha)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.semilogy(x, y, linewidth=1.0, alpha=0.2, color=color)
        ax.semilogy(x, y_s, linewidth=1.8, color=color)

        self._format_axis(
            ax,
            xlabel="Epoch" if scale is None else f"Epoch (×{scale/1000}k)",
            ylabel=ylabel,
            scale=scale,
        )
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight")

    def plot_inverse_params(
        self,
        alpha: float = 0.10,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot inverse-parameter history. One figure is generated per parameter.
        """
        inv_param = self.history.get("inv_param", [])
        if not self._has_data(inv_param):
            return

        inv_param = np.asarray(inv_param)
        if inv_param.ndim == 1:
            inv_param = inv_param.reshape(-1, 1)

        if np.all(np.nan_to_num(inv_param.astype(float), nan=0.0) == 0):
            return

        x = self._val_epochs(inv_param.shape[0])

        for j in range(inv_param.shape[1]):
            y = inv_param[:, j]
            if not self._has_nonzero_data(y):
                continue
            y_s = self._smooth(y, alpha=alpha)

            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.plot(x, y, linewidth=1.0, alpha=0.2, color="purple")
            ax.plot(x, y_s, linewidth=1.8, color="purple")

            self._format_axis(
                ax,
                xlabel="Epoch" if scale is None else f"Epoch (×{scale/1000}k)",
                ylabel=f"Inverse Parameter {j+1}",
                scale=scale,
            )
            plt.tight_layout()

            if savepath is not None:
                os.makedirs(savepath, exist_ok=True)
                plt.savefig(
                    os.path.join(savepath, f"inv_param_{j+1}.png"),
                    bbox_inches="tight",
                )

    def plot_everything(self, savepath: str | None = None, scale: float | None = None):
        """
        Generate all standard plots from the available history entries.
        """
        if savepath is not None:
            os.makedirs(savepath, exist_ok=True)

        # Individual training losses
        self.plot_individual_loss(
            "loss_t",
            ylabel="Training Total Loss",
            color="black",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "train_loss_total.png")
            ),
        )
        self.plot_individual_loss(
            "loss_d",
            ylabel="Training Data Loss",
            color="blue",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "train_loss_data.png")
            ),
        )
        self.plot_individual_loss(
            "loss_p",
            ylabel="Training Physics Loss",
            color="red",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "train_loss_physics.png")
            ),
        )
        self.plot_individual_loss(
            "loss_b",
            ylabel="Training Boundary Loss",
            color="green",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "train_loss_boundary.png")
            ),
        )

        # Individual validation losses
        self.plot_individual_loss(
            "val_loss_t",
            ylabel="Validation Total Loss",
            color="black",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "val_loss_total.png")
            ),
        )
        self.plot_individual_loss(
            "val_loss_d",
            ylabel="Validation Data Loss",
            color="blue",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "val_loss_data.png")
            ),
        )
        self.plot_individual_loss(
            "val_loss_p",
            ylabel="Validation Physics Loss",
            color="red",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "val_loss_physics.png")
            ),
        )
        self.plot_individual_loss(
            "val_loss_b",
            ylabel="Validation Boundary Loss",
            color="green",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "val_loss_boundary.png")
            ),
        )

        self.plot_all_train_losses(
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "all_train_losses.png")
            ),
        )
        self.plot_all_val_losses(
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "all_val_losses.png")
            ),
        )

        # Adaptive weights
        self.plot_weights(
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "adaptive_weights.png")
            ),
        )

        # Gradients
        self.plot_gradient_history(
            "grad_data_hist",
            ylabel="Gradient Norm (Data)",
            color="blue",
            scale=scale,
            savepath=(
                None if savepath is None else os.path.join(savepath, "grad_data.png")
            ),
        )
        self.plot_gradient_history(
            "grad_phys_hist",
            ylabel="Gradient Norm (Physics)",
            color="red",
            scale=scale,
            savepath=(
                None if savepath is None else os.path.join(savepath, "grad_phys.png")
            ),
        )
        self.plot_gradient_history(
            "grad_bnd_hist",
            ylabel="Gradient Norm (Boundary)",
            color="green",
            scale=scale,
            savepath=(
                None
                if savepath is None
                else os.path.join(savepath, "grad_boundary.png")
            ),
        )

        # Inverse parameters
        self.plot_inverse_params(
            scale=scale,
            savepath=(
                None if savepath is None else os.path.join(savepath, "inverse_params")
            ),
        )
