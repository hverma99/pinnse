import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Loss histories of `Training.adam_step`: key -> (legend label, color).
TRAIN_LOSSES = {
    "loss_t": ("Total Loss", "black"),
    "loss_d": ("Data Loss", "blue"),
    "loss_p": ("Physics Loss", "red"),
    "loss_b": ("Boundary Loss", "green"),
}
VAL_LOSSES = {"val_" + key: value for key, value in TRAIN_LOSSES.items()}

# Figures of `plot_everything`, in drawing order: (method, arguments, file name).
EVERYTHING = [
    ("plot_individual_loss", ("loss_t", "Training Total Loss", "black"), "train_loss_total.png"),
    ("plot_individual_loss", ("loss_d", "Training Data Loss", "blue"), "train_loss_data.png"),
    ("plot_individual_loss", ("loss_p", "Training Physics Loss", "red"), "train_loss_physics.png"),
    ("plot_individual_loss", ("loss_b", "Training Boundary Loss", "green"), "train_loss_boundary.png"),
    ("plot_individual_loss", ("val_loss_t", "Validation Total Loss", "black"), "val_loss_total.png"),
    ("plot_individual_loss", ("val_loss_d", "Validation Data Loss", "blue"), "val_loss_data.png"),
    ("plot_individual_loss", ("val_loss_p", "Validation Physics Loss", "red"), "val_loss_physics.png"),
    ("plot_individual_loss", ("val_loss_b", "Validation Boundary Loss", "green"), "val_loss_boundary.png"),
    ("plot_all_train_losses", (), "all_train_losses.png"),
    ("plot_all_val_losses", (), "all_val_losses.png"),
    ("plot_weights", (), "adaptive_weights.png"),
    ("plot_gradient_history", ("grad_data_hist", "Gradient Norm (Data)", "blue"), "grad_data.png"),
    ("plot_gradient_history", ("grad_phys_hist", "Gradient Norm (Physics)", "red"), "grad_phys.png"),
    ("plot_gradient_history", ("grad_bnd_hist", "Gradient Norm (Boundary)", "green"), "grad_boundary.png"),
    ("plot_inverse_params", (), "inverse_params"),
]  # fmt: skip


class Plotter:
    def __init__(
        self,
        history: dict,
        val_every: int | None = None,
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

        val_every : int, optional (validation frequency used during training)
                Only needed for histories without "val_epochs" (recorded by
                pinnse >= 1.0), which give the validation epochs directly;
                defaults to 1.

        fontsize : int, optional, default=20

        fontfamily : str, optional, default="Arial"

        dpi : int, optional, default=300

        figsize : tuple, optional, default=(8, 6)

        Notes
        -----
        - When `savepath` is given, each figure is written to disk and closed;
          otherwise figures are left open for interactive display.
        """
        self.history = history
        self.val_every = val_every
        self.fontsize = fontsize
        self.fontfamily = fontfamily
        self.dpi = dpi
        self.figsize = figsize

    # --------------------------------- helpers ---------------------------------
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
        recorded = self.history.get("val_epochs")
        if recorded is not None and len(recorded) >= n:
            return np.asarray(recorded[:n])
        step = self.val_every or 1
        return np.arange(step, step * n + 1, step)

    def _epochs_for(self, key, n):
        return self._val_epochs(n) if key.startswith("val_") else self._train_epochs(n)

    @staticmethod
    def _xlabel(scale):
        return "Epoch" if scale is None else f"Epoch (×{scale / 1000}k)"

    def _new_axes(self):
        return plt.subplots(figsize=self.figsize, dpi=self.dpi)

    def _draw(self, ax, x, y, color, alpha, raw_alpha, width, label=None, log=True):
        """Draw a raw series (faint) and its exponential moving average."""
        plot = ax.semilogy if log else ax.plot
        plot(x, y, linewidth=1.0, alpha=raw_alpha, color=color)
        plot(x, self._smooth(y, alpha=alpha), linewidth=width, color=color, label=label)

    def _legend(self, ax):
        ax.legend(
            frameon=False, prop={"family": self.fontfamily, "size": self.fontsize - 2}
        )

    @staticmethod
    def _finish(fig, savepath):
        """Lay out the figure and, when `savepath` is given, save and close it."""
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight")
            plt.close(fig)

    def _format_axis(self, ax, xlabel="Epoch", ylabel=None, scale=None):
        if scale is not None:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: f"{x / scale:.0f}")
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

    def _plot_single(
        self, key, ylabel, color, alpha, scale, savepath, raw_alpha, width
    ):
        y = self.history.get(key, [])
        if not self._has_nonzero_data(y):
            return
        fig, ax = self._new_axes()
        self._draw(ax, self._epochs_for(key, len(y)), y, color, alpha, raw_alpha, width)
        self._format_axis(ax, xlabel=self._xlabel(scale), ylabel=ylabel, scale=scale)
        self._finish(fig, savepath)

    def _plot_group(self, series, ylabel, alpha, scale, savepath):
        present = {
            key: style
            for key, style in series.items()
            if self._has_nonzero_data(self.history.get(key, []))
        }
        if not present:
            return
        fig, ax = self._new_axes()
        for key, (label, color) in present.items():
            y = self.history[key]
            x = self._epochs_for(key, len(y))
            self._draw(ax, x, y, color, alpha, 0.25, 2, label=label)
        self._format_axis(ax, xlabel=self._xlabel(scale), ylabel=ylabel, scale=scale)
        self._legend(ax)
        self._finish(fig, savepath)

    # ------------------------------ public plots -------------------------------
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
        self._plot_single(key, ylabel, color, alpha, scale, savepath, 0.25, 2)

    def plot_all_train_losses(
        self,
        alpha: float = 0.05,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot all training losses.
        """
        self._plot_group(TRAIN_LOSSES, "Training Loss", alpha, scale, savepath)

    def plot_all_val_losses(
        self,
        alpha: float = 0.05,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot all validation losses.
        """
        self._plot_group(VAL_LOSSES, "Validation Loss", alpha, scale, savepath)

    def plot_weights(
        self,
        alpha: float = 0.01,
        scale: float | None = None,
        savepath: str | None = None,
    ):
        """
        Plot adaptive physics and boundary weights on dual y-axes.
        """
        weights = [
            ("wt_phys", r"Adaptive Physics Weight ($\lambda_P$)", "firebrick", "left"),
            ("wt_bnd", r"Adaptive Boundary Weight ($\lambda_B$)", "teal", "right"),
        ]
        if not any(
            self._has_nonzero_data(self.history.get(k, [])) for k, *_ in weights
        ):
            return

        fig, ax1 = self._new_axes()
        ax2 = ax1.twinx()

        for ax, (key, ylabel, color, side) in zip((ax1, ax2), weights):
            y = self.history.get(key, [])
            if not self._has_nonzero_data(y):
                continue
            x = self._train_epochs(len(y))
            ax.plot(x, y, alpha=0.5, linewidth=1.5, color=color)
            ax.plot(x, self._smooth(y, alpha=alpha), linewidth=2.0, color=color)
            ax.set_ylabel(ylabel, fontsize=self.fontsize, fontfamily=self.fontfamily)
            ax.spines[side].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(
                axis="y",
                colors=color,
                labelsize=self.fontsize,
                labelfontfamily=self.fontfamily,
            )

        self._format_axis(ax1, xlabel=self._xlabel(scale), ylabel=None, scale=scale)
        self._finish(fig, savepath)

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
        self._plot_single(key, ylabel, color, alpha, scale, savepath, 0.2, 1.8)

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

            fig, ax = self._new_axes()
            self._draw(ax, x, y, "purple", alpha, 0.2, 1.8, log=False)
            self._format_axis(
                ax,
                xlabel=self._xlabel(scale),
                ylabel=f"Inverse Parameter {j + 1}",
                scale=scale,
            )

            path = None
            if savepath is not None:
                os.makedirs(savepath, exist_ok=True)
                path = os.path.join(savepath, f"inv_param_{j + 1}.png")
            self._finish(fig, path)

    def plot_everything(self, savepath: str | None = None, scale: float | None = None):
        """
        Generate all standard plots from the available history entries.
        """
        if savepath is not None:
            os.makedirs(savepath, exist_ok=True)

        for method, args, filename in EVERYTHING:
            path = None if savepath is None else os.path.join(savepath, filename)
            getattr(self, method)(*args, scale=scale, savepath=path)
