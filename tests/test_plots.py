import matplotlib.pyplot as plt
import numpy as np
import pytest

from pinnse import Plotter


def full_history(n=20, n_val=4):
    rng = np.random.default_rng(0)

    def pos(k):
        return list(rng.uniform(0.1, 1.0, k))

    return {
        "loss_t": pos(n),
        "loss_d": pos(n),
        "loss_p": pos(n),
        "loss_b": pos(n),
        "val_loss_t": pos(n_val),
        "val_loss_d": pos(n_val),
        "val_loss_p": pos(n_val),
        "val_loss_b": pos(n_val),
        "wt_phys": pos(n),
        "wt_bnd": pos(n),
        "grad_data_hist": pos(n),
        "grad_phys_hist": pos(n),
        "grad_bnd_hist": pos(n),
        "inv_param": [rng.uniform(1, 2, 2) for _ in range(n_val)],
    }


def only_figure():
    assert len(plt.get_fignums()) == 1
    return plt.gcf()


def no_figures():
    return plt.get_fignums() == []


# ---------------------------------- helpers ----------------------------------
def test_smooth_is_exponential_moving_average():
    y = [1.0, 3.0, 2.0, 5.0]
    expected = [1.0]
    for v in y[1:]:
        expected.append(0.5 * v + 0.5 * expected[-1])
    np.testing.assert_allclose(Plotter._smooth(y, alpha=0.5), expected)


def test_data_presence_checks():
    assert not Plotter._has_data(None) and not Plotter._has_data([])
    assert Plotter._has_data([0])
    assert not Plotter._has_nonzero_data(None)
    assert not Plotter._has_nonzero_data([])
    assert not Plotter._has_nonzero_data([0.0, np.nan, 0.0])
    assert Plotter._has_nonzero_data([0.0, 1e-9])


def test_epoch_axes():
    p = Plotter({}, val_every=50)
    np.testing.assert_array_equal(p._train_epochs(3), [1, 2, 3])
    np.testing.assert_array_equal(p._val_epochs(3), [50, 100, 150])


# ------------------------------ individual plots -----------------------------
def test_individual_training_loss(tmp_path):
    h = full_history()
    path = tmp_path / "loss.png"
    Plotter(h, fontsize=10, dpi=50).plot_individual_loss(
        "loss_d", ylabel="Data", savepath=str(path)
    )

    ax = only_figure().axes[0]
    raw, smooth = ax.lines
    np.testing.assert_array_equal(raw.get_xdata(), np.arange(1, 21))
    np.testing.assert_allclose(raw.get_ydata(), h["loss_d"])
    np.testing.assert_allclose(smooth.get_ydata(), Plotter._smooth(h["loss_d"], 0.05))
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "Epoch" and ax.get_ylabel() == "Data"
    assert path.exists()


def test_individual_validation_loss_uses_validation_epochs():
    Plotter(full_history(), val_every=25, dpi=50).plot_individual_loss("val_loss_t")
    np.testing.assert_array_equal(
        only_figure().axes[0].lines[0].get_xdata(), [25, 50, 75, 100]
    )


def test_epoch_axis_scaling():
    Plotter(full_history(), dpi=50).plot_individual_loss("loss_t", scale=1000)
    ax = only_figure().axes[0]
    assert ax.get_xlabel() == "Epoch (×1.0k)"
    assert ax.xaxis.get_major_formatter()(5000, 0) == "5"


@pytest.mark.parametrize("value", [None, [], [0.0, 0.0]])
def test_individual_loss_skips_missing_or_zero_history(value):
    h = {} if value is None else {"loss_p": value}
    Plotter(h).plot_individual_loss("loss_p")
    assert no_figures()


def test_gradient_history(tmp_path):
    h = full_history()
    path = tmp_path / "grad.png"
    Plotter(h, dpi=50).plot_gradient_history(
        "grad_phys_hist", ylabel="G", color="red", savepath=str(path)
    )
    ax = only_figure().axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), h["grad_phys_hist"])
    assert ax.get_ylabel() == "G" and path.exists()

    plt.close("all")
    Plotter({"grad_phys_hist": [0.0] * 5}).plot_gradient_history(
        "grad_phys_hist", ylabel="G"
    )
    assert no_figures()


# ------------------------------- combined plots ------------------------------
def test_all_train_losses_legend_lists_nonzero_terms(tmp_path):
    h = full_history()
    h["loss_b"] = [0.0] * 20
    path = tmp_path / "train.png"
    Plotter(h, dpi=50).plot_all_train_losses(savepath=str(path))

    ax = only_figure().axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Total Loss",
        "Data Loss",
        "Physics Loss",
    ]
    assert ax.get_ylabel() == "Training Loss" and path.exists()


def test_all_val_losses_legend_and_axis(tmp_path):
    h = full_history()
    del h["val_loss_p"]
    path = tmp_path / "val.png"
    Plotter(h, val_every=10, dpi=50).plot_all_val_losses(savepath=str(path))

    ax = only_figure().axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Total Loss",
        "Data Loss",
        "Boundary Loss",
    ]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [10, 20, 30, 40])
    assert path.exists()


def test_combined_loss_plots_skip_empty_history():
    p = Plotter({"loss_t": [0.0], "val_loss_t": []})
    p.plot_all_train_losses()
    p.plot_all_val_losses()
    assert no_figures()


def test_weights_on_twin_axes(tmp_path):
    h = full_history()
    path = tmp_path / "w.png"
    Plotter(h, dpi=50).plot_weights(savepath=str(path))

    ax1, ax2 = only_figure().axes
    np.testing.assert_allclose(ax1.lines[0].get_ydata(), h["wt_phys"])
    np.testing.assert_allclose(ax2.lines[0].get_ydata(), h["wt_bnd"])
    assert "lambda_P" in ax1.get_ylabel() and "lambda_B" in ax2.get_ylabel()
    assert path.exists()


@pytest.mark.parametrize("present", ["wt_phys", "wt_bnd"])
def test_weights_with_single_weight(present):
    Plotter({present: [1.0, 2.0, 3.0]}, dpi=50).plot_weights()
    ax1, ax2 = only_figure().axes
    drawn = ax1 if present == "wt_phys" else ax2
    empty = ax2 if present == "wt_phys" else ax1
    assert len(drawn.lines) == 2 and len(empty.lines) == 0


def test_weights_skip_when_absent():
    Plotter({"wt_phys": [0.0], "wt_bnd": []}).plot_weights()
    assert no_figures()


# ------------------------------ inverse params -------------------------------
def test_inverse_params_one_figure_per_parameter(tmp_path):
    h = {
        "inv_param": [np.array([1.0, 0.0]), np.array([1.5, 0.0]), np.array([2.0, 0.0])]
    }
    out = tmp_path / "inv"
    Plotter(h, val_every=100, dpi=50).plot_inverse_params(savepath=str(out))

    # the second parameter is identically zero, so only one figure is drawn
    ax = only_figure().axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [100, 200, 300])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 1.5, 2.0])
    assert ax.get_ylabel() == "Inverse Parameter 1"
    assert sorted(p.name for p in out.iterdir()) == ["inv_param_1.png"]


def test_inverse_params_scalar_history_without_saving():
    Plotter({"inv_param": [1.0, 1.1, 1.2]}, dpi=50).plot_inverse_params()
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("value", [[], [np.zeros(2), np.zeros(2)]])
def test_inverse_params_skip_empty_or_zero(value, tmp_path):
    Plotter({"inv_param": value}).plot_inverse_params(savepath=str(tmp_path / "inv"))
    assert no_figures()
    assert not (tmp_path / "inv").exists()


# ------------------------------- plot_everything ------------------------------
def test_plot_everything_writes_every_figure(tmp_path):
    out = tmp_path / "figs"
    Plotter(full_history(), dpi=30).plot_everything(savepath=str(out), scale=100)

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [
            "train_loss_total.png",
            "train_loss_data.png",
            "train_loss_physics.png",
            "train_loss_boundary.png",
            "val_loss_total.png",
            "val_loss_data.png",
            "val_loss_physics.png",
            "val_loss_boundary.png",
            "all_train_losses.png",
            "all_val_losses.png",
            "adaptive_weights.png",
            "grad_data.png",
            "grad_phys.png",
            "grad_boundary.png",
            "inverse_params",
        ]
    )
    assert sorted(p.name for p in (out / "inverse_params").iterdir()) == [
        "inv_param_1.png",
        "inv_param_2.png",
    ]


def test_plot_everything_without_saving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Plotter(full_history(), dpi=30).plot_everything()
    # 8 individual losses, 2 combined, weights, 3 gradients, 2 inverse params
    assert len(plt.get_fignums()) == 16
    assert list(tmp_path.iterdir()) == []


def test_plot_everything_skips_missing_history():
    h = {"loss_t": [1.0, 0.5], "loss_d": [1.0, 0.5]}
    Plotter(h, dpi=30).plot_everything()
    # two individual loss plots and the combined training-loss plot
    assert len(plt.get_fignums()) == 3
