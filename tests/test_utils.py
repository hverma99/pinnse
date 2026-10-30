import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import pytest

from pinnse import ANN, Analyze, Denormalization, Noise, Normalization, Save

SPECIES = ["A", "B"]


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 4.0, 9.0],
            "b": [-3.0, 0.0, 3.0, 6.0],
            "c": [5.0, 5.0, 6.0, 7.0],
        }
    )


@pytest.fixture
def pfr_data():
    """Isothermal PFR dataset: inlet flows, P, T, V in; outlet flows, extents, conversions out."""
    rng = np.random.default_rng(0)
    n = 30
    I_S = pd.DataFrame(
        {
            "F_in_A": rng.uniform(50, 100, n),
            "F_in_B": rng.uniform(0, 10, n),
            "P": rng.uniform(1, 2, n),
            "T": rng.uniform(600, 700, n),
            "V": rng.uniform(0, 250, n),
        }
    )
    D_S = pd.DataFrame(
        {
            "F_ot_A": I_S["F_in_A"] * 0.5,
            "F_ot_B": I_S["F_in_B"] + 30,
            "E_R1": rng.uniform(0, 5, n),
            "E_R2": rng.uniform(10, 20, n),
            "X_R1": rng.uniform(0, 1, n),
        }
    )
    return I_S, D_S


# ------------------------------- Normalization -------------------------------
def test_min_max_scales_to_unit_interval_and_round_trips(frame):
    norm, metrics = Normalization.min_max(frame)
    assert (norm.min() == 0).all() and (norm.max() == 1).all()
    assert metrics["a"] == {"min": 1.0, "max": 9.0}
    pd.testing.assert_frame_equal(Denormalization.min_max(norm, metrics), frame)


def test_min_max_selected_columns_only(frame):
    norm, metrics = Normalization.min_max(frame, normalize_cols=["b"])
    assert set(metrics) == {"b"}
    pd.testing.assert_series_equal(norm["a"], frame["a"])
    np.testing.assert_allclose(norm["b"], [0, 1 / 3, 2 / 3, 1])
    # denormalization ignores columns without metrics
    pd.testing.assert_frame_equal(Denormalization.min_max(norm, metrics), frame)


def test_min_max_defined_metrics_reuses_training_ranges(frame):
    _, metrics = Normalization.min_max(frame)
    new = pd.DataFrame({"a": [5.0], "b": [-6.0], "c": [5.0]})
    out = Normalization.min_max_defined_metrics(new, metrics)
    np.testing.assert_allclose(out.iloc[0], [0.5, -1 / 3, 0.0])


def test_scale_centered(frame):
    out = Normalization.scale_centered(frame, "b", -3.0, 6.0)
    np.testing.assert_allclose(out, [-1, -1 / 3, 1 / 3, 1], rtol=1e-6)
    assert out.dtype == np.float32
    constant = Normalization.scale_centered(frame, "b", 2.0, 2.0)
    np.testing.assert_array_equal(constant, np.zeros(4))
    assert constant.dtype == np.float32


def test_apply_global_shares_one_range(frame):
    norm, metrics = frame.copy(), {}
    Normalization.apply_global(frame, norm, metrics, ["a", "b"])
    assert metrics["a"] == metrics["b"] == {"min": -3.0, "max": 9.0}
    assert norm["a"].max() == 1.0 and norm["b"].min() == -1.0
    pd.testing.assert_series_equal(norm["c"], frame["c"])


def test_apply_global_with_explicit_range_and_empty_columns(frame):
    norm, metrics = frame.copy(), {}
    Normalization.apply_global(frame, norm, metrics, ["c"], gmin=5.0, gmax=7.0)
    np.testing.assert_allclose(norm["c"], [-1, -1, 0, 1])

    untouched = frame.copy()
    assert Normalization.apply_global(frame, untouched, metrics, []) is None
    pd.testing.assert_frame_equal(untouched, frame)


def test_apply_local_uses_each_columns_range(frame):
    norm, metrics = frame.copy(), {}
    Normalization.apply_local(frame, norm, metrics, ["a", "c"])
    assert metrics == {"a": {"min": 1.0, "max": 9.0}, "c": {"min": 5.0, "max": 7.0}}
    assert norm[["a", "c"]].min().tolist() == [-1, -1]
    assert norm[["a", "c"]].max().tolist() == [1, 1]


def test_min_max_pfr_efm_shares_range_between_inlet_and_outlet_flows(pfr_data):
    I_S, D_S = pfr_data
    D_S = D_S[["F_ot_A", "F_ot_B"]]
    nI, nD, mI, mD = Normalization.min_max_pfr(I_S, D_S, "efm", SPECIES)

    flows = np.concatenate(
        [I_S[["F_in_A", "F_in_B"]].values.ravel(), D_S.values.ravel()]
    )
    shared = {"min": pytest.approx(flows.min()), "max": pytest.approx(flows.max())}
    for col in ("F_in_A", "F_in_B"):
        assert mI[col] == shared
    for col in ("F_ot_A", "F_ot_B"):
        assert mD[col] == shared
    for col in ("P", "T", "V"):
        assert mI[col] == {"min": I_S[col].min(), "max": I_S[col].max()}
    assert nI.min().min() >= -1 and nI.max().max() <= 1
    assert nD.min().min() >= -1 and nD.max().max() <= 1

    # inverse transform recovers the dimensional outputs
    X = torch.tensor(nD.to_numpy(), dtype=torch.float64)
    X_dim, rngs = Denormalization.min_max_pfr(X, mD, list(D_S.columns))
    np.testing.assert_allclose(X_dim.numpy(), D_S.to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(rngs.numpy(), [flows.max() - flows.min()] * 2, rtol=1e-6)


def test_min_max_pfr_efm_without_flow_columns(pfr_data):
    I_S, D_S = pfr_data
    nI, _, mI, mD = Normalization.min_max_pfr(
        I_S[["P", "T", "V"]], D_S[["E_R1"]], "EFM", SPECIES
    )
    assert set(mI) == {"P", "T", "V"} and mD == {}


def test_min_max_pfr_erm(pfr_data):
    I_S, D_S = pfr_data
    nI, nD, mI, mD = Normalization.min_max_pfr(I_S, D_S, "ERM", SPECIES, N_reaction=3)

    lo = I_S[["F_in_A", "F_in_B"]].values.min()
    hi = I_S[["F_in_A", "F_in_B"]].values.max()
    assert (
        mI["F_in_A"]
        == mI["F_in_B"]
        == {"min": pytest.approx(lo), "max": pytest.approx(hi)}
    )
    # only extents present in the data (E_R1, E_R2 of three reactions), each local
    assert set(mD) == {"E_R1", "E_R2"}
    assert nD[["E_R1", "E_R2"]].min().tolist() == [-1, -1]
    pd.testing.assert_series_equal(nD["X_R1"], D_S["X_R1"])


def test_min_max_pfr_cm(pfr_data):
    I_S, D_S = pfr_data
    nI, nD, mI, mD = Normalization.min_max_pfr(I_S, D_S, "CM", SPECIES, N_reaction=2)
    # conversion-model inputs are scaled independently
    assert mI["F_in_A"]["min"] == I_S["F_in_A"].min()
    assert mI["F_in_B"]["min"] == I_S["F_in_B"].min()
    assert set(mD) == {"X_R1"}
    assert nI.min().tolist() == [-1] * 5 and nI.max().tolist() == [1] * 5


def test_min_max_pfr_rejects_unknown_formulation(pfr_data):
    with pytest.raises(ValueError, match="formulation must be one of"):
        Normalization.min_max_pfr(*pfr_data, formulation="XYZ", species=SPECIES)


def test_scale_centered_defined_metrics(frame):
    metrics = {c: {"min": frame[c].min(), "max": frame[c].max()} for c in frame}
    out = Normalization.scale_centered_defined_metrics(frame, metrics)
    assert out.min().tolist() == [-1] * 3 and out.max().tolist() == [1] * 3


def test_max_abs_round_trip(frame):
    norm, metrics = Normalization.max_abs(frame)
    assert metrics["b"] == {"max": 6.0}
    assert norm.abs().max().tolist() == [1, 1, 1]
    pd.testing.assert_frame_equal(Denormalization.max_abs(norm, metrics), frame)

    norm, metrics = Normalization.max_abs(frame, normalize_cols=["a"])
    pd.testing.assert_frame_equal(Denormalization.max_abs(norm, metrics), frame)


def test_normalizations_map_constant_columns_to_zero():
    df = pd.DataFrame({"k": [2.0, 2.0, 2.0], "a": [1.0, 2.0, 3.0]})

    norm, metrics = Normalization.min_max(df)
    assert norm["k"].tolist() == [0.0, 0.0, 0.0]
    assert metrics["k"] == {"min": 2.0, "max": 2.0}
    np.testing.assert_allclose(norm["a"], [0.0, 0.5, 1.0])
    assert Normalization.min_max_defined_metrics(df, metrics)["k"].tolist() == [0.0] * 3

    norm, _ = Normalization.mean_norm(df)
    assert norm["k"].tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(norm["a"], [-0.5, 0.0, 0.5])

    norm, _ = Normalization.z_score(df)
    assert norm["k"].tolist() == [0.0, 0.0, 0.0]
    assert not norm.isna().any().any()


def test_mean_norm_round_trip(frame):
    norm, metrics = Normalization.mean_norm(frame)
    np.testing.assert_allclose(norm.mean(), 0, atol=1e-12)
    assert metrics["a"] == {"max": 9.0, "min": 1.0, "mean": 4.0}
    pd.testing.assert_frame_equal(Denormalization.mean_abs(norm, metrics), frame)

    norm, metrics = Normalization.mean_norm(frame, normalize_cols=["c"])
    pd.testing.assert_frame_equal(Denormalization.mean_abs(norm, metrics), frame)


def test_z_score_round_trip(frame):
    norm, metrics = Normalization.z_score(frame)
    np.testing.assert_allclose(norm.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(norm.std(), 1)
    pd.testing.assert_frame_equal(Denormalization.z_norm(norm, metrics), frame)

    norm, metrics = Normalization.z_score(frame, normalize_cols=["b"])
    pd.testing.assert_frame_equal(Denormalization.z_norm(norm, metrics), frame)


@pytest.mark.parametrize("fn", ["min_max", "max_abs", "mean_abs", "z_norm"])
def test_denormalization_without_metrics_returns_copy(frame, fn):
    out = getattr(Denormalization, fn)(frame, {})
    pd.testing.assert_frame_equal(out, frame)
    assert out is not frame


def test_min_max_col_supports_tensors_arrays_and_scalars():
    metrics = {"T": {"min": 300.0, "max": 400.0}}
    torch.testing.assert_close(
        Denormalization.min_max_col(torch.tensor([0.0, 0.5]), "T", metrics),
        torch.tensor([300.0, 350.0]),
    )
    np.testing.assert_allclose(
        Denormalization.min_max_col(np.array([1.0]), "T", metrics), [400.0]
    )
    assert Denormalization.min_max_col(0.25, "T", metrics) == 325.0
    assert Denormalization.min_max_col(0.25, "P", metrics) == 0.25
    assert Denormalization.min_max_col(0.25, "T", {}) == 0.25


# ---------------------------------- Analyze ----------------------------------
@pytest.mark.parametrize("wrapped", [True, False])
def test_load_ann_pinn_restores_checkpoint(tmp_path, wrapped):
    model = ANN([3, 5, 5, 2], activation=nn.ReLU)
    state = model.state_dict()
    path = tmp_path / "m.pth"
    torch.save({"model_state_dict": state} if wrapped else state, path)

    loaded = Analyze.load_ann_pinn(3, 2, 2, 5, nn.ReLU, str(path), torch.device("cpu"))

    assert not loaded.training
    x = torch.randn(4, 3)
    torch.testing.assert_close(loaded(x), model(x))


def test_pinn_eval_normalizes_inputs_and_denormalizes_outputs(frame):
    model = ANN([3, 2], activation=nn.Tanh)
    W = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    with torch.no_grad():
        model.net[0].weight.copy_(W)

    _, I_metrics = Normalization.min_max(frame)
    D_metrics = {"y1": {"min": 10.0, "max": 20.0}, "y2": {"min": 0.0, "max": 2.0}}
    out = Analyze.pinn_eval(
        model, frame, I_metrics, D_metrics, ["y1", "y2"], torch.device("cpu")
    )

    x = Normalization.min_max_defined_metrics(frame, I_metrics).to_numpy()
    np.testing.assert_allclose(out["y1"], 10 + 10 * x[:, 0], rtol=1e-6)
    np.testing.assert_allclose(out["y2"], 2 * (x[:, 1] + x[:, 2]), rtol=1e-6)
    assert list(out.columns) == ["y1", "y2"]


def test_pinn_eval_pfr_uses_centered_scaling(frame):
    model = ANN([3, 1], activation=nn.Tanh)
    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))

    I_metrics = {c: {"min": frame[c].min(), "max": frame[c].max()} for c in frame}
    D_metrics = {"F": {"min": 0.0, "max": 100.0}}
    out = Analyze.pinn_eval_pfr(
        model, frame, I_metrics, D_metrics, ["F"], torch.device("cpu")
    )

    x_a = 2 * (frame["a"] - 1) / 8 - 1  # input "a" scaled to [-1, 1]
    np.testing.assert_allclose(out["F"], 50 * (x_a + 1), rtol=1e-5)


def test_error_metrics_known_values():
    truth = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0]})
    pred = pd.DataFrame({"a": [1.0, 2.0, 5.0], "b": [1.0, -1.0, 0.0]})
    out = Analyze.error_metrics(
        truth, pred, {"A": ["a"], "B": ["b"], "All": ["a", "b"]}
    )

    assert out["Output"].tolist() == ["A", "B", "All"]
    a, b, both = out.to_dict("records")
    assert a["MAE"] == pytest.approx(2 / 3)
    assert a["RMSE"] == pytest.approx(np.sqrt(4 / 3))
    assert a["R2"] == pytest.approx(1 - 4 / 2)
    assert b["MAE"] == pytest.approx(2 / 3) and np.isnan(b["R2"])  # constant truth
    assert both["MAE"] == pytest.approx(4 / 6)


def test_error_metrics_perfect_prediction_and_non_finite_values():
    truth = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [np.nan] * 4})
    pred = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.inf], "b": [1.0] * 4})
    a, b = Analyze.error_metrics(truth, pred, {"a": ["a"], "b": ["b"]}).to_dict(
        "records"
    )

    # only the two finite pairs are scored
    assert (a["MAE"], a["RMSE"], a["R2"]) == (0.0, 0.0, 1.0)
    assert np.isnan(b["MAE"]) and np.isnan(b["RMSE"]) and np.isnan(b["R2"])


# ------------------------------------ Save -----------------------------------
def history():
    return {
        "loss_t": [1.0, 0.5, 0.25],
        "inv_param": [np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        "a_very_long_history_key_name_exceeding_limits": [1, 2],
        "skipped": None,
        "best_epoch": 7,
    }


def test_save_excel(tmp_path, capsys):
    folder = tmp_path / "logs"
    Save.excel(history(), foldername=str(folder), filename="h.xlsx")

    sheets = pd.read_excel(folder / "h.xlsx", sheet_name=None)
    # sheet names are truncated to Excel's 31-character limit
    assert list(sheets) == [
        "loss_t",
        "inv_param",
        "a_very_long_history_key_name_ex",
        "best_epoch",
    ]
    assert sheets["loss_t"]["loss_t"].tolist() == [1.0, 0.5, 0.25]
    assert list(sheets["inv_param"].columns) == ["inv_param_1", "inv_param_2"]
    assert sheets["inv_param"].to_numpy().tolist() == [[1.0, 2.0], [1.5, 2.5]]
    assert sheets["best_epoch"]["best_epoch"].tolist() == [7]
    assert "Saved training history XLSX" in capsys.readouterr().out


def test_save_csv(tmp_path, capsys):
    Save.csv(history(), foldername=str(tmp_path), filename="csv")
    folder = tmp_path / "csv"

    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [
            "loss_t.csv",
            "inv_param.csv",
            "a_very_long_history_key_name_exceeding_limits.csv",
            "best_epoch.csv",
        ]
    )
    assert pd.read_csv(folder / "inv_param.csv").to_numpy().tolist() == [
        [1.0, 2.0],
        [1.5, 2.5],
    ]
    assert pd.read_csv(folder / "best_epoch.csv")["best_epoch"].tolist() == [7]
    assert "Saved training history CSV" in capsys.readouterr().out


def test_save_defaults_write_to_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Save.excel({"loss_t": [1.0]})
    Save.csv({"loss_t": [1.0]})
    assert (tmp_path / "logs" / "training_history.xlsx").exists()
    assert (tmp_path / "logs" / "csv_files" / "loss_t.csv").exists()


# ----------------------------------- Noise -----------------------------------
@pytest.fixture
def measurements():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "F": rng.uniform(10, 20, 20000),
            "T": rng.uniform(300, 400, 20000),
            "Z": np.zeros(20000),
        }
    )


def test_gaussian_relative_noise(measurements):
    noisy, metrics = Noise.gaussian(measurements, 0.1, noise_cols=["F", "T"])

    for col in ("F", "T"):
        sigma = 0.1 * measurements[col].std(ddof=0)
        assert metrics[col] == {
            "mode": "relative",
            "level": 0.1,
            "sigma": pytest.approx(sigma),
        }
        err = noisy[col] - measurements[col]
        assert err.mean() == pytest.approx(0, abs=4 * sigma / np.sqrt(len(err)))
        assert err.std() == pytest.approx(sigma, rel=0.03)
    pd.testing.assert_series_equal(noisy["Z"], measurements["Z"])


def test_gaussian_proportional_noise_scales_with_magnitude(measurements):
    noisy, metrics = Noise.gaussian(measurements, 0.05, mode="proportional")

    rel = (noisy["F"] - measurements["F"]) / measurements["F"]
    assert rel.std() == pytest.approx(0.05, rel=0.03)
    assert metrics["F"]["sigma"] == pytest.approx(0.05 * measurements["F"].mean())
    # zero-valued measurements carry no proportional noise
    assert (noisy["Z"] == 0).all() and metrics["Z"]["sigma"] == 0.0


def test_gaussian_absolute_noise_with_per_column_levels(measurements):
    noisy, metrics = Noise.gaussian(
        measurements, {"F": 0.5, "Z": 2.0}, mode="absolute", noise_cols=["F", "Z"]
    )
    assert (noisy["F"] - measurements["F"]).std() == pytest.approx(0.5, rel=0.03)
    assert noisy["Z"].std() == pytest.approx(2.0, rel=0.03)
    assert metrics["Z"] == {"mode": "absolute", "level": 2.0, "sigma": 2.0}
    pd.testing.assert_series_equal(noisy["T"], measurements["T"])


def test_gaussian_clipping_and_reproducibility(measurements):
    a, _ = Noise.gaussian(measurements, 1.0, mode="absolute", clip_min=0.0)
    b, _ = Noise.gaussian(measurements, 1.0, mode="absolute", clip_min=0.0)
    c, _ = Noise.gaussian(measurements, 1.0, mode="absolute", random_state=7)

    assert (a["Z"] >= 0).all() and (a["Z"] == 0).any()
    assert (c["Z"] < 0).any()
    pd.testing.assert_frame_equal(a, b)
    assert not a["F"].equals(c["F"])


def test_gaussian_rejects_unknown_mode(measurements):
    with pytest.raises(ValueError, match="mode must be one of"):
        Noise.gaussian(measurements, 0.1, mode="uniform")
