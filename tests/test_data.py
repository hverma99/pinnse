import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import numpy as np
import pandas as pd
import pytest

from pinnse import DataModule, SequenceDataModule


def gather(loader, idx=0):
    return torch.cat([batch[idx] for batch in loader]).numpy()


@pytest.fixture
def module(tabular_data):
    I_S, D_S = tabular_data
    dm = DataModule(
        I_S_data=I_S,
        D_S_data=D_S,
        labeled_data_batch_size=16,
        physics_coll_data_size=64,
        physics_coll_batch_size=16,
        boundary_coll_data_size=32,
        boundry_coll_batch_size=8,
        test_frac=0.1,
        val_frac=0.1,
    )
    loaders = dm.labeled_data_loader()
    return dm, loaders


# ------------------------------ labeled loaders ------------------------------
def test_labeled_loaders_partition_the_dataset(module, tabular_data):
    I_S, D_S = tabular_data
    dm, (train, val, test) = module

    sizes = [len(dl.dataset) for dl in (train, val, test)]
    assert sizes == [81, 9, 10]

    # gather (X, Y) pairs in a single pass, since the training loader reshuffles
    pairs = [batch for dl in (train, val, test) for batch in dl]
    X = torch.cat([x for x, _ in pairs]).numpy()
    Y = torch.cat([y for _, y in pairs]).numpy()
    # every labeled row appears exactly once, still paired with its output
    order = np.lexsort(X.T)
    ref = I_S.to_numpy(np.float32)
    ref_order = np.lexsort(ref.T)
    np.testing.assert_array_equal(X[order], ref[ref_order])
    np.testing.assert_array_equal(Y[order], D_S.to_numpy(np.float32)[ref_order])


def test_labeled_loaders_shuffle_only_training(module):
    _, (train, val, test) = module
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    assert train.batch_size == 16
    assert gather(train).dtype == np.float32


def test_labeled_loaders_record_input_bounds(module, tabular_data):
    I_S, _ = tabular_data
    dm, _ = module
    np.testing.assert_allclose(dm.lower_bnd, I_S.min().to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(dm.upper_bnd, I_S.max().to_numpy(), rtol=1e-6)


def test_colloc_loaders_do_not_depend_on_labeled_loader(tabular_data):
    I_S, D_S = tabular_data
    dm = DataModule(
        I_S_data=I_S,
        D_S_data=D_S,
        labeled_data_batch_size=16,
        physics_coll_data_size=32,
        physics_coll_batch_size=16,
        boundary_coll_data_size=16,
        boundry_coll_batch_size=8,
    )
    # bounds are available from construction, so either loader may come first
    np.testing.assert_allclose(dm.lower_bnd, I_S.min().to_numpy(), rtol=1e-6)
    assert gather(dm.phys_colloc_loader()).shape == (32, I_S.shape[1])
    assert gather(dm.bnd_colloc_loader()).shape == (16, I_S.shape[1])


def test_labeled_split_is_reproducible(tabular_data):
    I_S, D_S = tabular_data

    def test_rows(seed):
        dm = DataModule(I_S, D_S, 8, test_frac=0.2, val_frac=0.2, random_state=seed)
        return gather(dm.labeled_data_loader()[2])

    np.testing.assert_array_equal(test_rows(1), test_rows(1))
    assert not np.array_equal(test_rows(1), test_rows(2))


# ------------------------- physics-collocation loader ------------------------
def test_phys_colloc_samples_respect_bounds_and_simplex(module):
    dm, _ = module
    loader = dm.phys_colloc_loader()
    X = gather(loader)

    assert X.shape == (64, 5)
    assert len(loader) == 4
    assert isinstance(loader.sampler, RandomSampler)

    box = [0, 1, 4]  # P, T, V
    assert np.all(X[:, box] >= dm.lower_bnd[box] - 1e-5)
    assert np.all(X[:, box] <= dm.upper_bnd[box] + 1e-5)
    # mole fractions are sampled on the simplex, not the bounding box
    np.testing.assert_allclose(X[:, 2] + X[:, 3], 1.0, rtol=1e-5)
    assert np.all(X[:, 2:4] >= 0)


def test_phys_colloc_box_columns_cover_the_range(module):
    dm, _ = module
    X = gather(dm.phys_colloc_loader(shuffle=False))
    # Latin hypercube: exactly one sample in each of the N equal strata per column
    for j in (0, 1, 4):
        unit = (X[:, j] - dm.lower_bnd[j]) / (dm.upper_bnd[j] - dm.lower_bnd[j])
        strata = np.clip(np.floor(unit * 64).astype(int), 0, 63)
        assert sorted(strata) == list(range(64))


def test_phys_colloc_is_reproducible_and_unshuffled_on_request(module):
    dm, _ = module
    a = gather(dm.phys_colloc_loader(shuffle=False))
    b = gather(dm.phys_colloc_loader(shuffle=False))
    np.testing.assert_array_equal(a, b)
    loader = dm.phys_colloc_loader(shuffle=False, drop_last=True)
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last


def test_phys_colloc_dirichlet_alpha_controls_concentration(module):
    dm, _ = module
    dm.physics_coll_data_size = 2000
    flat = gather(dm.phys_colloc_loader(alpha=1.0))[:, 2]
    peaked = gather(dm.phys_colloc_loader(alpha=50.0))[:, 2]
    assert peaked.std() < flat.std() / 3


def test_phys_colloc_with_only_simplex_inputs():
    rng = np.random.default_rng(1)
    z = rng.dirichlet([1, 1, 1], size=20)
    x = rng.dirichlet([1, 1], size=20)
    I_S = pd.DataFrame(np.hstack([z, x]), columns=["Z_1", "Z_2", "Z_3", "X_1", "X_2"])
    D_S = pd.DataFrame({"y": rng.random(20)})
    dm = DataModule(I_S, D_S, 4, 10, 5, test_frac=0.2, val_frac=0.2)
    dm.labeled_data_loader()

    X = gather(dm.phys_colloc_loader())
    assert X.shape == (10, 5)
    np.testing.assert_allclose(X[:, :3].sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(X[:, 3:].sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize(
    "sizes, missing",
    [
        ({}, "physics_coll_data_size"),
        ({"physics_coll_data_size": 0}, "physics_coll_data_size"),
        ({"physics_coll_data_size": 16}, "physics_coll_batch_size"),
        (
            {"physics_coll_data_size": 16, "physics_coll_batch_size": 0},
            "physics_coll_batch_size",
        ),
    ],
)
def test_phys_colloc_requires_sizes(tabular_data, sizes, missing):
    I_S, D_S = tabular_data
    dm = DataModule(I_S, D_S, 8, test_frac=0.1, val_frac=0.1, **sizes)
    dm.labeled_data_loader()
    with pytest.raises(ValueError, match=f"{missing} must be a positive integer"):
        dm.phys_colloc_loader()


@pytest.mark.parametrize(
    "sizes, missing",
    [
        ({}, "boundary_coll_data_size"),
        ({"boundary_coll_data_size": -5}, "boundary_coll_data_size"),
        ({"boundary_coll_data_size": 16}, "boundry_coll_batch_size"),
    ],
)
def test_bnd_colloc_requires_sizes(tabular_data, sizes, missing):
    I_S, D_S = tabular_data
    dm = DataModule(I_S, D_S, 8, test_frac=0.1, val_frac=0.1, **sizes)
    dm.labeled_data_loader()
    with pytest.raises(ValueError, match=f"{missing} must be a positive integer"):
        dm.bnd_colloc_loader()


# ------------------------ boundary-collocation loader ------------------------
def test_bnd_colloc_fixes_last_column(module):
    dm, _ = module
    loader = dm.bnd_colloc_loader(bnd_value=0.25)
    X = gather(loader)

    assert X.shape == (32, 5)
    assert len(loader) == 4
    np.testing.assert_array_equal(X[:, -1], 0.25)
    assert np.all(X[:, :-1] >= dm.lower_bnd[:-1] - 1e-5)
    assert np.all(X[:, :-1] <= dm.upper_bnd[:-1] + 1e-5)


def test_bnd_colloc_default_value_and_options(module):
    dm, _ = module
    loader = dm.bnd_colloc_loader(shuffle=False, drop_last=True)
    np.testing.assert_array_equal(gather(loader)[:, -1], -1.0)
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last


# ------------------------------ inspect / save -------------------------------
def test_inspect_loader_labeled(module, capsys):
    dm, (train, _, _) = module
    dm.inspect_loader("train", train)
    assert capsys.readouterr().out.strip() == (
        "train_loader -> X: torch.Size([16, 5]), Y: torch.Size([16, 2]), batches: 6"
    )


def test_inspect_loader_unlabeled(module, capsys):
    dm, _ = module
    dm.inspect_loader("phys", dm.phys_colloc_loader())
    assert capsys.readouterr().out.strip() == (
        "phys_loader -> X: torch.Size([16, 5]), batches: 4"
    )


def test_inspect_loader_plain_tensor_batches(module, capsys):
    dm, _ = module
    dm.inspect_loader("raw", DataLoader(torch.zeros(10, 5), batch_size=4))
    assert capsys.readouterr().out.strip() == (
        "raw_loader -> X: torch.Size([4, 5]), batches: 3"
    )


def test_save_labeled_loader(module, tmp_path, tabular_data, capsys):
    I_S, D_S = tabular_data
    dm, (_, _, test) = module
    path = tmp_path / "test.xlsx"
    dm.save_loaders(test, str(path))

    df = pd.read_excel(path)
    assert list(df.columns) == list(I_S.columns) + list(D_S.columns)
    np.testing.assert_allclose(df[I_S.columns].to_numpy(), gather(test), rtol=1e-6)
    np.testing.assert_allclose(df[D_S.columns].to_numpy(), gather(test, 1), rtol=1e-6)
    assert f"Saved 10 samples to {path}" in capsys.readouterr().out


def test_save_unlabeled_loaders(module, tmp_path, tabular_data):
    I_S, _ = tabular_data
    dm, _ = module

    path = tmp_path / "bnd.xlsx"
    dm.save_loaders(dm.bnd_colloc_loader(shuffle=False), str(path))
    df = pd.read_excel(path)
    assert list(df.columns) == list(I_S.columns) and len(df) == 32

    path = tmp_path / "raw.xlsx"
    dm.save_loaders(DataLoader(torch.ones(3, 5), batch_size=2), str(path))
    np.testing.assert_array_equal(pd.read_excel(path).to_numpy(), np.ones((3, 5)))


def test_save_empty_loader_raises(module, tmp_path):
    dm, _ = module
    empty = DataLoader(TensorDataset(torch.empty(0, 5)), batch_size=4)
    with pytest.raises(ValueError, match="empty"):
        dm.save_loaders(empty, str(tmp_path / "x.xlsx"))


# ---------------------------- SequenceDataModule -----------------------------
def make_sequences(n_traj=20, seq_len=10, seed=0):
    """Inputs: [initial condition (constant), coolant (piecewise), time (varies)]."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_traj, seq_len, 3))
    X[:, :, 0] = rng.uniform(0, 1, (n_traj, 1))
    X[:, :, 1] = np.repeat(rng.uniform(2, 3, (n_traj, 2)), seq_len // 2, axis=1)
    X[:, :, 2] = np.linspace(0, 1, seq_len)
    Y = np.cumsum(X[:, :, :2], axis=1)
    return X, Y


def test_sequence_module_rejects_bad_shapes():
    X, Y = make_sequences()
    with pytest.raises(ValueError, match="n_trajectories, seq_len, n_features"):
        SequenceDataModule(X[0], Y, 4)
    with pytest.raises(ValueError, match="must agree"):
        SequenceDataModule(X, Y[:, :-1], 4)


def test_sequence_module_detects_context_columns():
    X, Y = make_sequences()
    sdm = SequenceDataModule(X, Y, 4, test_frac=0.2, val_frac=0.25)
    assert (sdm.n_traj, sdm.seq_len, sdm.dim_in) == (20, 10, 3)
    assert sdm.context_cols == [0]
    assert sdm.driving_cols == [1, 2]
    np.testing.assert_allclose(sdm.lower_bnd, X.min(axis=(0, 1)))
    np.testing.assert_allclose(sdm.upper_bnd, X.max(axis=(0, 1)))


def test_sequence_module_accepts_explicit_context_columns():
    X, Y = make_sequences()
    sdm = SequenceDataModule(
        X.tolist(), Y.tolist(), 4, context_cols=[0, 2], test_frac=0.2, val_frac=0.25
    )
    assert sdm.context_cols == [0, 2]
    assert sdm.driving_cols == [1]


def test_sequence_split_is_by_whole_trajectory():
    X, Y = make_sequences()
    sdm = SequenceDataModule(X, Y, 4, test_frac=0.2, val_frac=0.25)
    parts = [set(sdm.idx_train), set(sdm.idx_val), set(sdm.idx_test)]
    assert [len(p) for p in parts] == [12, 4, 4]
    assert set.union(*parts) == set(range(20))
    assert sum(len(p) for p in parts) == 20

    train, val, test = sdm.labeled_data_loader()
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(test.sampler, SequentialSampler)
    np.testing.assert_array_equal(gather(test), X[sdm.idx_test].astype(np.float32))
    np.testing.assert_array_equal(gather(val, 1), Y[sdm.idx_val].astype(np.float32))
    xb, yb = next(iter(train))
    assert xb.shape == (4, 10, 3) and yb.shape == (4, 10, 2)


def test_sequence_phys_colloc_structure():
    X, Y = make_sequences()
    sdm = SequenceDataModule(
        X,
        Y,
        4,
        physics_coll_data_size=30,
        physics_coll_batch_size=8,
        n_segments=3,
        test_frac=0.2,
        val_frac=0.25,
    )
    loader = sdm.phys_colloc_loader()
    Xc = gather(loader)

    assert Xc.shape == (30, 10, 3)
    assert len(loader) == 4
    assert np.all(Xc >= sdm.lower_bnd - 1e-6) and np.all(Xc <= sdm.upper_bnd + 1e-6)
    # context feature is constant along each trajectory
    np.testing.assert_array_equal(Xc[:, :, 0], Xc[:, :1, 0].repeat(10, axis=1))
    # driving features follow a piecewise-constant schedule of 3 segments of
    # ceil(10 / 3) = 4 steps, the last one truncated to the horizon
    for col in (1, 2):
        for seg in (slice(0, 4), slice(4, 8), slice(8, 10)):
            block = Xc[:, seg, col]
            np.testing.assert_array_equal(block, block[:, :1].repeat(block.shape[1], 1))
        assert not np.array_equal(Xc[:, 0, col], Xc[:, 4, col])


def test_sequence_phys_colloc_options_and_default_size():
    X, Y = make_sequences()
    sdm = SequenceDataModule(X, Y, 4, test_frac=0.2, val_frac=0.25)
    with pytest.raises(ValueError, match="physics_coll_data_size must be"):
        sdm.phys_colloc_loader()
    sdm.physics_coll_data_size = 10
    with pytest.raises(ValueError, match="physics_coll_batch_size must be"):
        sdm.phys_colloc_loader()

    sdm.physics_coll_batch_size = 4
    loader = sdm.phys_colloc_loader(shuffle=False, drop_last=True)
    assert isinstance(loader.sampler, SequentialSampler)
    assert len(loader) == 2
    full = gather(sdm.phys_colloc_loader(shuffle=False))
    assert full.shape[0] == 10
    np.testing.assert_array_equal(gather(loader), full[:8])
