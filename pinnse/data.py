import numbers
import warnings
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.model_selection import train_test_split

# Column-name prefixes recognized as mole-fraction groups when `simplex_groups`
# is not given, and the tolerance on their row sums.
SIMPLEX_PREFIXES = ("Z_", "X_", "Y_")
SIMPLEX_TOL = 1e-4

# Default partition fractions (scikit-learn's own default test size).
DEFAULT_TEST_FRAC = 0.25
DEFAULT_VAL_FRAC = 0.25


def _frac(value: Optional[float], default: float) -> float:
    """Return a partition fraction, with None standing for the default."""
    return default if value is None else value


def _generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Return a seeded generator for DataLoader shuffling, or None (global RNG)."""
    return None if seed is None else torch.Generator().manual_seed(seed)


def _require_size(value: Optional[int], name: str) -> int:
    """
    Return a collocation size, rejecting a missing or non-positive value.

    The collocation sizes are optional at construction, since a purely
    data-driven model never builds a collocation loader, but building one
    without them would yield an empty or unbatched loader.
    """
    if value is None or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer to build this collocation "
            f"loader; received {value!r}."
        )
    return value


class DataModule:
    """
    Construct the labeled, physics-collocation and boundary-collocation
    DataLoaders of a steady-state (row-wise) formulation.

    Inputs
    ------
    I_S_data : pd.DataFrame
        Input-space dataset (normalized as the model expects), one row per
        sample. The last column is the boundary variable of
        `bnd_colloc_loader`.

    D_S_data : pd.DataFrame
        Output-space dataset, row-aligned with `I_S_data`.

    labeled_data_batch_size : int
        Number of labeled samples per batch.

    physics_coll_data_size, physics_coll_batch_size : int, optional
        Number of physics-collocation points and points per batch. Required
        only by `phys_colloc_loader`.

    boundary_coll_data_size, boundry_coll_batch_size : int, optional
        Number of boundary-collocation points and points per batch. Required
        only by `bnd_colloc_loader`.

    test_frac : float, optional, default=0.25
        Fraction of the labeled samples held out as the test partition.
        None is accepted and means the default.

    val_frac : float, optional, default=0.25
        Fraction of the remaining samples used for validation. None is
        accepted and means the default.

    random_state : int, optional, default=42
        Seed of the partitioning and of the collocation sampling.

    simplex_groups : list[list[str]], optional
        Groups of input columns that are mole fractions and are sampled
        jointly from a Dirichlet distribution, so that each group sums to one.
        Every row of the labeled data must sum to one over each group (within
        1e-4), so these columns must be left unnormalized. If None, groups are
        detected from the column prefixes ``Z_``, ``X_`` and ``Y_``; a
        detected group with fewer than two columns, or whose rows do not sum
        to one (for instance because it was normalized), is sampled like any
        other column instead, with a warning. Pass ``[]`` to disable
        Dirichlet sampling.

    shuffle_seed : int, optional
        If given, the shuffled loaders draw their batch order from a
        generator seeded with this value instead of PyTorch's global
        generator.
    """

    def __init__(
        self,
        I_S_data: pd.DataFrame,
        D_S_data: pd.DataFrame,
        labeled_data_batch_size: int,
        physics_coll_data_size: Optional[int] = None,
        physics_coll_batch_size: Optional[int] = None,
        boundary_coll_data_size: Optional[int] = None,
        boundry_coll_batch_size: Optional[int] = None,
        test_frac: Optional[float] = DEFAULT_TEST_FRAC,
        val_frac: Optional[float] = DEFAULT_VAL_FRAC,
        random_state: Optional[int] = 42,
        simplex_groups: Optional[list[list[str]]] = None,
        shuffle_seed: Optional[int] = None,
    ):
        self.I_S_data = I_S_data
        self.D_S_data = D_S_data
        self.labeled_data_batch_size = labeled_data_batch_size
        self.physics_coll_data_size = physics_coll_data_size
        self.physics_coll_batch_size = physics_coll_batch_size
        self.boundry_coll_data_size = boundary_coll_data_size
        self.boundry_coll_batch_size = boundry_coll_batch_size
        self.test_frac = _frac(test_frac, DEFAULT_TEST_FRAC)
        self.val_frac = _frac(val_frac, DEFAULT_VAL_FRAC)
        self.random_state = random_state
        self.shuffle_seed = shuffle_seed

        # Input bounds over the whole labeled dataset, used by the collocation
        # loaders, so that those may be built in any order.
        X = self.I_S_data.to_numpy(dtype=np.float32)
        self.lower_bnd = X.min(axis=0)
        self.upper_bnd = X.max(axis=0)

        self.simplex_groups = self._resolve_simplex_groups(simplex_groups)

    def _resolve_simplex_groups(
        self, simplex_groups: Optional[list[list[str]]]
    ) -> list[list[str]]:
        """
        Validate the mole-fraction groups sampled from Dirichlet distributions.

        Explicit groups that are invalid raise a ValueError; groups detected
        from the column prefixes that are invalid are dropped with a warning.
        """
        in_cols = list(self.I_S_data.columns)
        explicit = simplex_groups is not None
        if explicit:
            groups = [list(g) for g in simplex_groups]
        else:
            groups = [
                [col for col in in_cols if col.startswith(prefix)]
                for prefix in SIMPLEX_PREFIXES
            ]
            groups = [g for g in groups if g]

        resolved: list[list[str]] = []
        used: set[str] = set()
        for group in groups:
            unknown = [col for col in group if col not in in_cols]
            if unknown:
                problem = f"unknown input columns {unknown}"
            elif len(group) < 2:
                problem = "a mole-fraction group needs at least two columns"
            elif used.intersection(group):
                problem = "its columns already belong to another group"
            else:
                sums = self.I_S_data[group].to_numpy(dtype=np.float64).sum(axis=1)
                deviation = float(np.max(np.abs(sums - 1.0)))
                problem = (
                    None
                    if deviation <= SIMPLEX_TOL
                    else (
                        "its rows do not sum to one (largest deviation "
                        f"{deviation:.3g}); mole fractions must be left "
                        "unnormalized"
                    )
                )

            if problem is None:
                resolved.append(group)
                used.update(group)
            elif explicit:
                raise ValueError(f"Invalid simplex group {group}: {problem}.")
            else:
                warnings.warn(
                    f"Columns {group} match a mole-fraction prefix "
                    f"({', '.join(SIMPLEX_PREFIXES)}) but {problem}. They are "
                    "sampled by Latin hypercube sampling within their bounds "
                    "instead; pass `simplex_groups` to DataModule to state the "
                    "groups explicitly.",
                    UserWarning,
                    stacklevel=3,
                )
        return resolved

    def labeled_data_loader(self):
        """
        Construct labeled DataLoaders for training, validation, and test datasets.

        Inputs
        ------
        None
            Uses I_S as the input dataset and D_S as the output dataset.

        Returns
        -------
        train_loader : torch.utils.data.DataLoader
            DataLoader containing the training subset.

        val_loader : torch.utils.data.DataLoader
            DataLoader containing the validation subset.

        test_loader : torch.utils.data.DataLoader
            DataLoader containing the test subset.
        """
        X = self.I_S_data.to_numpy(dtype=np.float32)
        Y = self.D_S_data.to_numpy(dtype=np.float32)

        X_tv, X_test, Y_tv, Y_test = train_test_split(
            X, Y, test_size=self.test_frac, random_state=self.random_state
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_tv, Y_tv, test_size=self.val_frac, random_state=self.random_state
        )

        def make_loader(
            X: np.ndarray,
            Y: np.ndarray,
            shuffle: bool,
            drop_last: bool = False,
        ):
            X_t = torch.from_numpy(X)
            Y_t = torch.from_numpy(Y)
            dataset = TensorDataset(X_t, Y_t)
            return DataLoader(
                dataset=dataset,
                batch_size=self.labeled_data_batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                generator=_generator(self.shuffle_seed) if shuffle else None,
            )

        train_loader = make_loader(X_train, Y_train, shuffle=True)
        val_loader = make_loader(X_val, Y_val, shuffle=False)
        test_loader = make_loader(X_test, Y_test, shuffle=False)

        return train_loader, val_loader, test_loader

    def phys_colloc_loader(
        self,
        shuffle: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        drop_last: bool = False,
    ):
        """
        Generate a physics-collocation DataLoader by sampling the surrogate input space.

        Inputs
        ------
        shuffle : bool, optional, default=True
            Whether to shuffle the collocation samples in the returned DataLoader.

        alpha : float, optional, default=1.0
            Dirichlet concentration parameter used for sampling the mole-fraction
            groups (see `simplex_groups`). A value of 1.0 gives a uniform
            distribution over the simplex.

        drop_last : bool, optional, default=False
            Whether to drop the last incomplete batch in the DataLoader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader containing collocation inputs only.
            Each batch has the form (X_coll,), where X_coll is a tensor of shape
            (batch_size, n_input_features).

        Notes
        -----
        - Each mole-fraction group (see `simplex_groups`) is sampled from a
          Dirichlet distribution over the whole simplex, so that it sums to 1.
        - All other input variables are sampled using Latin Hypercube Sampling
          within the bounds of the labeled data.
        """
        N_colloc = _require_size(self.physics_coll_data_size, "physics_coll_data_size")
        _require_size(self.physics_coll_batch_size, "physics_coll_batch_size")
        rng = np.random.default_rng(self.random_state)

        in_cols = list(self.I_S_data.columns)
        grouped = {col for group in self.simplex_groups for col in group}
        box_cols = [col for col in in_cols if col not in grouped]

        parts = {}

        # Latin Hypercube sampling for ordinary bounded variables
        if box_cols:
            box_idx = [in_cols.index(col) for col in box_cols]
            lb = self.lower_bnd[box_idx].astype(np.float32)
            ub = self.upper_bnd[box_idx].astype(np.float32)

            sampler = qmc.LatinHypercube(d=len(box_cols), rng=rng)
            unit = sampler.random(N_colloc)
            X_box = qmc.scale(unit, lb, ub).astype(np.float32)
            for j, col in enumerate(box_cols):
                parts[col] = X_box[:, j : j + 1]

        # Dirichlet sampling for mole-fraction groups
        for cols in self.simplex_groups:
            X_dir = rng.dirichlet(
                alpha=np.full(len(cols), alpha, dtype=np.float32), size=N_colloc
            ).astype(np.float32)

            for j, col in enumerate(cols):
                parts[col] = X_dir[:, j : j + 1]

        # Reassemble in original I_S column order
        X_coll = np.hstack([parts[col] for col in in_cols]).astype(np.float32)
        X_coll = torch.from_numpy(X_coll)

        dataset = TensorDataset(X_coll)
        return DataLoader(
            dataset=dataset,
            batch_size=self.physics_coll_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=_generator(self.shuffle_seed) if shuffle else None,
        )

    def bnd_colloc_loader(
        self,
        shuffle: bool = True,
        drop_last: bool = False,
        bnd_value: Optional[float] = None,
    ):
        """
        Construct a boundary-collocation DataLoader by fixing the last input column
        to a prescribed boundary value and sampling all preceding input columns
        within their admissible bounds.

        Inputs
        ------
        shuffle : bool, optional, default=True
            Whether to shuffle the collocation samples.

        drop_last : bool, optional, default=False
            Whether to drop the last incomplete batch.

        bnd_value : float, optional, default=None
            Value imposed on the last input column, in the same (normalized)
            units as `I_S_data`. If None, the smallest value of that column in
            the labeled data is used: -1 for data scaled to [-1, 1], 0 for data
            scaled to [0, 1]. A value outside the range of the labeled data
            triggers a warning.

        Returns
        -------
        DataLoader
            DataLoader containing boundary-collocation inputs only.

        Notes
        -----
        - The last column of `I_S` is treated as the boundary variable.
        - All other input variables are sampled using Latin Hypercube Sampling
          over their admissible bounds.
        """
        N_bc = _require_size(self.boundry_coll_data_size, "boundary_coll_data_size")
        _require_size(self.boundry_coll_batch_size, "boundry_coll_batch_size")
        rng = np.random.default_rng(self.random_state)
        dim_in = self.I_S_data.shape[1]

        lb_last, ub_last = float(self.lower_bnd[-1]), float(self.upper_bnd[-1])
        if bnd_value is None:
            bnd_value = lb_last
        elif not lb_last <= bnd_value <= ub_last:
            warnings.warn(
                f"bnd_value={bnd_value} lies outside the range "
                f"[{lb_last:.6g}, {ub_last:.6g}] of the boundary column "
                f"{self.I_S_data.columns[-1]!r} in the labeled data; check that it "
                "is expressed in the same (normalized) units.",
                UserWarning,
                stacklevel=2,
            )

        low = self.lower_bnd[:-1].astype(np.float32)
        up = self.upper_bnd[:-1].astype(np.float32)

        sampler = qmc.LatinHypercube(d=dim_in - 1, rng=rng)
        unit = sampler.random(N_bc)
        X_free = qmc.scale(unit, low, up).astype(np.float32)

        X_bc = np.hstack([X_free, np.full((N_bc, 1), bnd_value, dtype=np.float32)])
        X_bc = torch.from_numpy(X_bc)
        dataset = TensorDataset(X_bc)
        return DataLoader(
            dataset=dataset,
            batch_size=self.boundry_coll_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=_generator(self.shuffle_seed) if shuffle else None,
        )

    def inspect_loader(self, name: str, loader: torch.utils.data.DataLoader):
        """
        Inspect a constructed DataLoader.

        Inputs
        ------
        name : str
            Name of the loader to be displayed in the printed summary.

        loader : torch.utils.data.DataLoader
            DataLoader to inspect. The loader may contain either:
            - labeled batches of the form (X, Y), or
            - unlabeled batches of the form (X,).

        Returns
        -------
        None
            Prints the shape of the batch tensors and the number of batches
            in the loader.

        Notes
        -----
        - For labeled loaders, prints the shapes of both input and output tensors.
        - For unlabeled (collocattion) loaders, prints only the shape of the input tensor.
        """
        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, Y = batch
            print(
                f"{name}_loader -> X: {X.shape}, Y: {Y.shape}, batches: {len(loader)}"
            )
        else:
            X = batch[0] if isinstance(batch, (list, tuple)) else batch
            print(f"{name}_loader -> X: {X.shape}, batches: {len(loader)}")
        return

    def save_loaders(self, loader, filename):
        """
        Save the contents of a DataLoader to an Excel file.

        Inputs
        ------
        loader : torch.utils.data.DataLoader
            DataLoader to save. The loader may contain either:

            - labeled batches of the form (X, Y), or
            - unlabeled batches of the form (X,).

        filename : str
            Name or path of the output Excel file.

        Returns
        -------

        Notes
        -----
        - For labeled loaders, the saved file contains both input and output
          columns concatenated side by side.
        - For unlabeled loaders, only the input columns are saved.
        """
        X_all, Y_all = [], []

        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X, Y = batch
                X_all.append(X.detach().cpu().numpy())
                Y_all.append(Y.detach().cpu().numpy())
            else:
                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                X_all.append(X.detach().cpu().numpy())

        if not X_all:
            raise ValueError("The provided DataLoader is empty.")

        X_all = np.vstack(X_all)
        df_X = pd.DataFrame(X_all, columns=self.I_S_data.columns)

        if Y_all:
            Y_all = np.vstack(Y_all)
            df_Y = pd.DataFrame(Y_all, columns=self.D_S_data.columns)
            df = pd.concat([df_X, df_Y], axis=1)
        else:
            df = df_X

        df.to_excel(filename, index=False)
        print(f"Saved {len(df)} samples to {filename}")


class SequenceDataModule:
    """
    Construct DataLoaders for sequence (trajectory) formulations.

    Counterpart of `DataModule` for models whose samples are whole trajectories
    rather than independent rows. Inputs and outputs are three-dimensional
    arrays of shape (n_trajectories, seq_len, n_features), partitioning is by
    whole trajectory so that no trajectory contributes to more than one
    partition, and physics-collocation samples are trajectories rather than
    points.

    Inputs
    ------
    I_S_data : np.ndarray
        Normalized input sequences of shape (n_trajectories, seq_len, dim_in).

    D_S_data : np.ndarray
        Normalized output sequences of shape (n_trajectories, seq_len, dim_ot).

    labeled_data_batch_size : int
        Number of trajectories per labeled batch.

    physics_coll_data_size : int, optional
        Number of collocation trajectories to generate.

    physics_coll_batch_size : int, optional
        Number of collocation trajectories per batch.

    context_cols : list[int], optional
        Indices of the input features that stay constant along a trajectory,
        such as an initial condition carried as a context feature. If omitted,
        they are detected from the labeled data.

    n_segments : int, optional, default=1
        Number of piecewise-constant segments used when sampling the remaining
        (driving) input features for collocation. Must be a positive integer.

    test_frac : float, optional, default=0.25
        Fraction of trajectories held out as the test partition. None is
        accepted and means the default.

    val_frac : float, optional, default=0.25
        Fraction of the remaining trajectories used for validation. None is
        accepted and means the default.

    random_state : int, optional, default=42
        Seed used for partitioning and for collocation sampling.

    shuffle_seed : int, optional
        If given, the shuffled loaders draw their batch order from a
        generator seeded with this value instead of PyTorch's global
        generator.
    """

    def __init__(
        self,
        I_S_data: np.ndarray,
        D_S_data: np.ndarray,
        labeled_data_batch_size: int,
        physics_coll_data_size: Optional[int] = None,
        physics_coll_batch_size: Optional[int] = None,
        context_cols: Optional[list[int]] = None,
        n_segments: int = 1,
        test_frac: Optional[float] = DEFAULT_TEST_FRAC,
        val_frac: Optional[float] = DEFAULT_VAL_FRAC,
        random_state: Optional[int] = 42,
        shuffle_seed: Optional[int] = None,
    ):
        I_S_data = np.asarray(I_S_data)
        D_S_data = np.asarray(D_S_data)

        if (
            isinstance(n_segments, bool)
            or not isinstance(n_segments, numbers.Integral)
            or n_segments < 1
        ):
            raise ValueError(
                f"n_segments must be a positive integer; received {n_segments!r}."
            )

        if I_S_data.ndim != 3 or D_S_data.ndim != 3:
            raise ValueError(
                "SequenceDataModule expects arrays of shape "
                "(n_trajectories, seq_len, n_features); received "
                f"{I_S_data.shape} and {D_S_data.shape}."
            )
        if I_S_data.shape[:2] != D_S_data.shape[:2]:
            raise ValueError(
                "Input and output sequences must agree in the number of "
                f"trajectories and steps; received {I_S_data.shape[:2]} and "
                f"{D_S_data.shape[:2]}."
            )

        self.I_S_data = I_S_data
        self.D_S_data = D_S_data
        self.labeled_data_batch_size = labeled_data_batch_size
        self.physics_coll_data_size = physics_coll_data_size
        self.physics_coll_batch_size = physics_coll_batch_size
        self.n_segments = int(n_segments)
        self.test_frac = _frac(test_frac, DEFAULT_TEST_FRAC)
        self.val_frac = _frac(val_frac, DEFAULT_VAL_FRAC)
        self.random_state = random_state
        self.shuffle_seed = shuffle_seed

        self.n_traj, self.seq_len, self.dim_in = I_S_data.shape

        self.lower_bnd, self.upper_bnd = (
            I_S_data.min(axis=(0, 1)),
            I_S_data.max(axis=(0, 1)),
        )

        # initial & boundary conditions are held constant when sampling collocation
        if context_cols is None:
            spread = (I_S_data.max(axis=1) - I_S_data.min(axis=1)).max(axis=0)
            context_cols = np.flatnonzero(spread == 0).tolist()
        self.context_cols = list(context_cols)
        self.driving_cols = [
            j for j in range(self.dim_in) if j not in self.context_cols
        ]

        # Partition by whole trajectory
        idx = np.arange(self.n_traj)
        idx_tv, self.idx_test = train_test_split(
            idx, test_size=self.test_frac, random_state=self.random_state
        )
        self.idx_train, self.idx_val = train_test_split(
            idx_tv, test_size=self.val_frac, random_state=self.random_state
        )

    def labeled_data_loader(self):
        """
        Construct labeled DataLoaders for the training, validation and test
        partitions, batched over whole trajectories.

        Returns
        -------
        train_loader : torch.utils.data.DataLoader
            DataLoader containing the training trajectories.

        val_loader : torch.utils.data.DataLoader
            DataLoader containing the validation trajectories.

        test_loader : torch.utils.data.DataLoader
            DataLoader containing the test trajectories.

        Notes
        -----
        - Each batch has the form (X, Y) with respective shapes of
          (batch_size, seq_len, dim_in) and (batch_size, seq_len, dim_ot).
        """
        X = self.I_S_data.astype(np.float32)
        Y = self.D_S_data.astype(np.float32)

        def make_loader(idx: np.ndarray, shuffle: bool, drop_last: bool = False):
            dataset = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(Y[idx]))
            return DataLoader(
                dataset=dataset,
                batch_size=self.labeled_data_batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                generator=_generator(self.shuffle_seed) if shuffle else None,
            )

        train_loader = make_loader(self.idx_train, shuffle=True)
        val_loader = make_loader(self.idx_val, shuffle=False)
        test_loader = make_loader(self.idx_test, shuffle=False)

        return train_loader, val_loader, test_loader

    def phys_colloc_loader(self, shuffle: bool = True, drop_last: bool = False):
        """
        Generate a physics-collocation DataLoader of unlabeled trajectories.

        Inputs
        ------
        shuffle : bool, optional, default=True
            Whether to shuffle the collocation trajectories.

        drop_last : bool, optional, default=False
            Whether to drop the last incomplete batch.

        Returns
        -------
        DataLoader
            DataLoader containing collocation inputs only. Each batch has the
            form (X_coll,), where X_coll has shape (batch_size, seq_len, dim_in).

        Notes
        -----
        - Context features are sampled once per trajectory and held constant
          along it; driving features are sampled as `n_segments` levels and
          expanded into a piecewise-constant schedule.
        - All features are sampled using Latin Hypercube Sampling
        """
        N_colloc = _require_size(self.physics_coll_data_size, "physics_coll_data_size")
        _require_size(self.physics_coll_batch_size, "physics_coll_batch_size")
        rng = np.random.default_rng(self.random_state)

        n_ctx = len(self.context_cols)
        sampler = qmc.LatinHypercube(
            d=n_ctx + len(self.driving_cols) * self.n_segments, rng=rng
        )
        unit = sampler.random(N_colloc)

        X_coll = np.empty((N_colloc, self.seq_len, self.dim_in), dtype=np.float32)

        for i, col in enumerate(self.context_cols):
            lb, ub = self.lower_bnd[col], self.upper_bnd[col]
            X_coll[:, :, col] = (lb + unit[:, i] * (ub - lb))[:, None]

        per_seg = int(np.ceil(self.seq_len / self.n_segments))
        for i, col in enumerate(self.driving_cols):
            start = n_ctx + i * self.n_segments
            levels = unit[:, start : start + self.n_segments]
            lb, ub = self.lower_bnd[col], self.upper_bnd[col]
            levels = lb + levels * (ub - lb)
            X_coll[:, :, col] = np.repeat(levels, per_seg, axis=1)[:, : self.seq_len]

        dataset = TensorDataset(torch.from_numpy(X_coll))
        return DataLoader(
            dataset=dataset,
            batch_size=self.physics_coll_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=_generator(self.shuffle_seed) if shuffle else None,
        )
