import pandas as pd
import numpy as np
import torch
from typing import Optional
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import qmc


class DataModule:
    def __init__(
        self,
        I_S_data: pd.DataFrame,
        D_S_data: pd.DataFrame,
        labeled_data_batch_size: int,
        physics_coll_data_size: Optional[int] = None,
        physics_coll_batch_size: Optional[int] = None,
        boundary_coll_data_size: Optional[int] = None,
        boundry_coll_batch_size: Optional[int] = None,
        test_frac: Optional[float] = None,
        val_frac: Optional[float] = None,
        random_state: Optional[int] = 42,
    ):
        self.I_S_data = I_S_data
        self.D_S_data = D_S_data
        self.labeled_data_batch_size = labeled_data_batch_size
        self.physics_coll_data_size = physics_coll_data_size
        self.physics_coll_batch_size = physics_coll_batch_size
        self.boundry_coll_data_size = boundary_coll_data_size
        self.boundry_coll_batch_size = boundry_coll_batch_size
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.random_state = random_state

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

        self.lower_bnd = X.min(axis=0)
        self.upper_bnd = X.max(axis=0)

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
            Dirichlet concentration parameter used for sampling molar-fraction
            variables whose names start with 'Z_', 'X_', or 'Y_'.
            A value of 1.0 gives a uniform distribution over the simplex.

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
        - Variables with prefixes 'Z_', 'X_', and 'Y_' are sampled using
        Dirichlet distributions so that each group sums to 1.
        - All other input variables are sampled using Latin Hypercube Sampling
        """
        rng = np.random.default_rng(self.random_state)
        N_colloc = (
            self.physics_coll_data_size
            if self.physics_coll_data_size is not None
            else 0
        )

        in_cols = list(self.I_S_data.columns)
        simplex_prefixes = ("Z_", "X_", "Y_")

        box_cols = [col for col in in_cols if not col.startswith(simplex_prefixes)]
        dirichlet_groups = {
            prefix: [col for col in in_cols if col.startswith(prefix)]
            for prefix in simplex_prefixes
        }

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

        # Dirichlet sampling for molar fraction variables
        for prefix, cols in dirichlet_groups.items():
            if cols:
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
        )

    def bnd_colloc_loader(
        self, shuffle: bool = True, drop_last: bool = False, bnd_value: float = -1.0
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

        boundary_value : float, optional, default=-1.0
            Boundary value imposed on the last input column (normalized).

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
        rng = np.random.default_rng(self.random_state)
        dim_in = self.I_S_data.shape[1]

        N_bc = (
            self.boundry_coll_data_size
            if self.boundry_coll_data_size is not None
            else 0
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
