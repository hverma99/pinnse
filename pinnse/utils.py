import pandas as pd, numpy as np, os
import torch.nn as nn, torch
from pinnse import ANN


class Normalization:
    @staticmethod
    def min_max(data: pd.DataFrame, normalize_cols: list[str] | None = None):
        """
        Min-max normalize only selected columns.

        Inputs
        ------
        data : pd.DataFrame
            Input DataFrame.
        normalize_cols : list[str], optional
            Columns to normalize. If None, all columns are normalized.

        Returns
        -------
        data_norm : pd.DataFrame
            DataFrame with selected columns normalized.
        metrics : dict
            Dictionary of min/max values for normalized columns only.
        """
        data_norm, metrics = data.copy(), {}
        cols = normalize_cols if normalize_cols is not None else list(data.columns)
        for col in cols:
            cmin, cmax = data[col].min(), data[col].max()
            data_norm[col] = (data[col] - cmin) / (cmax - cmin)
            metrics[col] = {"min": float(cmin), "max": float(cmax)}
        return data_norm, metrics

    @staticmethod
    def min_max_defined_metrics(data, metrics):
        """
        Apply min-max normalization using precomputed (defined) metrics.
        """
        data_norm = data.copy()
        for col, vals in metrics.items():
            cmin = vals["min"]
            cmax = vals["max"]
            data_norm[col] = (data_norm[col] - cmin) / (cmax - cmin)
        return data_norm

    @staticmethod
    def scale_centered(data: pd.DataFrame, col: str, cmin: float, cmax: float):
        """
        Scale one column to the centered min-max range [-1, 1].
        """
        if cmax == cmin:
            return np.zeros(len(data), dtype=np.float32)
        return (2.0 * (data[col] - cmin) / (cmax - cmin) - 1.0).astype(np.float32)

    @staticmethod
    def apply_global(
        data: pd.DataFrame,
        data_norm: pd.DataFrame,
        norm_metrics: dict,
        cols: list[str],
        gmin: float | None = None,
        gmax: float | None = None,
    ):
        """
        Apply centered min-max normalization to a group of columns using one shared
        global min and max over all columns in the group.
        """
        if not cols:
            return

        if gmin is None or gmax is None:
            vals = data[cols].values.astype(np.float32)
            gmin, gmax = vals.min(), vals.max()

        if gmin is not None and gmax is not None:
            for col in cols:
                data_norm[col] = Normalization.scale_centered(data, col, gmin, gmax)
                norm_metrics[col] = {"min": float(gmin), "max": float(gmax)}

    @staticmethod
    def apply_local(
        data: pd.DataFrame, data_norm: pd.DataFrame, norm_metrics: dict, cols: list[str]
    ):
        """
        Apply centered min-max normalization to each column independently using its
        own min and max.
        """
        for col in cols:
            cmin, cmax = data[col].min(), data[col].max()
            data_norm[col] = Normalization.scale_centered(data, col, cmin, cmax)
            norm_metrics[col] = {"min": float(cmin), "max": float(cmax)}

    @staticmethod
    def min_max_pfr(
        I_S_data: pd.DataFrame,
        D_S_data: pd.DataFrame,
        formulation: str,
        species: list[str],
        N_reaction: int = 3,
    ):
        """
        Centered min-max normalization to [-1, 1] for PFR formulations.

        Inputs
        ------
        I_S_data : pd.DataFrame
            Input-space dataset.
        D_S_data : pd.DataFrame
            Output-space dataset.
        formulation : str
            PFR formulation: "EFM", "ERM", or "CM".
        species : list[str]
            Species names.
        N_reaction : int, optional
            Number of reactions. Required for ERM and CM.

        Returns
        -------
        norm_I_S_data : pd.DataFrame
            Normalized input-space dataset.
        norm_D_S_data : pd.DataFrame
            Normalized output-space dataset.
        norm_I_S_metrics : dict
            Input normalization metrics.
        norm_D_S_metrics : dict
            Output normalization metrics.
        """
        formulation = formulation.upper()

        norm_I_S_data = I_S_data.copy()
        norm_D_S_data = D_S_data.copy()
        norm_I_S_metrics = {}
        norm_D_S_metrics = {}

        common_I_S_cols = [col for col in ["P", "T", "V"] if col in I_S_data.columns]

        if formulation == "EFM":
            in_cols = [
                f"F_in_{sp}" for sp in species if f"F_in_{sp}" in I_S_data.columns
            ]
            ot_cols = [
                f"F_ot_{sp}" for sp in species if f"F_ot_{sp}" in D_S_data.columns
            ]

            flow_vals = []
            if in_cols:
                flow_vals.append(I_S_data[in_cols].values.astype(np.float32).ravel())
            if ot_cols:
                flow_vals.append(D_S_data[ot_cols].values.astype(np.float32).ravel())

            if flow_vals:
                all_flow = np.concatenate(flow_vals)
                gmin, gmax = all_flow.min(), all_flow.max()
                Normalization.apply_global(
                    I_S_data, norm_I_S_data, norm_I_S_metrics, in_cols, gmin, gmax
                )
                Normalization.apply_global(
                    D_S_data, norm_D_S_data, norm_D_S_metrics, ot_cols, gmin, gmax
                )

            Normalization.apply_local(
                I_S_data, norm_I_S_data, norm_I_S_metrics, common_I_S_cols
            )

        elif formulation == "ERM":
            in_cols = [
                f"F_in_{sp}" for sp in species if f"F_in_{sp}" in I_S_data.columns
            ]
            E_cols = [
                f"E_R{j+1}"
                for j in range(N_reaction)
                if f"E_R{j+1}" in D_S_data.columns
            ]

            Normalization.apply_global(
                I_S_data, norm_I_S_data, norm_I_S_metrics, in_cols
            )
            Normalization.apply_local(
                I_S_data, norm_I_S_data, norm_I_S_metrics, common_I_S_cols
            )
            Normalization.apply_local(D_S_data, norm_D_S_data, norm_D_S_metrics, E_cols)

        elif formulation == "CM":
            in_cols = [
                f"F_in_{sp}" for sp in species if f"F_in_{sp}" in I_S_data.columns
            ]
            X_cols = [
                f"X_R{j+1}"
                for j in range(N_reaction)
                if f"X_R{j+1}" in D_S_data.columns
            ]

            Normalization.apply_local(
                I_S_data, norm_I_S_data, norm_I_S_metrics, in_cols
            )
            Normalization.apply_local(
                I_S_data, norm_I_S_data, norm_I_S_metrics, common_I_S_cols
            )
            Normalization.apply_local(D_S_data, norm_D_S_data, norm_D_S_metrics, X_cols)

        else:
            raise ValueError("formulation must be one of: 'EFM', 'ERM', 'CM'")

        return norm_I_S_data, norm_D_S_data, norm_I_S_metrics, norm_D_S_metrics

    @staticmethod
    def scale_centered_defined_metrics(I_S: pd.DataFrame, I_S_metrics: dict):
        """
        Apply centered min-max normalization to [-1, 1] using predefined metrics.
        """
        I_S_norm = I_S.copy()
        for col in I_S.columns:
            cmin = I_S_metrics[col]["min"]
            cmax = I_S_metrics[col]["max"]
            I_S_norm[col] = Normalization.scale_centered(I_S, col, cmin, cmax)
        return I_S_norm

    @staticmethod
    def max_abs(data: pd.DataFrame, normalize_cols: list[str] | None = None):
        """
        Max-abs normalize only selected columns.

        Inputs
        ------
        data : pd.DataFrame
            Input DataFrame.
        normalize_cols : list[str], optional
            Columns to normalize. If None, all columns are normalized.

        Returns
        -------
        data_norm : pd.DataFrame
            DataFrame with selected columns normalized.
        metrics : dict
            Dictionary of max absolute values for normalized columns only.
        """
        data_norm, metrics = data.copy(), {}
        cols = normalize_cols if normalize_cols is not None else list(data.columns)
        for col in cols:
            cmax = abs(data[col]).max()
            data_norm[col] = data[col] / cmax
            metrics[col] = {"max": float(cmax)}
        return data_norm, metrics

    @staticmethod
    def mean_norm(data: pd.DataFrame, normalize_cols: list[str] | None = None):
        """
        Mean normalize only selected columns.

        Inputs
        ------
        data : pd.DataFrame
            Input DataFrame.
        normalize_cols : list[str], optional
            Columns to normalize. If None, all columns are normalized.

        Returns
        -------
        data_norm : pd.DataFrame
            DataFrame with selected columns normalized.
        metrics : dict
            Dictionary of mean, max, and min values for normalized columns only.
        """
        data_norm, metrics = data.copy(), {}
        cols = normalize_cols if normalize_cols is not None else list(data.columns)
        for col in cols:
            cmin, cmax = data[col].min(), data[col].max()
            cmean = data[col].mean()
            data_norm[col] = (data[col] - cmean) / (cmax - cmin)
            metrics[col] = {
                "max": float(cmax),
                "min": float(cmin),
                "mean": float(cmean),
            }
        return data_norm, metrics

    @staticmethod
    def z_score(data: pd.DataFrame, normalize_cols: list[str] | None = None):
        """
        Z-score normalization of only selected columns.

        Inputs
        ------
        data : pd.DataFrame
            Input DataFrame.
        normalize_cols : list[str], optional
            Columns to normalize. If None, all columns are normalized.

        Returns
        -------
        data_norm : pd.DataFrame
            DataFrame with selected columns normalized.
        metrics : dict
            Dictionary of mean and standard deviation values for normalized columns only.
        """
        data_norm, metrics = data.copy(), {}
        cols = normalize_cols if normalize_cols is not None else list(data.columns)
        for col in cols:
            cmean, cstd = data[col].mean(), data[col].std()
            data_norm[col] = (data[col] - cmean) / cstd
            metrics[col] = {"mean": float(cmean), "std": float(cstd)}
        return data_norm, metrics


class Denormalization:
    @staticmethod
    def min_max(data_norm, metrics):
        """
        Inverse of min-max normalization
        z = z_norm * (z_max - z_min) + z_min

        Inputs
        ------
        data_norm : pd.DataFrame (Normalized DataFrame)

        metrics : dict
                Dictionary of the form:
                {column: {"min": ..., "max": ...}}

        Returns
        -------
        data : pd.DataFrame (Denormalized DataFrame)
        """
        data = data_norm.copy()
        if not metrics:
            return data

        for col in data.columns:
            if col not in metrics:
                continue
            cmin = metrics[col]["min"]
            cmax = metrics[col]["max"]
            data[col] = data[col] * (cmax - cmin) + cmin
        return data

    @staticmethod
    def min_max_col(x_norm, col, metrics):
        """
        Denormalize a single variable using min-max metrics.

        Inputs
        ------
        x_norm  : torch.Tensor, np.ndarray, or scalar (Normalized variable)
        col     : str (Column name whose min/max values are used)

        metrics : dict (Dictionary of normalization metrics)

        Returns
        -------
        x       : same type as x_norm (Denormalized variable)
        """
        if not metrics or col not in metrics:
            return x_norm
        cmin = metrics[col]["min"]
        cmax = metrics[col]["max"]
        return x_norm * (cmax - cmin) + cmin

    @staticmethod
    def min_max_pfr(X: torch.Tensor, norm_metrics: dict, keys: list[str]):
        """
        Denormalize centered min-max PFR variables from [-1, 1] to dimensional values.

        Inputs
        ------
        X           : torch.Tensor
            Normalized tensor of shape (batch, k).
        norm_metrics: dict
            Dictionary storing min/max for each key.
        keys        : list[str]
            Column names corresponding to the columns of X.

        Returns
        -------
        X_dim : torch.Tensor
            Dimensional tensor of shape (batch, k).
        rngs  : torch.Tensor
            Column-wise ranges (max - min).
        """
        device, dtype = X.device, X.dtype
        k = len(keys)

        mins = torch.tensor(
            [norm_metrics[key]["min"] for key in keys], device=device, dtype=dtype
        )
        rngs = torch.tensor(
            [norm_metrics[key]["max"] - norm_metrics[key]["min"] for key in keys],
            device=device,
            dtype=dtype,
        )
        X_dim = mins.view(1, k) + 0.5 * (X + 1.0) * rngs.view(1, k)
        return X_dim, rngs

    @staticmethod
    def max_abs(data_norm, metrics):
        """
        Inverse of max_abs normalization
        z = z_norm * max(abs(z))

        Inputs
        ------
        data_norm : pd.DataFrame (Normalized DataFrame)

        metrics : dict
                Dictionary of the form:
                {column: {"max": ...}}

        Returns
        -------
        data : pd.DataFrame (Denormalized DataFrame)
        """
        data = data_norm.copy()
        if not metrics:
            return data

        for col in data.columns:
            if col not in metrics:
                continue
            cmax = metrics[col]["max"]
            data[col] = data[col] * cmax
        return data

    @staticmethod
    def mean_abs(data_norm, metrics):
        """
        Inverse of mean normalization
        z = z_norm * (z_max - z_min) + z_mean

        Inputs
        ------
        data_norm : pd.DataFrame (Normalized DataFrame)

        metrics : dict
                Dictionary of the form:
                {column: {"min": ..., "max": ..., "mean":...}}

        Returns
        -------
        data : pd.DataFrame (Denormalized DataFrame)
        """
        data = data_norm.copy()
        if not metrics:
            return data

        for col in data.columns:
            if col not in metrics:
                continue
            cmin, cmax = metrics[col]["min"], metrics[col]["max"]
            cmean = metrics[col]["mean"]
            data[col] = data[col] * (cmax - cmin) + cmean
        return data

    @staticmethod
    def z_norm(data_norm, metrics):
        """
        Inverse of Z-score normalization
        z = z_norm * z_std + z_mean

        Inputs
        ------
        data_norm : pd.DataFrame (Normalized DataFrame)

        metrics : dict
                Dictionary of the form:
                {column: {"mean": ..., "std":...}}

        Returns
        -------
        data : pd.DataFrame (Denormalized DataFrame)
        """
        data = data_norm.copy()
        if not metrics:
            return data

        for col in data.columns:
            if col not in metrics:
                continue
            cmean, cstd = metrics[col]["mean"], metrics[col]["std"]
            data[col] = data[col] * cstd + cmean
        return data


class Analyze:
    @staticmethod
    def load_ann_pinn(
        dim_in: int,
        dim_out: int,
        depth: int,
        width: int,
        activation: type[nn.Module],
        ckpt_path: str,
        device: torch.device,
    ):
        """
        Load a trained ANN/PINN model from checkpoint.

        Inputs
        ------
        dim_in, dim_out : int
            Input and output dimensions.
        depth, width : int
            Number of hidden layers and neurons per hidden layer.
        activation : type[nn.Module]
            Activation function class.
        ckpt_path : str
            Path to saved checkpoint.
        device : torch.device
            Device to load the model on.

        Returns
        -------
        model : nn.Module
            Loaded model in evaluation mode.
        """
        layer_size = [dim_in] + [width] * depth + [dim_out]
        model = ANN(layer_size=layer_size, activation=activation).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()
        return model

    @staticmethod
    def pinn_eval(
        model: nn.Module,
        I_S_model: pd.DataFrame,
        I_S_metrics: dict,
        D_S_metrics: dict,
        output_cols: list[str],
        device: torch.device,
    ):
        """
        Evaluate the trained model on selected inputs and denormalize the outputs.

        Inputs
        ------
        model : nn.Module
            Trained neural network model.
        I_S_model : pd.DataFrame
            Input DataFrame for model evaluation.
        I_S_metrics : dict
            Input normalization metrics.
        D_S_metrics : dict
            Output normalization metrics.
        output_cols : list[str]
            Output column names.
        device : torch.device
            Device for model evaluation.

        Returns
        -------
        D_S_pred : pd.DataFrame
            Denormalized model predictions.
        """
        I_S_model_norm = Normalization.min_max_defined_metrics(I_S_model, I_S_metrics)
        x = torch.tensor(I_S_model_norm.to_numpy(), dtype=torch.float32).to(device)

        with torch.no_grad():
            y_pred_norm = model(x).cpu().numpy()

        D_S_pred_norm = pd.DataFrame(y_pred_norm, columns=output_cols)
        D_S_pred = Denormalization.min_max(D_S_pred_norm, D_S_metrics)
        return D_S_pred

    @staticmethod
    def pinn_eval_pfr(
        model: nn.Module,
        I_S_model: pd.DataFrame,
        I_S_metrics: dict,
        D_S_metrics: dict,
        output_cols: list[str],
        device: torch.device,
    ):
        """
        Evaluate the trained model on selected inputs and denormalize the outputs.

        Inputs
        ------
        model : nn.Module
            Trained neural network model.
        I_S_model : pd.DataFrame
            Input DataFrame for model evaluation.
        I_S_metrics : dict
            Input normalization metrics.
        D_S_metrics : dict
            Output normalization metrics.
        output_cols : list[str]
            Output column names.
        device : torch.device
            Device for model evaluation.

        Returns
        -------
        D_S_pred : pd.DataFrame
            Denormalized model predictions.
        """
        I_S_model_norm = Normalization.scale_centered_defined_metrics(
            I_S_model, I_S_metrics
        )
        x = torch.tensor(I_S_model_norm.to_numpy(), dtype=torch.float32).to(device)

        with torch.no_grad():
            y_pred_norm = model(x)

        D_S_pred, _ = Denormalization.min_max_pfr(y_pred_norm, D_S_metrics, output_cols)
        D_S_pred = pd.DataFrame(D_S_pred.cpu().numpy(), columns=output_cols)
        return D_S_pred

    @staticmethod
    def error_metrics(
        df_true: pd.DataFrame,
        df_pred: pd.DataFrame,
        groups: dict[str, list[str]],
    ):
        """
        Compute MAE, RMSE, and R2 for grouped outputs.

        Inputs
        ------
        df_true : pd.DataFrame
            Reference output DataFrame.
        df_pred : pd.DataFrame
            Predicted output DataFrame.
        groups : dict[str, list[str]]
            Mapping of group name to column names.

        Returns
        -------
        pd.DataFrame
            Error metrics for each output group.
        """

        def calc_metrics(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0:
                return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
            return {"MAE": mae, "RMSE": rmse, "R2": r2}

        rows = []
        for name, cols in groups.items():
            m = calc_metrics(df_true[cols].to_numpy(), df_pred[cols].to_numpy())
            rows.append({"Output": name, **m})

        return pd.DataFrame(rows)


class Save:
    @staticmethod
    def excel(
        history: dict, foldername: str = "logs", filename: str = "training_history.xlsx"
    ):
        """
        Save the training history dictionary to a multi-sheet Excel file.

        Inputs
        ------
        history : dict
            Dictionary containing training-history items.
        foldername : str, optional, default="logs"
            Folder in which the Excel file is saved.
        filename : str, optional, default="training_history.xlsx"
        Filename.

        Returns
        -------
        None
            Writes XLSX file to `<foldername>/<filename>/`.
        """
        os.makedirs(foldername, exist_ok=True)
        filepath = os.path.join(foldername, filename)

        with pd.ExcelWriter(filepath) as writer:
            for key, value in history.items():
                if value is None:
                    continue
                arr = np.asarray(value)
                if arr.ndim == 1:
                    df = pd.DataFrame({key: value})
                elif arr.ndim == 2:
                    df = pd.DataFrame(
                        arr, columns=[f"{key}_{j+1}" for j in range(arr.shape[1])]
                    )
                else:
                    df = pd.DataFrame({key: pd.Series(value, dtype=object)})

                df.to_excel(writer, sheet_name=key[:31], index=False)
        print(f"Saved training history XLSX files to folder: {filepath}")

    @staticmethod
    def csv(history: dict, foldername: str = "logs", filename: str = "csv_files"):
        """
        Save the training history dictionary as multiple CSV files.

        Inputs
        ------
        history : dict
            Dictionary containing training-history items.
        foldername : str, optional, default="logs"
            Parent folder in which the CSV files are saved.
        filename : str, optional, default="csv_files"
            Filename.

        Returns
        -------
        None
            Writes one CSV file per history item to `<foldername>/<filename>/`.
        """
        foldername = os.path.join(foldername, filename)
        os.makedirs(foldername, exist_ok=True)
        for key, value in history.items():
            if value is None:
                continue

            arr = np.asarray(value)

            if arr.ndim == 1:
                df = pd.DataFrame({key: value})
            elif arr.ndim == 2:
                df = pd.DataFrame(
                    arr, columns=[f"{key}_{j+1}" for j in range(arr.shape[1])]
                )
            else:
                df = pd.DataFrame({key: pd.Series(value, dtype=object)})

            df.to_csv(os.path.join(foldername, f"{key}.csv"), index=False)

        print(f"Saved training history CSV files to folder: {foldername}")
