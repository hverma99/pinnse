import numpy as np
import pandas as pd
from scipy.stats import qmc
import shutil, uuid, pywintypes, tempfile
import win32com.client as win32, pythoncom, time
from tqdm import tqdm
from pathlib import Path

"""
Generate labeled input-output (I_S/D_S) datasets for the nonideal flash process.

Inputs
------
- Process specifications defining the flash process
- Admissible operating bounds for I_S
- Labeled dataset size used for sample generation

Outputs
-------
- I_S : sampled input dataset
- D_S : Aspen-generated output dataset corresponding to I_S
"""

components = ["O2", "CO2", "H2O", "C6H6", "C4H2O3"]
N_comp = len(components)
T_bds = [60, 90]  # Flash temperature bounds in C
P_bds = [1, 2]  # Flash pressure bounds in bar
F_bds = [450, 7450]  # Flash inlet total molar flowrate bounds in kmol/h
Tf_bds = [350, 450]  # Flash inlet feed temperature bounds in C
Pf_bds = [1, 2]  # Flash inlet feed pressure bounds in bar
N_D = 25000  # Labeled dataset size


def I_S_samples(seed=42):
    """
    Generate input samples for the flash process.

    Inputs
    ------
    seed : int, optional, default=42
        Random seed used to initialize the NumPy random number generator
        for reproducible sampling.

    Returns
    -------
    I_S : pd.DataFrame
        DataFrame containing the generated input samples with columns:
        - T_flash : flash temperature
        - P_flash : flash pressure
        - T_feed  : feed temperature
        - P_feed  : feed pressure
        - F       : total feed molar flowrate
        - Z_i     : feed molar fractions for each component

    Notes
    -----
    - Flash temperature, flash pressure, feed temperature, feed pressure,
      and total feed flowrate are sampled using Latin Hypercube Sampling
      within their prescribed bounds.
    - Feed molar compositions are sampled from a Dirichlet distribution
      so that the component fractions are nonnegative and sum to 1.
    - The returned DataFrame represents the sampled I_S used for data generation.
    """
    rng = np.random.default_rng(seed)
    # Flash temperature and pressure
    unit = qmc.LatinHypercube(d=2, rng=rng).random(N_D).astype(np.float32)
    T_flash = T_bds[0] + (T_bds[1] - T_bds[0]) * unit[:, 0]
    P_flash = P_bds[0] + (P_bds[1] - P_bds[0]) * unit[:, 1]

    # Total inlet molar flowrate
    unit_F = qmc.LatinHypercube(d=1, rng=rng).random(N_D).astype(np.float32)
    F = (F_bds[0] + (F_bds[1] - F_bds[0]) * unit_F).flatten()

    # Feed temperature and pressure
    unit = qmc.LatinHypercube(d=2, rng=rng).random(N_D).astype(np.float32)
    T_feed = Tf_bds[0] + (Tf_bds[1] - Tf_bds[0]) * unit[:, 0]
    P_feed = Pf_bds[0] + (Pf_bds[1] - Pf_bds[0]) * unit[:, 1]

    # Feed compositions
    comp_label = [f"Z_{i}" for i in components]
    Z = rng.dirichlet(alpha=np.ones(N_comp), size=N_D)

    I_S = pd.DataFrame(
        {
            "T_flash": T_flash,
            "P_flash": P_flash,
            "T_feed": T_feed,
            "P_feed": P_feed,
            "F": F,
            **dict(zip(comp_label, Z.T)),
        }
    )
    return I_S


def D_S_samples(I_S: pd.DataFrame):
    """
    Generate D_S for the flash process by running Aspen simulations
    for each input sample in I_S.

    Inputs
    ------
    I_S : pd.DataFrame
        DataFrame containing the sampled input conditions for the flash process.

    Returns
    -------
    D_S : pd.DataFrame
        DataFrame containing the simulated output samples with columns:
        - case_id   : index of the corresponding input case
        - L         : liquid outlet total flowrate
        - X_i       : liquid-phase molar fractions for each component
        - V         : vapor outlet total flowrate
        - Y_i       : vapor-phase molar fractions for each component
        - VF        : vapor fraction
        - Q_flash   : flash heat duty in kW

    Notes
    -----
    - A temporary Aspen backup file is created from the template `Flash.bkp`
      and used as the working simulation file.
    - Aspen Plus is accessed through COM automation using `win32com`.
    - For each input sample, the corresponding process conditions and feed
      composition are written into the Aspen model, and the simulation is run.
    - If a COM-related error occurs during execution, Aspen is reopened and
      the case is retried once.
    - Cases with failed runs or nonzero convergence status (`CVSTAT == 1`)
      are skipped and logged in `flash_fail_log.txt`.
    """

    base_dir = Path(__file__).resolve().parent.parent / "Aspen Simulations"
    template = base_dir / "Flash.bkp"

    if not template.exists():
        raise FileNotFoundError(f"Template file not found: {template}")

    case_dir = Path(tempfile.gettempdir()) / "aspen_temp_cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    workfile = case_dir / f"flash_{uuid.uuid4().hex}.bkp"
    shutil.copy2(template, workfile)

    fail_log = base_dir / "flash_fail_log.txt"

    pythoncom.CoInitialize()
    aspen = None

    def start_aspen():
        app = win32.DispatchEx("Apwn.Document")
        app.InitFromArchive2(str(workfile.resolve()))
        app.Visible = False
        try:
            app.SuppressDialogs = True
        except Exception:
            pass
        return app

    def reopen_aspen():
        nonlocal aspen
        try:
            if aspen is not None:
                aspen.Close()
        except Exception:
            pass
        aspen = None
        time.sleep(1.0)
        aspen = start_aspen()
        aspen.Engine.Reinit()

    nodes = {
        "F": r"\Data\Streams\FEED\Input\TOTFLOW\MIXED",
        "Z_frac": r"\Data\Streams\FEED\Input\FLOW\MIXED\{}",
        "T_feed": r"\Data\Streams\FEED\Input\TEMP\MIXED",
        "P_feed": r"\Data\Streams\FEED\Input\PRES\MIXED",
        "T_flash": r"\Data\Blocks\B1\Input\TEMP",
        "P_flash": r"\Data\Blocks\B1\Input\PRES",
        "L": r"\Data\Streams\LIQUID\Output\TOT_FLOW",
        "X_frac": r"\Data\Blocks\B1\Output\X\{}",
        "V": r"\Data\Streams\VAPOR\Output\TOT_FLOW",
        "Y_frac": r"\Data\Blocks\B1\Output\Y\{}",
        "Q_flash": r"\Data\Blocks\B1\Output\QNET",
        "vfrac": r"\Data\Blocks\B1\Output\B_VFRAC",
    }
    status_node = r"\Data\Results Summary\Run-Status\Output\CVSTAT"

    out_rows = []

    aspen = start_aspen()

    for case_id, row in tqdm(I_S.iterrows(), total=len(I_S)):
        feed_vec = [row[f"Z_{c}"] for c in components]

        aspen.Tree.FindNode(nodes["F"]).Value = row["F"]
        for z, comp in zip(feed_vec, components):
            aspen.Tree.FindNode(nodes["Z_frac"].format(comp)).Value = z

        aspen.Tree.FindNode(nodes["T_feed"]).Value = row["T_feed"]
        aspen.Tree.FindNode(nodes["P_feed"]).Value = row["P_feed"]
        aspen.Tree.FindNode(nodes["T_flash"]).Value = row["T_flash"]
        aspen.Tree.FindNode(nodes["P_flash"]).Value = row["P_flash"]

        try:
            aspen.Engine.Run2()
            while aspen.Engine.IsRunning:
                time.sleep(0.1)

        except pywintypes.com_error as e:
            reopen_aspen()
            try:
                aspen.Tree.FindNode(nodes["F"]).Value = row["F"]
                for z, comp in zip(feed_vec, components):
                    aspen.Tree.FindNode(nodes["Z_frac"].format(comp)).Value = z
                aspen.Tree.FindNode(nodes["T_feed"]).Value = row["T_feed"]
                aspen.Tree.FindNode(nodes["P_feed"]).Value = row["P_feed"]
                aspen.Tree.FindNode(nodes["T_flash"]).Value = row["T_flash"]
                aspen.Tree.FindNode(nodes["P_flash"]).Value = row["P_flash"]

                aspen.Engine.Run2()
                while aspen.Engine.IsRunning:
                    time.sleep(0.1)

            except pywintypes.com_error as e2:
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"Run failed at case={case_id}: {repr(e2)}\n")
                continue

        CVSTAT = aspen.Tree.FindNode(status_node).Value
        if CVSTAT == 1:
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"Nonzero CVSTAT at case={case_id}: {CVSTAT}\n")
            continue

        L = aspen.Tree.FindNode(nodes["L"]).Value
        X = [aspen.Tree.FindNode(nodes["X_frac"].format(c)).Value for c in components]
        V = aspen.Tree.FindNode(nodes["V"]).Value
        Y = [aspen.Tree.FindNode(nodes["Y_frac"].format(c)).Value for c in components]
        VF = aspen.Tree.FindNode(nodes["vfrac"]).Value
        Q = aspen.Tree.FindNode(nodes["Q_flash"]).Value * 4.184 / 1000  # kW

        out_rows.append([case_id, L, *X, V, *Y, VF, Q])

    D_S = pd.DataFrame(
        out_rows,
        columns=(
            ["case_id", "L"]
            + [f"X_{c}" for c in components]
            + ["V"]
            + [f"Y_{c}" for c in components]
            + ["VF", "Q"]
        ),
    )
    aspen.Close()
    return D_S


if __name__ == "__main__":
    """
    Generate and save the case-specific input and output datasets.
    """
    I_S = I_S_samples()
    D_S = D_S_samples(I_S)

    I_S_data = I_S[["T_flash", "P_flash", *(f"Z_{c}" for c in components)]]
    D_S_data = D_S[
        ["VF", *(f"Y_{c}" for c in components), *(f"X_{c}" for c in components)]
    ]

    I_S_data.to_excel("I_S_data.xlsx", index=False)
    D_S_data.to_excel("D_S_data.xlsx", index=False)
