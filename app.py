"""Chemical compound QSAR web app built with Streamlit and RDKit."""

from __future__ import annotations

import base64
import io
import math
import pickle
import re
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Fragments, Lipinski, MACCSkeys, QED, rdFMCS, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOT_TEMPLATE = "plotly_white"
APP_NAME = "ChemBlast"
DEVELOPER_NAME = "Ahmed G. Soliman"
DEVELOPER_PORTFOLIO = "https://sites.google.com/view/ahmed-g-soliman/home"
DEVELOPER_PROFILE = {
    "Developer": DEVELOPER_NAME,
    "Current role": "MEXT master's student at Kyutech, Japan, School of Life Science and Engineering",
    "Background": "Biotechnology Program, Faculty of Agriculture, Ain Shams University, Cairo, Egypt",
    "Experience": "Previous instructor at ACGEB in in-silico drug design and immune-informatics",
    "Scopus ID": "58569160700",
    "ResearcherID (WOS)": "ABE-8406-2021",
    "ORCID": "0000-0002-1122-3993",
    "Portfolio": DEVELOPER_PORTFOLIO,
}


st.set_page_config(
    page_title=APP_NAME,
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: "Manrope", "Segoe UI", sans-serif; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f3fbf8 0%, #edf7ff 100%);
  border-right: 1px solid #d9ebe9;
}
.chem-hero {
  border: 1px solid #d4e8e4;
  background:
    radial-gradient(circle at 8% 18%, rgba(0, 137, 123, .14), transparent 19rem),
    linear-gradient(135deg, #f8fffc 0%, #edf8f7 55%, #e9f3ff 100%);
  border-radius: 22px;
  padding: 1.25rem 1.45rem;
  margin-bottom: 1rem;
  box-shadow: 0 18px 54px rgba(26, 69, 75, .07);
}
.chem-hero h1 { margin: 0; color: #093b3d; font-size: 2rem; letter-spacing: -0.04em; }
.chem-hero p { margin: .4rem 0 0; color: #315c62; font-size: .98rem; }
.metric-note { color: #52727a; font-size: .86rem; }
.soft-box {
  border: 1px solid #d7e9e8;
  border-radius: 16px;
  padding: .9rem;
  background: rgba(255,255,255,.92);
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@dataclass
class DescriptorOptions:
    include_morgan: bool = True
    include_maccs: bool = False
    morgan_bits: int = 1024
    morgan_radius: int = 2
    include_full_rdkit: bool = True
    include_functional_groups: bool = True
    include_element_counts: bool = True
    include_3d: bool = True


def clean_smiles(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def descriptor_options_to_dict(options: Any) -> dict[str, Any]:
    """Convert old Streamlit session objects or dicts into pickle-safe options."""
    defaults = {field.name: field.default for field in fields(DescriptorOptions)}
    if isinstance(options, dict):
        source = options
    else:
        source = {name: getattr(options, name, default) for name, default in defaults.items()}
    return {name: source.get(name, default) for name, default in defaults.items()}


def normalize_descriptor_options(options: Any) -> DescriptorOptions:
    return DescriptorOptions(**descriptor_options_to_dict(options))


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol


def molecule_has_3d_coordinates(mol: Chem.Mol) -> bool:
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    for idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        if abs(pos.z) > 1e-3:
            return True
    return False


def mol_with_3d_from_smiles(smiles: str) -> tuple[Chem.Mol | None, str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if status != 0:
        return None, "3D embedding failed"
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        pass
    return mol, "ETKDG 3D conformer generated"


def read_sdf_bytes(file_bytes: bytes) -> pd.DataFrame:
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(file_bytes), sanitize=True, removeHs=False)
    rows: list[dict[str, Any]] = []
    for idx, mol in enumerate(supplier, start=1):
        if mol is None:
            continue
        props = {name: mol.GetProp(name) for name in mol.GetPropNames()}
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Molecule_{idx}"
        rows.append({"Name": name, "SMILES": Chem.MolToSmiles(mol), **props})
    return pd.DataFrame(rows)


def read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.read()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(raw))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))
    if suffix in {".sdf", ".sd"}:
        return read_sdf_bytes(raw)
    if suffix == ".txt":
        text = raw.decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return pd.DataFrame({"SMILES": lines, "Name": [f"Molecule_{i+1}" for i in range(len(lines))]})
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_manual_smiles(text: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"[\t,;]+", line, maxsplit=1)
        smiles = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else f"Manual_{idx}"
        rows.append({"Name": name, "SMILES": smiles})
    return pd.DataFrame(rows)


def guess_smiles_column(df: pd.DataFrame) -> str | None:
    preferred = ["SMILES", "smiles", "CanonicalSMILES", "canonical_smiles", "Structure", "Mol"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(30)
        if len(sample) and sample.map(lambda x: mol_from_smiles(x) is not None).mean() >= 0.6:
            return col
    return None


def guess_target_columns(df: pd.DataFrame, smiles_col: str | None) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col == smiles_col or col.lower() in {"name", "id", "compound", "compoundid"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique(dropna=True) <= 20:
            out.append(col)
    return out


def validate_molecules(df: pd.DataFrame, smiles_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        smiles = clean_smiles(row.get(smiles_col, ""))
        mol = mol_from_smiles(smiles)
        data = row.to_dict()
        if mol is None:
            data["ValidationIssue"] = "Invalid SMILES"
            invalid_rows.append(data)
            continue
        data["SMILES"] = Chem.MolToSmiles(mol)
        if "Name" not in data or not str(data.get("Name", "")).strip():
            data["Name"] = f"Mol_{idx + 1}"
        valid_rows.append(data)
    return pd.DataFrame(valid_rows), pd.DataFrame(invalid_rows)


def base_descriptors(mol: Chem.Mol) -> dict[str, float]:
    charge_values = []
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for atom in mol.GetAtoms():
            val = atom.GetProp("_GasteigerCharge")
            if val and val not in {"nan", "-nan", "inf", "-inf"}:
                charge_values.append(float(val))
    except Exception:
        charge_values = []

    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "HeavyAtomMolWt": float(Descriptors.HeavyAtomMolWt(mol)),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
        "AtomCount": float(mol.GetNumAtoms()),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "SaturatedRings": float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "AliphaticRings": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "FormalCharge": float(Chem.GetFormalCharge(mol)),
        "QED": float(QED.qed(mol)),
        "BertzCT": float(Descriptors.BertzCT(mol)),
        "BalabanJ": float(Descriptors.BalabanJ(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "Chi0": float(Descriptors.Chi0(mol)),
        "Chi1": float(Descriptors.Chi1(mol)),
        "Kappa1": float(Descriptors.Kappa1(mol)),
        "Kappa2": float(Descriptors.Kappa2(mol)),
        "Kappa3": float(Descriptors.Kappa3(mol)),
        "MaxPartialCharge": float(max(charge_values) if charge_values else 0.0),
        "MinPartialCharge": float(min(charge_values) if charge_values else 0.0),
    }


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def all_rdkit_descriptors(mol: Chem.Mol) -> dict[str, float]:
    descriptors: dict[str, float] = {}
    for name, func in Descriptors.descList:
        try:
            value = safe_float(func(mol))
        except Exception:
            value = None
        if value is not None:
            descriptors[f"RDKit_{name}"] = value
    return descriptors


def functional_group_descriptors(mol: Chem.Mol) -> dict[str, float]:
    groups: dict[str, float] = {}
    for name in dir(Fragments):
        if not name.startswith("fr_"):
            continue
        func = getattr(Fragments, name)
        if not callable(func):
            continue
        try:
            value = safe_float(func(mol))
        except Exception:
            value = None
        if value is not None:
            groups[f"FG_{name[3:]}"] = value
    return groups


def element_descriptors(mol: Chem.Mol) -> dict[str, float]:
    elements = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"]
    formula = rdMolDescriptors.CalcMolFormula(mol)
    atom_counts = {symbol: 0 for symbol in elements}
    hetero_atoms = 0
    heavy_atoms = max(1, mol.GetNumHeavyAtoms())
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_counts:
            atom_counts[symbol] += 1
        atom_counts["H"] += int(atom.GetTotalNumHs())
        if symbol not in {"C", "H"}:
            hetero_atoms += 1
    out = {"MolecularFormulaLength": float(len(formula)), "HeteroAtomCount": float(hetero_atoms), "HeteroAtomFraction": float(hetero_atoms / heavy_atoms)}
    for symbol, count in atom_counts.items():
        safe_symbol = symbol.replace("Cl", "Chlorine").replace("Br", "Bromine")
        out[f"Elem_{safe_symbol}_Count"] = float(count)
        out[f"Elem_{safe_symbol}_FractionHeavy"] = float(count / heavy_atoms)
    return out


def descriptors_3d_from_smiles(smiles: str) -> dict[str, float]:
    mol3d, status = mol_with_3d_from_smiles(smiles)
    out: dict[str, float] = {
        "3D_EmbedSuccess": 1.0 if mol3d is not None else 0.0,
        "3D_StatusCode": 1.0 if mol3d is not None and "generated" in status.lower() else 0.0,
    }
    descriptor_funcs = {
        "3D_Asphericity": getattr(rdMolDescriptors, "CalcAsphericity", None),
        "3D_Eccentricity": getattr(rdMolDescriptors, "CalcEccentricity", None),
        "3D_InertialShapeFactor": getattr(rdMolDescriptors, "CalcInertialShapeFactor", None),
        "3D_NPR1": getattr(rdMolDescriptors, "CalcNPR1", None),
        "3D_NPR2": getattr(rdMolDescriptors, "CalcNPR2", None),
        "3D_PMI1": getattr(rdMolDescriptors, "CalcPMI1", None),
        "3D_PMI2": getattr(rdMolDescriptors, "CalcPMI2", None),
        "3D_PMI3": getattr(rdMolDescriptors, "CalcPMI3", None),
        "3D_RadiusOfGyration": getattr(rdMolDescriptors, "CalcRadiusOfGyration", None),
        "3D_SpherocityIndex": getattr(rdMolDescriptors, "CalcSpherocityIndex", None),
        "3D_PBF": getattr(rdMolDescriptors, "CalcPBF", None),
    }
    for name in descriptor_funcs:
        out[name] = 0.0
    if mol3d is None:
        return out
    for name, func in descriptor_funcs.items():
        if func is None:
            continue
        try:
            value = safe_float(func(mol3d))
        except Exception:
            value = None
        out[name] = value if value is not None else 0.0
    return out


def plot_molecule_3d(smiles: str) -> tuple[go.Figure, str]:
    mol3d, status = mol_with_3d_from_smiles(smiles)
    if mol3d is None:
        return px.scatter_3d(title=f"3D structure unavailable: {status}", template=PLOT_TEMPLATE), status
    conf = mol3d.GetConformer()
    atoms = []
    for atom in mol3d.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "Atom": atom.GetSymbol(),
                "Index": atom.GetIdx(),
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "AtomicNum": atom.GetAtomicNum(),
            }
        )
    atom_df = pd.DataFrame(atoms)
    fig = go.Figure()
    for bond in mol3d.GetBonds():
        begin = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conf.GetAtomPosition(bond.GetEndAtomIdx())
        fig.add_trace(
            go.Scatter3d(
                x=[begin.x, end.x],
                y=[begin.y, end.y],
                z=[begin.z, end.z],
                mode="lines",
                line=dict(color="#7a8b8f", width=5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=atom_df["x"],
            y=atom_df["y"],
            z=atom_df["z"],
            mode="markers+text",
            text=atom_df["Atom"],
            textposition="top center",
            marker=dict(size=np.clip(atom_df["AtomicNum"] * 1.8, 6, 20), color=atom_df["AtomicNum"], colorscale="Viridis", line=dict(width=1, color="#102a2d")),
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
            name="Atoms",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="3D conformer view (ETKDG/MMFF or UFF optimized)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig, status


def bitvect_to_dict(prefix: str, bitvect: Any, n_bits: int | None = None) -> dict[str, float]:
    if n_bits is None:
        n_bits = bitvect.GetNumBits()
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(arr)}


def calculate_descriptors(df: pd.DataFrame, options: DescriptorOptions) -> pd.DataFrame:
    options = normalize_descriptor_options(options)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        smiles = clean_smiles(row["SMILES"])
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        data = row.to_dict()
        data["SMILES"] = Chem.MolToSmiles(mol)
        data.update(base_descriptors(mol))
        if options.include_element_counts:
            data.update(element_descriptors(mol))
        if options.include_full_rdkit:
            data.update(all_rdkit_descriptors(mol))
        if options.include_functional_groups:
            data.update(functional_group_descriptors(mol))
        if options.include_3d:
            data.update(descriptors_3d_from_smiles(smiles))
        if options.include_morgan:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=options.morgan_radius, fpSize=options.morgan_bits)
            fp = generator.GetFingerprint(mol)
            data.update(bitvect_to_dict("Morgan", fp, options.morgan_bits))
        if options.include_maccs:
            fp = MACCSkeys.GenMACCSKeys(mol)
            data.update(bitvect_to_dict("MACCS", fp))
        rows.append(data)
    return pd.DataFrame(rows)


def infer_descriptor_options(features: list[str]) -> DescriptorOptions:
    """Recover descriptor settings from a trained model feature list."""
    morgan_indices: list[int] = []
    for feature in features:
        if feature.startswith("Morgan_"):
            try:
                morgan_indices.append(int(feature.split("_", 1)[1]))
            except ValueError:
                continue
    return DescriptorOptions(
        include_morgan=bool(morgan_indices),
        include_maccs=any(feature.startswith("MACCS_") for feature in features),
        morgan_bits=(max(morgan_indices) + 1) if morgan_indices else 1024,
        include_full_rdkit=any(feature.startswith("RDKit_") for feature in features),
        include_functional_groups=any(feature.startswith("FG_") for feature in features),
        include_element_counts=any(feature.startswith("Elem_") for feature in features),
        include_3d=any(feature.startswith("3D_") for feature in features),
    )


def descriptor_columns(df: pd.DataFrame) -> list[str]:
    protected = {"Name", "SMILES"}
    return [col for col in df.columns if col not in protected and pd.api.types.is_numeric_dtype(df[col])]


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == "Core descriptors + functional groups":
        return [f for f in features if not f.startswith(("Morgan_", "MACCS_", "RDKit_"))]
    if mode == "All descriptors except fingerprints":
        return [f for f in features if not f.startswith(("Morgan_", "MACCS_"))]
    return features


def nonzero_functional_group_table(df: pd.DataFrame) -> pd.DataFrame:
    fg_cols = [col for col in df.columns if col.startswith("FG_") and pd.api.types.is_numeric_dtype(df[col]) and df[col].sum() > 0]
    if not fg_cols:
        return pd.DataFrame()
    ordered = df[fg_cols].sum().sort_values(ascending=False).index.tolist()
    return df[["Name", "SMILES"] + ordered]


def drug_likeness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        mw = float(row.get("MolWt", np.nan))
        logp = float(row.get("MolLogP", np.nan))
        hbd = float(row.get("HBD", np.nan))
        hba = float(row.get("HBA", np.nan))
        tpsa = float(row.get("TPSA", np.nan))
        rb = float(row.get("RotatableBonds", np.nan))
        lipinski = int(mw <= 500) + int(logp <= 5) + int(hbd <= 5) + int(hba <= 10)
        veber = int(tpsa <= 140) + int(rb <= 10)
        ghose = int(160 <= mw <= 480) + int(-0.4 <= logp <= 5.6) + int(20 <= row.get("AtomCount", 0) <= 70) + int(40 <= row.get("MolMR", 0) <= 130)
        rows.append({
            "Name": row.get("Name", ""),
            "SMILES": row.get("SMILES", ""),
            "LipinskiPassCount": lipinski,
            "LipinskiPass": lipinski == 4,
            "VeberPassCount": veber,
            "VeberPass": veber == 2,
            "GhosePassCount": ghose,
            "GhosePass": ghose == 4,
            "DrugLikeScore": round((lipinski / 4 + veber / 2 + ghose / 4) / 3 * 100, 2),
        })
    return pd.DataFrame(rows)


def model_catalog(task_type: str, n_classes: int = 2) -> dict[str, Any]:
    if task_type == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001, max_iter=10000),
            "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
            "Random Forest": RandomForestRegressor(n_estimators=350, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR (RBF)": SVR(C=10.0, gamma="scale"),
            "kNN": KNeighborsRegressor(n_neighbors=3),
            "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42, early_stopping=True),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=4000),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=450, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(C=5.0, gamma="scale", probability=True, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=3),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42, early_stopping=True),
    }


def infer_direction(target: str) -> str:
    lower = target.lower()
    lower_is_better = ["ic50", "ec50", "ki", "kd", "mic", "docking", "vina", "bindingenergy", "delta_g"]
    if any(x in lower.replace(" ", "").replace("_", "") for x in lower_is_better):
        return "lower"
    return "higher"


def positive_lower_better_score(values: Any) -> pd.Series:
    """Convert lower-is-better predictions into a positive, monotonic design score."""
    raw = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float)
    score = pd.Series(np.nan, index=raw.index, dtype=float)
    valid = raw.dropna()
    if valid.empty:
        return score

    vmin = float(valid.min())
    vmax = float(valid.max())
    if len(valid) > 1 and not math.isclose(vmin, vmax):
        score.loc[valid.index] = 100.0 * (vmax - valid) / (vmax - vmin)
    else:
        single = valid.copy()
        # Docking-like values are often negative; positive assay units are better represented as inverse scores.
        score.loc[single.index] = np.where(single <= 0, -single, 100.0 / (1.0 + single))
    return score.clip(lower=0.0)


def normalized_score(values: Any, higher_is_better: bool = True) -> pd.Series:
    raw = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float)
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    valid = raw.dropna()
    if valid.empty:
        return out
    vmin = float(valid.min())
    vmax = float(valid.max())
    if math.isclose(vmin, vmax):
        out.loc[valid.index] = 100.0
    elif higher_is_better:
        out.loc[valid.index] = 100.0 * (valid - vmin) / (vmax - vmin)
    else:
        out.loc[valid.index] = 100.0 * (vmax - valid) / (vmax - vmin)
    return out.clip(lower=0.0, upper=100.0)


def prepare_xy(df: pd.DataFrame, features: list[str], target: str, task_type: str) -> tuple[pd.DataFrame, pd.Series, LabelEncoder | None]:
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y_raw = df[target]
    if task_type == "Regression":
        y = pd.to_numeric(y_raw, errors="coerce")
        mask = y.notna()
        return X.loc[mask], y.loc[mask], None
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y_raw.astype(str)), index=y_raw.index, name=target)
    return X, y, encoder


def evaluate_models(df: pd.DataFrame, features: list[str], target: str, task_type: str, selected_models: list[str], test_size: float, cv_folds: int) -> dict[str, Any]:
    X, y, encoder = prepare_xy(df, features, target, task_type)
    if len(X) < 5:
        raise ValueError("At least 5 valid rows are recommended for model training.")
    if task_type != "Regression" and (y.nunique() < 2 or y.value_counts().min() < 2):
        raise ValueError("Classification needs at least two classes and at least two molecules per class.")

    stratify = y if task_type != "Regression" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    catalog = model_catalog(task_type, n_classes=y.nunique())
    rows = []
    bundles = {}

    for name in selected_models:
        estimator = clone(catalog[name])
        score_func = f_regression if task_type == "Regression" else f_classif
        selector_percentile = 25 if len(features) > 80 else 100
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("selector", SelectPercentile(score_func=score_func, percentile=selector_percentile)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        if task_type == "Regression":
            metrics = {
                "R2": float(r2_score(y_test, pred)),
                "RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
                "MAE": float(mean_absolute_error(y_test, pred)),
            }
            n_splits = min(max(2, cv_folds), max(2, len(X) // 2))
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=42)
            cv_score = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            cv_rmse = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
            metrics["CV_R2_mean"] = float(np.nanmean(cv_score))
            metrics["CV_R2_std"] = float(np.nanstd(cv_score))
            metrics["CV_RMSE_mean"] = float(np.nanmean(cv_rmse))
            primary = metrics["CV_R2_mean"] if math.isfinite(metrics["CV_R2_mean"]) else metrics["R2"]
        else:
            proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
            average = "binary" if y.nunique() == 2 else "weighted"
            metrics = {
                "Accuracy": float(accuracy_score(y_test, pred)),
                "Precision": float(precision_score(y_test, pred, average=average, zero_division=0)),
                "Recall": float(recall_score(y_test, pred, average=average, zero_division=0)),
                "F1": float(f1_score(y_test, pred, average=average, zero_division=0)),
            }
            if proba is not None and y.nunique() == 2:
                metrics["ROC_AUC"] = float(roc_auc_score(y_test, proba[:, 1]))
            cv = StratifiedKFold(n_splits=max(2, min(cv_folds, y.value_counts().min())), shuffle=True, random_state=42) if y.value_counts().min() >= 2 else None
            if cv is not None:
                cv_score = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted")
                metrics["CV_F1_mean"] = float(np.mean(cv_score))
            primary = metrics.get("CV_F1_mean", metrics["F1"])

        domain_features = [col for col in features if not col.startswith(("Morgan_", "MACCS_"))]
        fit_quality = max(0.0, min(100.0, 100.0 * primary)) if math.isfinite(float(primary)) else 0.0
        row = {"Model": name, **metrics, "SelectionScore": float(primary), "FitQuality_0_100": fit_quality}
        rows.append(row)
        final_pipe = clone(pipe)
        final_pipe.fit(X, y)
        bundles[name] = {
            "pipeline": final_pipe,
            "model_name": name,
            "features": features,
            "target": target,
            "task_type": task_type,
            "label_encoder": encoder,
            "y_test": y_test,
            "y_pred": pred,
            "descriptor_options": descriptor_options_to_dict(infer_descriptor_options(features)),
            "domain_features": domain_features,
            "domain_min": X[domain_features].min(numeric_only=True).to_dict(),
            "domain_max": X[domain_features].max(numeric_only=True).to_dict(),
        }

    leaderboard = pd.DataFrame(rows)
    sort_metric = "SelectionScore"
    leaderboard = leaderboard.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    best_name = leaderboard.loc[0, "Model"]
    return {"leaderboard": leaderboard, "bundles": bundles, "best_name": best_name, "best": bundles[best_name]}


def model_feature_importance(bundle: dict[str, Any]) -> pd.DataFrame:
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    features = np.array(bundle["features"])
    support = pipe.named_steps["variance"].get_support()
    used_features = features[support]
    selector = pipe.named_steps.get("selector")
    if selector is not None and hasattr(selector, "get_support"):
        try:
            used_features = used_features[selector.get_support()]
        except Exception:
            pass
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        if len(values) != len(used_features):
            values = np.mean(np.abs(model.coef_), axis=0)
    else:
        return pd.DataFrame()
    out = pd.DataFrame({"Feature": used_features[: len(values)], "Importance": np.abs(values)})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def model_summary(bundle: dict[str, Any], model_name: str | None = None) -> pd.DataFrame:
    model = bundle["pipeline"].named_steps["model"]
    name = model_name or model.__class__.__name__
    nonlinear_names = {"SVR", "SVC", "KNeighborsRegressor", "KNeighborsClassifier", "RandomForestRegressor", "RandomForestClassifier", "ExtraTreesRegressor", "ExtraTreesClassifier", "GradientBoostingRegressor", "GradientBoostingClassifier", "MLPRegressor", "MLPClassifier"}
    family = "Nonlinear" if model.__class__.__name__ in nonlinear_names else "Linear / probabilistic"
    rows = [
        ("Selected model", name),
        ("Model class", model.__class__.__name__),
        ("Model family", family),
        ("QSAR task", bundle["task_type"]),
        ("Target", bundle["target"]),
        ("Input features", len(bundle["features"])),
        ("Descriptor engine", "RDKit descriptors + optional Morgan/MACCS fingerprints"),
    ]
    if model.__class__.__name__ in {"SVR", "SVC"}:
        rows.append(("Nonlinear behavior", "RBF kernel maps molecules into a nonlinear similarity space."))
    if model.__class__.__name__.startswith("KNeighbors"):
        rows.append(("Nonlinear behavior", "Prediction is based on nearest molecules in descriptor/fingerprint space."))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def applicability_domain(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    domain_features = bundle.get("domain_features", [])
    min_map = bundle.get("domain_min", {})
    max_map = bundle.get("domain_max", {})
    rows = []
    for _, row in df.iterrows():
        checked = 0
        inside = 0
        outside_features = []
        for feature in domain_features:
            if feature not in row or feature not in min_map or feature not in max_map:
                continue
            value = row.get(feature)
            if pd.isna(value):
                continue
            checked += 1
            if float(min_map[feature]) <= float(value) <= float(max_map[feature]):
                inside += 1
            elif len(outside_features) < 8:
                outside_features.append(feature)
        score = round(100 * inside / checked, 1) if checked else np.nan
        rows.append({
            "Name": row.get("Name", ""),
            "ApplicabilityDomain_%": score,
            "OutsideTrainingRange": ", ".join(outside_features) if outside_features else "None detected",
        })
    return pd.DataFrame(rows)


def predict_new(bundle: dict[str, Any], df: pd.DataFrame, direction: str) -> pd.DataFrame:
    X = df[bundle["features"]].replace([np.inf, -np.inf], np.nan)
    pred = bundle["pipeline"].predict(X)
    out = df[["Name", "SMILES"]].copy()
    if bundle["task_type"] == "Regression":
        raw_pred = pd.Series(pd.to_numeric(pred, errors="coerce"), index=out.index, dtype=float)
        if direction == "lower":
            optimized = positive_lower_better_score(raw_pred)
            out["Prediction"] = optimized
            out["RawModelPrediction"] = raw_pred
            out["PredictionMeaning"] = (
                "Positive optimized score; higher is better. "
                "Raw lower-is-better model output is stored in RawModelPrediction."
            )
            out["RankingScore"] = optimized
            out["NormalizedScore_0_100"] = normalized_score(raw_pred, higher_is_better=False)
        else:
            out["Prediction"] = raw_pred
            out["PredictionMeaning"] = "Predicted target value; higher is better."
            out["RankingScore"] = raw_pred
            out["NormalizedScore_0_100"] = normalized_score(raw_pred, higher_is_better=True)
    else:
        encoder = bundle["label_encoder"]
        labels = encoder.inverse_transform(pred.astype(int)) if encoder is not None else pred
        out["PredictedClass"] = labels
        if hasattr(bundle["pipeline"], "predict_proba"):
            proba = bundle["pipeline"].predict_proba(X)
            out["MaxProbability"] = np.max(proba, axis=1)
            out["RankingScore"] = out["MaxProbability"]
        else:
            out["RankingScore"] = 0.0
    return out.sort_values("RankingScore", ascending=False).reset_index(drop=True)


def mol_png_data_uri(smiles: str, size: tuple[int, int] = (220, 170)) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        from rdkit.Chem import Draw

        img = Draw.MolToImage(mol, size=size)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        # Some cloud builds can run RDKit descriptors but fail on optional drawing backends.
        return ""


def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> Any:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprint(mol)


def scaffold_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def mcs_alignment_summary(reference_smiles: str, candidate_smiles: str) -> dict[str, Any]:
    ref = mol_from_smiles(reference_smiles)
    cand = mol_from_smiles(candidate_smiles)
    if ref is None or cand is None:
        return {"MCSAtoms": 0, "MCSAtomFraction": 0.0, "MCSSmarts": ""}
    try:
        result = rdFMCS.FindMCS([ref, cand], timeout=3, ringMatchesRingOnly=True, completeRingsOnly=True)
        if result.canceled or not result.smartsString:
            return {"MCSAtoms": 0, "MCSAtomFraction": 0.0, "MCSSmarts": ""}
        mcs_mol = Chem.MolFromSmarts(result.smartsString)
        atom_count = int(mcs_mol.GetNumAtoms()) if mcs_mol is not None else int(result.numAtoms)
        denominator = max(1, min(ref.GetNumHeavyAtoms(), cand.GetNumHeavyAtoms()))
        return {
            "MCSAtoms": atom_count,
            "MCSAtomFraction": round(atom_count / denominator, 3),
            "MCSSmarts": result.smartsString,
        }
    except Exception:
        return {"MCSAtoms": 0, "MCSAtomFraction": 0.0, "MCSSmarts": ""}


def similarity_to_reference(df: pd.DataFrame, reference_row: pd.Series) -> pd.DataFrame:
    ref_mol = mol_from_smiles(str(reference_row["SMILES"]))
    if ref_mol is None:
        return pd.DataFrame()
    ref_fp = morgan_fingerprint(ref_mol)
    ref_scaffold = scaffold_smiles(str(reference_row["SMILES"]))
    rows = []
    for _, row in df.iterrows():
        mol = mol_from_smiles(str(row["SMILES"]))
        if mol is None:
            continue
        fp = morgan_fingerprint(mol)
        tanimoto = float(DataStructs.TanimotoSimilarity(ref_fp, fp))
        candidate_scaffold = scaffold_smiles(str(row["SMILES"]))
        mcs = mcs_alignment_summary(str(reference_row["SMILES"]), str(row["SMILES"]))
        rows.append({
            "Name": row.get("Name", ""),
            "SMILES": row.get("SMILES", ""),
            "MorganTanimotoToReference": round(tanimoto, 3),
            "MCSAtomFraction": mcs["MCSAtomFraction"],
            "MCSAtoms": mcs["MCSAtoms"],
            "SameMurckoScaffold": bool(ref_scaffold and candidate_scaffold and ref_scaffold == candidate_scaffold),
            "CandidateScaffold": candidate_scaffold,
            "MCSSmarts": mcs["MCSSmarts"],
        })
    return pd.DataFrame(rows).sort_values(["MorganTanimotoToReference", "MCSAtomFraction"], ascending=False).reset_index(drop=True)


def pairwise_similarity_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
    names = df.get("Name", pd.Series(range(len(df)))).astype(str).tolist()
    fps = []
    valid_names = []
    for name, smiles in zip(names, df["SMILES"].astype(str)):
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        fps.append(morgan_fingerprint(mol))
        valid_names.append(name)
    if not fps:
        return pd.DataFrame(), px.scatter(title="No valid molecules for similarity", template=PLOT_TEMPLATE)
    matrix = np.zeros((len(fps), len(fps)))
    for i, fp_i in enumerate(fps):
        for j, fp_j in enumerate(fps):
            matrix[i, j] = DataStructs.TanimotoSimilarity(fp_i, fp_j)
    sim_df = pd.DataFrame(matrix, index=valid_names, columns=valid_names)
    fig = px.imshow(sim_df, zmin=0, zmax=1, color_continuous_scale="Viridis", template=PLOT_TEMPLATE, title="Pairwise Morgan Similarity Matrix")
    return sim_df, fig


def best_reference_row(df: pd.DataFrame, target: str, direction: str, active_class: str | None = None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(df[target]):
        numeric = pd.to_numeric(df[target], errors="coerce")
        idx = numeric.idxmin() if direction == "lower" else numeric.idxmax()
        return df.loc[idx]
    subset = df[df[target].astype(str) == str(active_class)] if active_class else df
    if subset.empty:
        subset = df
    if "DrugLikeScore" in subset.columns:
        return subset.sort_values("DrugLikeScore", ascending=False).iloc[0]
    return subset.iloc[0]


def high_activity_mask(df: pd.DataFrame, target: str, direction: str, active_class: str | None = None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(df[target]):
        values = pd.to_numeric(df[target], errors="coerce")
        threshold = values.quantile(0.75 if direction == "higher" else 0.25)
        return values >= threshold if direction == "higher" else values <= threshold
    return df[target].astype(str) == str(active_class)


def activity_design_profile(df: pd.DataFrame, target: str, direction: str, active_class: str | None = None) -> pd.DataFrame:
    mask = high_activity_mask(df, target, direction, active_class)
    top_df = df[mask].copy()
    if top_df.empty:
        return pd.DataFrame()
    key_descriptors = [
        "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds", "AromaticRings",
        "FractionCSP3", "QED", "BertzCT", "HeteroAtomCount", "FormalCharge",
    ]
    rows = []
    for descriptor in key_descriptors:
        if descriptor not in top_df.columns:
            continue
        values = pd.to_numeric(top_df[descriptor], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "Criterion": descriptor,
            "RecommendedRange": f"{values.quantile(0.10):.2f} to {values.quantile(0.90):.2f}",
            "MedianInBestCompounds": round(float(values.median()), 3),
            "Reason": "Observed range among the best-activity compounds in this dataset.",
        })
    return pd.DataFrame(rows)


def functional_group_activity_suggestions(df: pd.DataFrame, target: str, direction: str, active_class: str | None = None) -> pd.DataFrame:
    fg_cols = [col for col in df.columns if col.startswith("FG_") and pd.api.types.is_numeric_dtype(df[col]) and df[col].sum() > 0]
    if not fg_cols:
        return pd.DataFrame()
    mask = high_activity_mask(df, target, direction, active_class)
    high = df[mask]
    low = df[~mask]
    if high.empty or low.empty:
        return pd.DataFrame()
    rows = []
    for col in fg_cols:
        high_mean = float(pd.to_numeric(high[col], errors="coerce").fillna(0).mean())
        low_mean = float(pd.to_numeric(low[col], errors="coerce").fillna(0).mean())
        delta = high_mean - low_mean
        if abs(delta) < 0.05:
            continue
        rows.append({
            "FunctionalGroup": col.replace("FG_", ""),
            "MeanInBestCompounds": round(high_mean, 3),
            "MeanInOtherCompounds": round(low_mean, 3),
            "EnrichmentDelta": round(delta, 3),
            "DesignSuggestion": "Consider retaining/adding this motif" if delta > 0 else "Consider reducing/avoiding this motif",
        })
    return pd.DataFrame(rows).sort_values("EnrichmentDelta", key=np.abs, ascending=False).reset_index(drop=True)


def descriptor_activity_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if not pd.api.types.is_numeric_dtype(df[target]):
        return pd.DataFrame()
    numeric_target = pd.to_numeric(df[target], errors="coerce")
    rows = []
    for col in descriptor_columns(df):
        if col == target or col.startswith(("Morgan_", "MACCS_")):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.nunique(dropna=True) < 3:
            continue
        corr = values.corr(numeric_target, method="spearman")
        if pd.notna(corr):
            rows.append({"Descriptor": col, "SpearmanCorrelation": round(float(corr), 3)})
    return pd.DataFrame(rows).sort_values("SpearmanCorrelation", key=np.abs, ascending=False).reset_index(drop=True)


def plot_descriptor_hist(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    if not cols:
        return px.scatter(title="No numeric descriptor columns available", template=PLOT_TEMPLATE)
    top = df[cols].var(numeric_only=True).sort_values(ascending=False).head(12).index.tolist()
    if not top:
        return px.scatter(title="No varying descriptors available", template=PLOT_TEMPLATE)
    long = df[top].melt(var_name="Descriptor", value_name="Value")
    return px.histogram(long, x="Value", color="Descriptor", facet_col="Descriptor", facet_col_wrap=4, template=PLOT_TEMPLATE, title="Descriptor Distributions")


def plot_corr(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    if not cols:
        return px.scatter(title="No numeric descriptor columns available", template=PLOT_TEMPLATE)
    selected = df[cols].var(numeric_only=True).sort_values(ascending=False).head(50).index.tolist()
    if len(selected) < 2:
        return px.scatter(title="At least two varying descriptors are needed for correlation", template=PLOT_TEMPLATE)
    corr = df[selected].corr()
    return px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=PLOT_TEMPLATE, title="Descriptor Correlation Heatmap")


def plot_functional_group_heatmap(df: pd.DataFrame) -> go.Figure:
    fg_cols = [col for col in df.columns if col.startswith("FG_") and pd.api.types.is_numeric_dtype(df[col]) and df[col].sum() > 0]
    if not fg_cols:
        return px.scatter(title="No functional groups detected", template=PLOT_TEMPLATE)
    top = df[fg_cols].sum().sort_values(ascending=False).head(30).index.tolist()
    matrix = df[top].copy()
    matrix.index = df.get("Name", pd.Series(range(len(df)))).astype(str)
    return px.imshow(matrix, labels={"x": "Functional group", "y": "Compound", "color": "Count"}, template=PLOT_TEMPLATE, title="Functional Group Count Heatmap")


def pca_plot(df: pd.DataFrame, features: list[str], color_col: str | None = None) -> tuple[go.Figure, pd.DataFrame]:
    valid_features = [feature for feature in features if feature in df.columns]
    if len(valid_features) < 2 or len(df) < 2:
        fig = px.scatter(title="PCA needs at least two molecules and two features", template=PLOT_TEMPLATE)
        return fig, pd.DataFrame()
    X_df = df[valid_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_df = X_df.dropna(axis=1, how="all")
    X_df = X_df.loc[:, X_df.nunique(dropna=True) > 1]
    valid_features = X_df.columns.tolist()
    if len(valid_features) < 2:
        fig = px.scatter(title="PCA needs at least two non-constant descriptor columns", template=PLOT_TEMPLATE)
        return fig, pd.DataFrame()
    X = SimpleImputer(strategy="median").fit_transform(X_df)
    X = RobustScaler().fit_transform(X)
    n_components = min(2, X.shape[0], X.shape[1])
    if n_components < 2:
        fig = px.scatter(title="PCA needs at least two samples and two usable features", template=PLOT_TEMPLATE)
        return fig, pd.DataFrame()
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(X)
    pca_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "Name": df.get("Name", pd.Series(range(len(df)))).astype(str)})
    if color_col and color_col in df.columns:
        pca_df[color_col] = df[color_col].values
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_col if color_col in pca_df.columns else None,
        hover_name="Name",
        template=PLOT_TEMPLATE,
        title=f"Chemical Space PCA (PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)",
    )
    loadings = pd.DataFrame({"Feature": valid_features, "PC1_Loading": pca.components_[0], "PC2_Loading": pca.components_[1]})
    return fig, loadings.sort_values("PC1_Loading", key=np.abs, ascending=False)


def dataframe_download(df: pd.DataFrame, file_name: str, key_prefix: str = "main") -> None:
    st.download_button(
        f"Download {file_name}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=f"download_{key_prefix}_{file_name}",
        use_container_width=True,
    )


def pickle_model_bundle(bundle: dict[str, Any]) -> bytes:
    safe_bundle = {key: value for key, value in bundle.items() if key not in {"y_test", "y_pred"}}
    if "descriptor_options" in safe_bundle:
        safe_bundle["descriptor_options"] = descriptor_options_to_dict(safe_bundle["descriptor_options"])
    return pickle.dumps(safe_bundle)


def html_table(df: pd.DataFrame | None, title: str, max_rows: int = 30) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return f"<h2>{title}</h2><p>No data available.</p>"
    return f"<h2>{title}</h2>{df.head(max_rows).to_html(index=False, escape=True)}"


def generate_html_report() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    desc = st.session_state.desc_df
    training = st.session_state.training
    bundle = st.session_state.selected_bundle
    pred = st.session_state.predictions
    leaderboard = training.get("leaderboard") if isinstance(training, dict) else None
    summary = model_summary(bundle, bundle.get("model_name")) if isinstance(bundle, dict) else None
    criteria = drug_likeness(desc) if isinstance(desc, pd.DataFrame) and not desc.empty else None
    body = "\n".join([
        html_table(summary, "Model Summary"),
        html_table(leaderboard, "Model Leaderboard"),
        html_table(criteria, "Drug-likeness Criteria"),
        html_table(pred, "Prediction Results"),
    ])
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>ChemBlast Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #153235; margin: 42px; line-height: 1.5; }}
    h1 {{ color: #073b3d; }}
    h2 {{ margin-top: 30px; color: #0c5457; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }}
    th, td {{ border: 1px solid #cfe2e1; padding: 8px; text-align: left; }}
    th {{ background: #edf8f7; }}
  </style>
</head>
<body>
  <h1>ChemBlast Report</h1>
  <p>Generated: {timestamp}</p>
  <p>This report summarizes RDKit 2D/3D descriptor calculation, model comparison, drug-likeness checks, and prediction outputs created in ChemBlast.</p>
  {body}
</body>
</html>"""


def init_state() -> None:
    defaults = {
        "raw_df": None,
        "valid_df": None,
        "invalid_df": None,
        "desc_df": None,
        "features": [],
        "target_candidates": [],
        "descriptor_options": None,
        "training": None,
        "selected_bundle": None,
        "predictions": None,
        "prediction_descriptors": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_home() -> None:
    st.markdown(
        """
<div class="chem-hero">
  <h1>ChemBlast</h1>
  <p>Developer-built chemical QSAR platform for small molecules: SMILES/SDF input, 2D and 3D RDKit descriptors, fingerprints, ML models, visualizations, and browser-based prediction.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Input", "SMILES / SDF / CSV")
    c2.metric("Descriptors", "2D + 3D RDKit")
    c3.metric("Models", "Regression + Classification")
    st.caption(f"Developed by {DEVELOPER_NAME}.")
    st.info("Start with the example dataset, or upload your own molecules with a target column such as IC50, pIC50, DockingScore, LogS, Toxicity, or ActivityClass. The built-in Demo_pIC50 endpoint is synthetic and is included only to test the QSAR workflow.")


def render_input() -> None:
    st.subheader("Input Molecules")
    left, right = st.columns([1, 1])
    with left:
        if st.button("Load Example Dataset", use_container_width=True):
            st.session_state.raw_df = pd.read_csv(DATA_DIR / "example_compounds.csv")
            st.success("Example compounds loaded.")
        uploaded = st.file_uploader("Upload CSV, Excel, TXT, or SDF", type=["csv", "xlsx", "xls", "txt", "sdf", "sd"])
        if uploaded is not None:
            try:
                st.session_state.raw_df = read_uploaded_file(uploaded)
                st.success(f"Loaded {len(st.session_state.raw_df)} rows from {uploaded.name}.")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
    with right:
        manual = st.text_area(
            "Paste SMILES, one per line. Optionally add a name after comma or tab.",
            height=180,
            placeholder="CCO, ethanol\nCC(=O)OC1=CC=CC=C1C(=O)O, aspirin",
        )
        if st.button("Use Manual SMILES", use_container_width=True):
            st.session_state.raw_df = parse_manual_smiles(manual)
            st.success("Manual molecules loaded.")

    raw = st.session_state.raw_df
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        st.write("Preview")
        st.dataframe(raw.head(30), use_container_width=True)
        smiles_col = guess_smiles_column(raw)
        if smiles_col is None:
            st.error("No SMILES-like column was detected.")
            return
        smiles_col = st.selectbox("SMILES column", raw.columns.tolist(), index=raw.columns.tolist().index(smiles_col))
        valid, invalid = validate_molecules(raw, smiles_col)
        st.session_state.valid_df = valid
        st.session_state.invalid_df = invalid
        st.session_state.target_candidates = guess_target_columns(valid, "SMILES")
        st.success(f"Valid molecules: {len(valid)}")
        if not invalid.empty:
            st.warning(f"Invalid molecules: {len(invalid)}")
            st.dataframe(invalid.head(20), use_container_width=True)


def render_descriptors() -> None:
    st.subheader("Descriptors")
    valid = st.session_state.valid_df
    if not isinstance(valid, pd.DataFrame) or valid.empty:
        st.warning("Load valid molecules first.")
        return
    c1, c2, c3 = st.columns(3)
    include_morgan = c1.checkbox("Morgan fingerprints", value=True)
    include_maccs = c2.checkbox("MACCS keys", value=False)
    morgan_bits = c3.selectbox("Morgan bits", [256, 512, 1024, 2048], index=2)
    with st.expander("Descriptor families", expanded=True):
        d1, d2, d3, d4 = st.columns(4)
        include_full_rdkit = d1.checkbox("All RDKit descriptors", value=True, help="Adds the complete numeric RDKit descriptor list where available.")
        include_functional_groups = d2.checkbox("Functional group counts", value=True, help="Adds RDKit fragment counters such as amide, ester, phenol, carboxylic acid, halogen, nitro, and more.")
        include_element_counts = d3.checkbox("Element counts", value=True, help="Adds atom/heteroatom counts and element fractions.")
        include_3d = d4.checkbox("3D conformer descriptors", value=True, help="Generates ETKDG 3D conformers and adds shape descriptors such as PMI, NPR, radius of gyration, asphericity, and spherocity.")
    if st.button("Calculate Descriptors", type="primary", use_container_width=True):
        with st.spinner("Calculating RDKit descriptors and fingerprints..."):
            options = DescriptorOptions(
                include_morgan=include_morgan,
                include_maccs=include_maccs,
                morgan_bits=int(morgan_bits),
                include_full_rdkit=include_full_rdkit,
                include_functional_groups=include_functional_groups,
                include_element_counts=include_element_counts,
                include_3d=include_3d,
            )
            st.session_state.desc_df = calculate_descriptors(valid, options)
            st.session_state.features = descriptor_columns(st.session_state.desc_df)
            st.session_state.descriptor_options = descriptor_options_to_dict(options)
        st.success(f"Calculated {len(st.session_state.features)} numeric features.")

    desc = st.session_state.desc_df
    if isinstance(desc, pd.DataFrame) and not desc.empty:
        st.dataframe(desc.head(25), use_container_width=True)
        dataframe_download(desc, "chemical_descriptors.csv", key_prefix="descriptors_tab")
        criteria = drug_likeness(desc)
        st.write("Drug-likeness criteria")
        st.dataframe(criteria, use_container_width=True)
        dataframe_download(criteria, "drug_likeness_criteria.csv", key_prefix="descriptors_tab")
        fg_table = nonzero_functional_group_table(desc)
        if not fg_table.empty:
            st.write("Detected functional groups")
            st.dataframe(fg_table, use_container_width=True)
            dataframe_download(fg_table, "functional_group_counts.csv", key_prefix="descriptors_tab")


def render_modeling() -> None:
    st.subheader("Train QSAR Models")
    desc = st.session_state.desc_df
    if not isinstance(desc, pd.DataFrame) or desc.empty:
        st.warning("Calculate descriptors first.")
        return
    smiles_col = "SMILES"
    stored_targets = st.session_state.get("target_candidates", [])
    targets = [col for col in stored_targets if col in desc.columns and col != smiles_col]
    if not targets:
        targets = guess_target_columns(desc[[col for col in desc.columns if not col.startswith(("RDKit_", "FG_", "Elem_", "Morgan_", "MACCS_"))]], smiles_col)
    if not targets:
        st.info("No target column found. Add an activity/property column to train QSAR models.")
        return
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target column", targets)
    task_type = c2.selectbox("Task type", ["Regression", "Classification"])
    test_size = c3.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    possible_target_columns = set(targets)
    all_features = [f for f in st.session_state.features if f not in possible_target_columns]
    feature_mode = st.selectbox(
        "Feature set for modeling",
        ["Core descriptors + functional groups", "All descriptors except fingerprints", "All descriptors + fingerprints"],
        index=0,
        help="The recommended setting avoids high-dimensional fingerprints for small datasets and usually gives more stable validation metrics.",
    )
    features = feature_subset(all_features, feature_mode)
    st.caption(f"Using {len(features)} model features from {len(all_features)} calculated numeric columns.")
    models = model_catalog(task_type, n_classes=desc[target].nunique())
    if task_type == "Regression":
        default = [name for name in ["Ridge", "Random Forest", "Extra Trees", "Gradient Boosting", "SVR (RBF)", "kNN"] if name in models]
    else:
        default = [name for name in ["Logistic Regression", "Random Forest", "Extra Trees", "SVM (RBF)", "kNN", "Naive Bayes"] if name in models]
    selected = st.multiselect("Models to compare", list(models.keys()), default=default)
    cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
    if st.button("Train and Compare Models", type="primary", use_container_width=True):
        try:
            with st.spinner("Training models..."):
                st.session_state.training = evaluate_models(desc, features, target, task_type, selected, test_size, cv_folds)
                st.session_state.selected_bundle = st.session_state.training["best"]
            st.success(f"Best model: {st.session_state.training['best_name']}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    result = st.session_state.training
    if isinstance(result, dict):
        leaderboard = result["leaderboard"]
        st.dataframe(leaderboard, use_container_width=True)
        metric = "FitQuality_0_100" if task_type == "Regression" and "FitQuality_0_100" in leaderboard.columns else ("R2" if task_type == "Regression" else "F1")
        st.plotly_chart(px.bar(leaderboard, x="Model", y=metric, template=PLOT_TEMPLATE, title="Model Leaderboard"), use_container_width=True, key="leaderboard")
        if task_type == "Regression" and metric == "FitQuality_0_100":
            st.caption("Model selection uses repeated cross-validation when possible. FitQuality_0_100 is a non-negative summary score; keep checking R2/RMSE for scientific reporting.")
        if task_type == "Regression" and "R2" in leaderboard.columns and leaderboard["R2"].max() < 0:
            st.warning("All test-set R2 values are negative. This usually means the dataset is too small, noisy, or chemically inconsistent for the chosen split/features. Try the recommended feature set, add more compounds, or use cross-validation/external validation.")


def render_evaluate() -> None:
    st.subheader("Evaluate and Interpret")
    bundle = st.session_state.selected_bundle
    if not isinstance(bundle, dict):
        st.warning("Train a model first.")
        return
    st.write("Model summary")
    st.dataframe(model_summary(bundle, bundle.get("model_name")), use_container_width=True, hide_index=True)
    y_test = bundle["y_test"]
    y_pred = bundle["y_pred"]
    if bundle["task_type"] == "Regression":
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, template=PLOT_TEMPLATE, title="Actual vs Predicted")
        min_v, max_v = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal"))
        st.plotly_chart(fig, use_container_width=True, key="actual_pred")
        residuals = np.asarray(y_test) - np.asarray(y_pred)
        st.plotly_chart(px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residual"}, template=PLOT_TEMPLATE, title="Residual Plot"), use_container_width=True, key="residuals")
    else:
        labels = bundle["label_encoder"].classes_.tolist() if bundle.get("label_encoder") is not None else sorted(set(y_test))
        matrix = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(matrix, x=labels, y=labels, text_auto=True, template=PLOT_TEMPLATE, title="Confusion Matrix"), use_container_width=True, key="confusion")

    imp = model_feature_importance(bundle)
    if not imp.empty:
        st.write("Feature importance / coefficients")
        st.plotly_chart(px.bar(imp.head(25), x="Importance", y="Feature", orientation="h", template=PLOT_TEMPLATE, title="Top Important Features"), use_container_width=True, key="importance")
        st.dataframe(imp.head(50), use_container_width=True)
    else:
        st.info("This model does not expose direct feature importance.")


def render_predict() -> None:
    st.subheader("Predict New Compounds")
    bundle = st.session_state.selected_bundle
    if not isinstance(bundle, dict):
        st.warning("Train a model first.")
        return
    left, right = st.columns([1, 1])
    with left:
        text = st.text_area("Paste new SMILES", height=160, placeholder="CCO, ethanol\nc1ccccc1, benzene")
    with right:
        pred_file = st.file_uploader("Or upload prediction CSV, Excel, TXT, or SDF", type=["csv", "xlsx", "xls", "txt", "sdf", "sd"], key="prediction_upload")
    direction = infer_direction(bundle["target"])
    direction = st.selectbox("Ranking direction for regression", ["higher", "lower"], index=0 if direction == "higher" else 1)
    if bundle.get("task_type") == "Regression" and direction == "lower":
        st.info(
            "For lower-is-better targets such as DockingScore, IC50, MIC, Ki, or binding energy, "
            "`Prediction` is shown as a positive optimized design score. The original model output remains in `RawModelPrediction`."
        )
    if st.button("Predict New Molecules", type="primary", use_container_width=True):
        if pred_file is not None:
            new_df = read_uploaded_file(pred_file)
            smiles_col = guess_smiles_column(new_df)
            if smiles_col is None:
                st.error("No SMILES-like column was detected in the prediction file.")
                return
            if smiles_col != "SMILES":
                new_df = new_df.rename(columns={smiles_col: "SMILES"})
        else:
            new_df = parse_manual_smiles(text)
        valid, invalid = validate_molecules(new_df, "SMILES")
        if not invalid.empty:
            st.warning(f"{len(invalid)} invalid molecules were skipped.")
        if valid.empty:
            st.error("No valid molecules were available for prediction.")
            return
        options = normalize_descriptor_options(bundle.get("descriptor_options") or infer_descriptor_options(bundle["features"]))
        desc = calculate_descriptors(valid, options)
        missing = [f for f in bundle["features"] if f not in desc.columns]
        for col in missing:
            desc[col] = 0.0
        st.session_state.prediction_descriptors = desc
        st.session_state.predictions = predict_new(bundle, desc, direction)

    pred = st.session_state.predictions
    if isinstance(pred, pd.DataFrame) and not pred.empty:
        st.dataframe(pred, use_container_width=True)
        st.plotly_chart(px.bar(pred.head(30), x="Name", y="RankingScore", template=PLOT_TEMPLATE, title="Prediction Ranking"), use_container_width=True, key="ranking")
        pred_desc = st.session_state.prediction_descriptors
        if isinstance(pred_desc, pd.DataFrame) and not pred_desc.empty:
            st.write("Drug-likeness criteria for new compounds")
            criteria = drug_likeness(pred_desc)
            st.dataframe(criteria, use_container_width=True)
            st.write("Applicability domain compared with the training set")
            domain = applicability_domain(pred_desc, bundle)
            st.dataframe(domain, use_container_width=True)
        cols = st.columns(min(4, len(pred)))
        for idx, (_, row) in enumerate(pred.head(4).iterrows()):
            with cols[idx % len(cols)]:
                image_uri = mol_png_data_uri(row["SMILES"])
                if image_uri:
                    st.image(image_uri, caption=row["Name"])
                else:
                    st.caption(row["Name"])
                    st.code(row["SMILES"], language="text")
        dataframe_download(pred, "compound_predictions.csv", key_prefix="predict_tab")


def render_molecule_viewer() -> None:
    st.subheader("Molecule Drawing & 3D Viewer")
    desc = st.session_state.desc_df
    example_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    options: list[str] = []
    if isinstance(desc, pd.DataFrame) and not desc.empty and "SMILES" in desc.columns:
        options = [f"{row.get('Name', f'Molecule_{idx+1}')} | {row['SMILES']}" for idx, row in desc.head(250).iterrows()]

    source = st.radio("Molecule source", ["Type SMILES", "Select from loaded data"], horizontal=True)
    if source == "Select from loaded data" and options:
        selected = st.selectbox("Loaded molecule", options)
        smiles = selected.split(" | ", 1)[1]
    else:
        smiles = st.text_input("SMILES", value=example_smiles, help="Paste a SMILES string to draw 2D and generate an approximate 3D conformer.")

    mol = mol_from_smiles(smiles)
    if mol is None:
        st.warning("Enter a valid SMILES string.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### 2D drawing")
        image_uri = mol_png_data_uri(smiles, size=(360, 260))
        if image_uri:
            st.image(image_uri)
        else:
            st.code(Chem.MolToSmiles(mol), language="text")
        st.write("Canonical SMILES")
        st.code(Chem.MolToSmiles(mol), language="text")

    with c2:
        st.markdown("#### 3D conformer")
        fig, status = plot_molecule_3d(smiles)
        st.caption(status)
        st.plotly_chart(fig, use_container_width=True, key="molecule_3d_viewer")

    desc3d = descriptors_3d_from_smiles(smiles)
    st.markdown("#### 3D shape descriptors")
    st.dataframe(pd.DataFrame([desc3d]).T.reset_index().rename(columns={"index": "Descriptor", 0: "Value"}), use_container_width=True, hide_index=True)


def render_alignment_design() -> None:
    st.subheader("Alignment & Activity-Guided Design")
    desc = st.session_state.desc_df
    if not isinstance(desc, pd.DataFrame) or desc.empty:
        st.warning("Calculate descriptors first.")
        return

    targets = [col for col in st.session_state.get("target_candidates", []) if col in desc.columns]
    if not targets:
        targets = guess_target_columns(desc, "SMILES")
    if not targets:
        st.warning("No activity/property target column is available for design guidance.")
        return

    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Activity/property column", targets, key="align_target")
    is_numeric = pd.api.types.is_numeric_dtype(desc[target])
    if is_numeric:
        direction = infer_direction(target)
        direction = c2.selectbox("Best activity means", ["higher", "lower"], index=0 if direction == "higher" else 1, key="align_direction")
        active_class = None
    else:
        classes = sorted(desc[target].dropna().astype(str).unique().tolist())
        default_idx = classes.index("Active") if "Active" in classes else 0
        active_class = c2.selectbox("Desired class", classes, index=default_idx, key="align_class")
        direction = "higher"

    reference = best_reference_row(desc, target, direction, active_class)
    c3.metric("Reference compound", str(reference.get("Name", "Reference")))

    st.info(
        "This tab uses RDKit Morgan similarity, maximum common substructure (MCS), Murcko scaffold matching, "
        "and functional-group enrichment. It suggests dataset-driven design criteria; it is not a docking or experimental guarantee."
    )

    ref_cols = st.columns([1, 2])
    with ref_cols[0]:
        image_uri = mol_png_data_uri(str(reference["SMILES"]), size=(300, 220))
        if image_uri:
            st.image(image_uri, caption=f"Reference: {reference.get('Name', '')}")
        else:
            st.code(str(reference["SMILES"]), language="text")
    with ref_cols[1]:
        st.write("Best-activity reference details")
        details = pd.DataFrame({
            "Item": ["Name", "SMILES", target, "Murcko scaffold"],
            "Value": [
                reference.get("Name", ""),
                reference.get("SMILES", ""),
                reference.get(target, ""),
                scaffold_smiles(str(reference["SMILES"])),
            ],
        })
        st.dataframe(details, use_container_width=True, hide_index=True)

    align_df = similarity_to_reference(desc, reference)
    if not align_df.empty:
        st.write("Compound alignment/similarity to the best reference")
        if target in desc.columns:
            align_df = align_df.merge(desc[["Name", target]], on="Name", how="left")
        st.dataframe(align_df.head(50), use_container_width=True)
        st.plotly_chart(
            px.scatter(
                align_df,
                x="MorganTanimotoToReference",
                y="MCSAtomFraction",
                color=target if target in align_df.columns else None,
                hover_name="Name",
                template=PLOT_TEMPLATE,
                title="Similarity and MCS Alignment to Best Reference",
            ),
            use_container_width=True,
            key="alignment_similarity_scatter",
        )
        dataframe_download(align_df, "compound_alignment_to_reference.csv", key_prefix="alignment_tab")

    sim_df, sim_fig = pairwise_similarity_matrix(desc)
    st.plotly_chart(sim_fig, use_container_width=True, key="pairwise_similarity_matrix")
    if not sim_df.empty:
        dataframe_download(sim_df.reset_index(names="Compound"), "pairwise_similarity_matrix.csv", key_prefix="alignment_tab")

    profile = activity_design_profile(desc, target, direction, active_class)
    if not profile.empty:
        st.write("Suggested property window from best-activity compounds")
        st.dataframe(profile, use_container_width=True, hide_index=True)

    fg_suggestions = functional_group_activity_suggestions(desc, target, direction, active_class)
    if not fg_suggestions.empty:
        st.write("Functional groups enriched or depleted in better compounds")
        st.dataframe(fg_suggestions.head(40), use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(
                fg_suggestions.head(20),
                x="EnrichmentDelta",
                y="FunctionalGroup",
                color="DesignSuggestion",
                orientation="h",
                template=PLOT_TEMPLATE,
                title="Functional Group Design Suggestions",
            ),
            use_container_width=True,
            key="fg_design_suggestions",
        )

    correlations = descriptor_activity_correlations(desc, target)
    if not correlations.empty:
        st.write("Descriptor correlations with activity/property")
        st.dataframe(correlations.head(40), use_container_width=True, hide_index=True)


def render_visuals() -> None:
    st.subheader("Visualizations")
    desc = st.session_state.desc_df
    if not isinstance(desc, pd.DataFrame) or desc.empty:
        st.warning("Calculate descriptors first.")
        return
    targets = [col for col in st.session_state.get("target_candidates", []) if col in desc.columns]
    if not targets:
        targets = guess_target_columns(desc, "SMILES")
    features = [feature for feature in st.session_state.features if feature not in set(targets)]
    color = st.selectbox("Color PCA by", ["None"] + targets)
    st.plotly_chart(plot_descriptor_hist(desc, [c for c in features if not c.startswith(("Morgan_", "MACCS_"))]), use_container_width=True, key="hist")
    st.plotly_chart(plot_corr(desc, [c for c in features if not c.startswith(("Morgan_", "MACCS_"))]), use_container_width=True, key="corr")
    st.plotly_chart(plot_functional_group_heatmap(desc), use_container_width=True, key="fg_heatmap")
    fig, loadings = pca_plot(desc, features, None if color == "None" else color)
    st.plotly_chart(fig, use_container_width=True, key="pca")
    st.write("Top PCA loadings")
    st.dataframe(loadings.head(30), use_container_width=True)


def render_export() -> None:
    st.subheader("Export")
    desc = st.session_state.desc_df
    training = st.session_state.training
    bundle = st.session_state.selected_bundle
    pred = st.session_state.predictions

    c1, c2 = st.columns(2)
    with c1:
        if isinstance(desc, pd.DataFrame) and not desc.empty:
            dataframe_download(desc, "chemical_descriptors.csv", key_prefix="export_tab")
        if isinstance(pred, pd.DataFrame) and not pred.empty:
            dataframe_download(pred, "compound_predictions.csv", key_prefix="export_tab")
    with c2:
        if isinstance(training, dict) and isinstance(training.get("leaderboard"), pd.DataFrame):
            dataframe_download(training["leaderboard"], "model_leaderboard.csv", key_prefix="export_tab")
        if isinstance(bundle, dict):
            st.download_button(
                "Download trained model bundle (.pkl)",
                data=pickle_model_bundle(bundle),
                file_name="chemical_qsar_model.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                key="download_model_bundle",
            )

    st.download_button(
        "Download HTML summary report",
        data=generate_html_report().encode("utf-8"),
        file_name="chemical_qsar_report.html",
        mime="text/html",
        use_container_width=True,
        key="download_html_report",
    )


def render_about() -> None:
    st.subheader("About ChemBlast")
    st.markdown("### Developed by Ahmed G. Soliman")
    st.dataframe(pd.DataFrame(DEVELOPER_PROFILE.items(), columns=["Item", "Details"]), use_container_width=True, hide_index=True)
    st.markdown(
        """
ChemBlast uses RDKit for 2D/3D chemical descriptor and fingerprint calculation and scikit-learn for QSAR modeling.

Scientific basis:

- Molecular descriptors summarize physicochemical properties such as molecular weight, LogP, TPSA, H-bond donors/acceptors, rings, rotatable bonds, aromaticity, complexity, QED, and partial-charge summaries.
- 3D conformer descriptors are generated from ETKDG conformers and summarize molecular shape using PMI, NPR, radius of gyration, asphericity, eccentricity, spherocity, and related geometric descriptors when embedding succeeds.
- Morgan fingerprints encode circular atom environments and are widely used for ligand similarity and QSAR modeling.
- MACCS keys encode predefined structural fragments.
- QSAR models learn statistical relationships between descriptors/fingerprints and biological activity or chemical properties.

Limitations:

- QSAR models are dataset-dependent and can overfit small datasets.
- Docking scores are computational proxies and should not be treated as experimental activity.
- Applicability domain and external validation are strongly recommended.
        """
    )
    st.link_button("Open developer portfolio", DEVELOPER_PORTFOLIO, use_container_width=True)


def main() -> None:
    init_state()
    st.sidebar.title(APP_NAME)
    st.sidebar.caption(f"Developed by {DEVELOPER_NAME}")
    tabs = st.tabs(["Home", "Input", "Descriptors", "Train Model", "Evaluate", "Predict", "Molecule Viewer", "Alignment & Design", "Visualizations", "Export", "About"])
    with tabs[0]:
        render_home()
    with tabs[1]:
        render_input()
    with tabs[2]:
        render_descriptors()
    with tabs[3]:
        render_modeling()
    with tabs[4]:
        render_evaluate()
    with tabs[5]:
        render_predict()
    with tabs[6]:
        render_molecule_viewer()
    with tabs[7]:
        render_alignment_design()
    with tabs[8]:
        render_visuals()
    with tabs[9]:
        render_export()
    with tabs[10]:
        render_about()


if __name__ == "__main__":
    main()

