"""Chemical compound QSAR web app built with Streamlit and RDKit."""

from __future__ import annotations

import base64
import io
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdFingerprintGenerator, rdMolDescriptors
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
from sklearn.feature_selection import VarianceThreshold
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOT_TEMPLATE = "plotly_white"


st.set_page_config(
    page_title="Chemical QSAR Web Tool",
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


def clean_smiles(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol


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


def bitvect_to_dict(prefix: str, bitvect: Any, n_bits: int | None = None) -> dict[str, float]:
    if n_bits is None:
        n_bits = bitvect.GetNumBits()
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(arr)}


def calculate_descriptors(df: pd.DataFrame, options: DescriptorOptions) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        smiles = clean_smiles(row["SMILES"])
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        data = row.to_dict()
        data["SMILES"] = Chem.MolToSmiles(mol)
        data.update(base_descriptors(mol))
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
    )


def descriptor_columns(df: pd.DataFrame) -> list[str]:
    protected = {"Name", "SMILES"}
    return [col for col in df.columns if col not in protected and pd.api.types.is_numeric_dtype(df[col])]


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
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
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
            cv = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            cv_score = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            metrics["CV_R2_mean"] = float(np.mean(cv_score))
            primary = metrics["R2"]
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
            primary = metrics["F1"]

        domain_features = [col for col in features if not col.startswith(("Morgan_", "MACCS_"))]
        row = {"Model": name, **metrics}
        rows.append(row)
        bundles[name] = {
            "pipeline": pipe,
            "model_name": name,
            "features": features,
            "target": target,
            "task_type": task_type,
            "label_encoder": encoder,
            "y_test": y_test,
            "y_pred": pred,
            "descriptor_options": infer_descriptor_options(features),
            "domain_features": domain_features,
            "domain_min": X[domain_features].min(numeric_only=True).to_dict(),
            "domain_max": X[domain_features].max(numeric_only=True).to_dict(),
        }

    leaderboard = pd.DataFrame(rows)
    sort_metric = "R2" if task_type == "Regression" else "F1"
    leaderboard = leaderboard.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    best_name = leaderboard.loc[0, "Model"]
    return {"leaderboard": leaderboard, "bundles": bundles, "best_name": best_name, "best": bundles[best_name]}


def model_feature_importance(bundle: dict[str, Any]) -> pd.DataFrame:
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    features = np.array(bundle["features"])
    support = pipe.named_steps["variance"].get_support()
    used_features = features[support]
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
        out["Prediction"] = pred
        out["RankingScore"] = -pred if direction == "lower" else pred
    else:
        encoder = bundle["label_encoder"]
        labels = encoder.inverse_transform(pred.astype(int)) if encoder is not None else pred
        out["PredictedClass"] = labels
        if hasattr(bundle["pipeline"], "predict_proba"):
            proba = bundle["pipeline"].predict_proba(X)
            out["MaxProbability"] = np.max(proba, axis=1)
            out["RankingScore"] = out["MaxProbability"]
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


def pca_plot(df: pd.DataFrame, features: list[str], color_col: str | None = None) -> tuple[go.Figure, pd.DataFrame]:
    if len(features) < 2 or len(df) < 2:
        fig = px.scatter(title="PCA needs at least two molecules and two features", template=PLOT_TEMPLATE)
        return fig, pd.DataFrame()
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = RobustScaler().fit_transform(X)
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
    loadings = pd.DataFrame({"Feature": features, "PC1_Loading": pca.components_[0], "PC2_Loading": pca.components_[1]})
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
  <title>Chemical QSAR Web Tool Report</title>
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
  <h1>Chemical QSAR Web Tool Report</h1>
  <p>Generated: {timestamp}</p>
  <p>This report summarizes RDKit descriptor calculation, model comparison, drug-likeness checks, and prediction outputs created in the web app.</p>
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
  <h1>Chemical QSAR Web Tool</h1>
  <p>Web-only QSAR platform for small molecules: SMILES/SDF input, RDKit descriptors, fingerprints, ML models, visualizations, and browser-based prediction.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Input", "SMILES / SDF / CSV")
    c2.metric("Descriptors", "RDKit + Fingerprints")
    c3.metric("Models", "Regression + Classification")
    st.info("Start with the example dataset, or upload your own molecules with a target column such as IC50, pIC50, DockingScore, LogS, Toxicity, or ActivityClass.")


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
    if st.button("Calculate Descriptors", type="primary", use_container_width=True):
        with st.spinner("Calculating RDKit descriptors and fingerprints..."):
            options = DescriptorOptions(include_morgan=include_morgan, include_maccs=include_maccs, morgan_bits=int(morgan_bits))
            st.session_state.desc_df = calculate_descriptors(valid, options)
            st.session_state.features = descriptor_columns(st.session_state.desc_df)
            st.session_state.descriptor_options = options
        st.success(f"Calculated {len(st.session_state.features)} numeric features.")

    desc = st.session_state.desc_df
    if isinstance(desc, pd.DataFrame) and not desc.empty:
        st.dataframe(desc.head(25), use_container_width=True)
        dataframe_download(desc, "chemical_descriptors.csv", key_prefix="descriptors_tab")
        criteria = drug_likeness(desc)
        st.write("Drug-likeness criteria")
        st.dataframe(criteria, use_container_width=True)
        dataframe_download(criteria, "drug_likeness_criteria.csv", key_prefix="descriptors_tab")


def render_modeling() -> None:
    st.subheader("Train QSAR Models")
    desc = st.session_state.desc_df
    if not isinstance(desc, pd.DataFrame) or desc.empty:
        st.warning("Calculate descriptors first.")
        return
    smiles_col = "SMILES"
    targets = guess_target_columns(desc, smiles_col)
    if not targets:
        st.info("No target column found. Add an activity/property column to train QSAR models.")
        return
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target column", targets)
    task_type = c2.selectbox("Task type", ["Regression", "Classification"])
    test_size = c3.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    features = [f for f in st.session_state.features if f != target]
    models = model_catalog(task_type, n_classes=desc[target].nunique())
    default = list(models.keys())[:6]
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
        metric = "R2" if task_type == "Regression" else "F1"
        st.plotly_chart(px.bar(leaderboard, x="Model", y=metric, template=PLOT_TEMPLATE, title="Model Leaderboard"), use_container_width=True, key="leaderboard")


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
        options = bundle.get("descriptor_options") or infer_descriptor_options(bundle["features"])
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


def render_visuals() -> None:
    st.subheader("Visualizations")
    desc = st.session_state.desc_df
    if not isinstance(desc, pd.DataFrame) or desc.empty:
        st.warning("Calculate descriptors first.")
        return
    features = st.session_state.features
    targets = guess_target_columns(desc, "SMILES")
    color = st.selectbox("Color PCA by", ["None"] + targets)
    st.plotly_chart(plot_descriptor_hist(desc, [c for c in features if not c.startswith(("Morgan_", "MACCS_"))]), use_container_width=True, key="hist")
    st.plotly_chart(plot_corr(desc, [c for c in features if not c.startswith(("Morgan_", "MACCS_"))]), use_container_width=True, key="corr")
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
    st.subheader("About")
    st.markdown(
        """
This web-only tool uses RDKit for chemical descriptor and fingerprint calculation and scikit-learn for QSAR modeling.

Scientific basis:

- Molecular descriptors summarize physicochemical properties such as molecular weight, LogP, TPSA, H-bond donors/acceptors, rings, rotatable bonds, aromaticity, complexity, QED, and partial-charge summaries.
- Morgan fingerprints encode circular atom environments and are widely used for ligand similarity and QSAR modeling.
- MACCS keys encode predefined structural fragments.
- QSAR models learn statistical relationships between descriptors/fingerprints and biological activity or chemical properties.

Limitations:

- QSAR models are dataset-dependent and can overfit small datasets.
- Docking scores are computational proxies and should not be treated as experimental activity.
- Applicability domain and external validation are strongly recommended.
        """
    )


def main() -> None:
    init_state()
    st.sidebar.title("Chemical QSAR")
    st.sidebar.caption("Web-only molecular QSAR workflow")
    tabs = st.tabs(["Home", "Input", "Descriptors", "Train Model", "Evaluate", "Predict", "Visualizations", "Export", "About"])
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
        render_visuals()
    with tabs[7]:
        render_export()
    with tabs[8]:
        render_about()


if __name__ == "__main__":
    main()

