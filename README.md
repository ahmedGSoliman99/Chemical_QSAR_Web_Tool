# ChemBlast

A web-only QSAR platform for chemical compounds and drug-like molecules. The app is built with Streamlit, RDKit, scikit-learn, and Plotly so non-programmers can upload molecules, calculate descriptors, train machine-learning models, visualize chemical space, and predict new compounds from a browser.

## What The Tool Does

- Accepts molecule input from SMILES, CSV, Excel, TXT, and SDF files.
- Validates and canonicalizes molecules with RDKit.
- Calculates chemistry-aware molecular descriptors, the full available RDKit numeric descriptor list, functional-group counters, element counts, and fingerprints.
- Supports experimental activities, physicochemical endpoints, toxicity labels, activity classes, and docking scores as target columns.
- Trains regression and classification QSAR models.
- Includes nonlinear models such as SVR/SVM, kNN, Random Forest, Extra Trees, Gradient Boosting, and MLP.
- Evaluates models with appropriate metrics and plots.
- Predicts new compounds and ranks them by predicted activity/property.
- Aligns compounds against the best-activity reference using Morgan similarity, MCS overlap, and Murcko scaffold matching.
- Suggests activity-guided property windows and functional groups enriched in better compounds.
- Shows drug-likeness criteria and applicability-domain checks for new molecules.
- Exports descriptor tables, prediction tables, model leaderboards, trained model bundles, and an HTML report.

## Scientific Basis

The descriptor engine uses RDKit to calculate accepted cheminformatics descriptors, including molecular weight, exact molecular weight, LogP, molar refractivity, TPSA, hydrogen-bond donors and acceptors, rotatable bonds, ring counts, aromatic rings, fraction sp3, formal charge, QED, Bertz complexity, Balaban J, Labute ASA, chi/kappa indices, partial-charge summaries, the full RDKit numeric descriptor list, element counts, and RDKit fragment-based functional-group counters.

The fingerprint engine can calculate Morgan fingerprints and MACCS keys. Morgan fingerprints represent circular atom neighborhoods and are widely used for ligand similarity and QSAR modeling. MACCS keys represent predefined chemical substructure patterns.

Functional-group counters are computed from RDKit `Fragments` patterns and summarize motifs such as amides, esters, phenols, carboxylic acids, halogens, nitro groups, amines, ethers, and related medicinal-chemistry fragments.

The alignment/design module chooses a best-activity reference molecule, then compares other compounds by Morgan Tanimoto similarity, maximum common substructure overlap, and Murcko scaffold matching. It also compares high-activity compounds against the rest of the dataset to propose property ranges and functional-group motifs associated with better activity.

The modeling workflow uses scikit-learn pipelines with imputation, variance filtering, scaling, model fitting, train/test split, and cross-validation. The tool can treat docking score columns as regression targets, but docking scores should be interpreted as computational proxies rather than experimental biological activity.

## Project Structure

```text
Chemical_QSAR_Web_Tool/
  app.py
  environment.yml
  requirements.txt
  packages.txt
  README.md
  DEPLOY_STREAMLIT.md
  data/
    example_compounds.csv
  docs/
    index.html
  .streamlit/
    config.toml
```

## Run Locally On Windows

### Easiest method

Double-click:

```text
run_app.bat
```

The launcher creates a local `.venv` folder, installs the required packages, starts Streamlit, and opens:

```text
http://localhost:8501
```

Keep the black launcher window open while using the app. Close the window to stop the local server.

### Manual method

1. Install Python 3.11 from https://www.python.org/downloads/windows/.
2. Open PowerShell in this project folder.
3. Create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The browser will open automatically. If not, copy the local URL shown in the terminal.

## Deploy As A Website

The easiest hosting option is Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io/.
3. Choose `New app`.
4. Select this GitHub repository.
5. Set the main file path to `app.py`.
6. Open `Advanced settings` and choose Python `3.11` if the selector appears.
7. Deploy.

GitHub Pages can host the landing page in `docs/index.html`, but it cannot run the Python/RDKit Streamlit app. The actual interactive QSAR app needs Streamlit Cloud or another Python-capable host.

This repository includes `environment.yml` for Streamlit Cloud. It requests Python 3.11 and RDKit from conda-forge, which avoids newer default Python runtimes that may not yet support all RDKit drawing modules. The `requirements.txt` file is kept for local Windows installation.

## Input File Formats

The built-in example dataset contains a synthetic column named `Demo_pIC50`. It is only for checking that the software workflow behaves correctly. Replace it with real experimental activity, physicochemical property, toxicity, docking score, or class labels for research use.

### CSV / Excel

Recommended columns:

```text
Name,SMILES,pIC50,ActivityClass,DockingScore,Toxicity,LogS
```

Only `SMILES` is required for descriptor calculation and prediction. A target column is required for model training.

### TXT

One SMILES per line. Optional compound names can be added after a comma or tab.

```text
CCO, ethanol
CC(=O)OC1=CC=CC=C1C(=O)O, aspirin
```

### SDF

The app reads molecules and SDF property fields. Any numeric SDF property can be selected as a regression target. Any categorical property can be used as a classification target.

## Model Types

Regression models:

- Linear Regression
- Ridge
- Lasso
- Elastic Net
- Random Forest
- Extra Trees
- Gradient Boosting
- SVR with RBF kernel
- kNN regressor
- MLP regressor

Classification models:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- SVM with RBF kernel
- kNN classifier
- Naive Bayes
- MLP classifier

## Important Scientific Notes

- QSAR quality depends strongly on dataset size, chemical diversity, assay quality, and endpoint consistency.
- Negative R2 can occur when a model predicts worse than the test-set mean; this is a diagnostic warning, not a software error.
- Always validate promising compounds experimentally.
- For publication work, report descriptor settings, model type, split strategy, cross-validation metrics, external validation results, and applicability-domain analysis.

## Developer

Prepared for Ahmed G. Soliman.

Developer website: https://sites.google.com/view/ahmed-g-soliman/home




