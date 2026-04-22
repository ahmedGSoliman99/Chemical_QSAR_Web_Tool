# Deploy ChemBlast On Streamlit Community Cloud

This project is designed as a web-only Streamlit app. It does not need an EXE file.

## Fast Deployment

1. Open https://share.streamlit.io/.
2. Sign in with the same GitHub account that owns the repository.
3. Click `Create app` or `New app`.
4. Select the repository: `ahmedGSoliman99/Chemical_QSAR_Web_Tool`.
5. Select branch: `main`.
6. Main file path: `app.py`.
7. Open `Advanced settings`.
8. Choose Python `3.12` first. If RDKit still fails, redeploy with Python `3.11`.
9. Click `Deploy`.

After deployment, Streamlit will give you a public link ending in `.streamlit.app`. Anyone can open that link in a browser.

## If Deployment Is Already Broken

If the failed app was created with Python `3.14`, delete that Streamlit app and create it again. Streamlit Community Cloud does not change the Python runtime of an already-created app in place.

Use these exact deployment values:

- Repository: `ahmedGSoliman99/Chemical_QSAR_Web_Tool`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12` or `3.11`

## Notes

- GitHub Pages can display the landing page only.
- The interactive RDKit and machine-learning app must run on Streamlit Cloud or another Python host.
- This repository intentionally uses only `requirements.txt` for Python packages on Streamlit Cloud.
- `packages.txt` provides the Linux system libraries needed by RDKit/Pillow rendering.
- Do not add another dependency file such as `environment.yml` unless you intentionally want Streamlit Cloud to use conda instead of `requirements.txt`.
