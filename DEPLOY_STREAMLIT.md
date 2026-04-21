# Deploy The Chemical QSAR Web Tool On Streamlit Cloud

This project is designed as a web-only Streamlit app. It does not need an EXE file.

## Fast Deployment

1. Open https://share.streamlit.io/.
2. Sign in with the same GitHub account that owns the repository.
3. Click `Create app` or `New app`.
4. Select the repository: `ahmedGSoliman99/Chemical_QSAR_Web_Tool`.
5. Select branch: `main`.
6. Main file path: `app.py`.
7. Open `Advanced settings`.
8. Choose Python `3.11` if Streamlit shows a Python selector.
9. Click `Deploy`.

After deployment, Streamlit will give you a public link ending in `.streamlit.app`. Anyone can open that link in a browser.

## Notes

- GitHub Pages can display the landing page only.
- The interactive RDKit and machine-learning app must run on Streamlit Cloud or another Python host.
- This repository includes `environment.yml` so Streamlit Cloud uses conda-forge with Python 3.11 and RDKit.
- `requirements.txt` is mainly for local Windows runs; Streamlit Cloud should prioritize `environment.yml`.
- If an old deployment already used Python 3.14, delete that Streamlit app and redeploy it. Streamlit does not change the Python runtime of an already-created app in place.
