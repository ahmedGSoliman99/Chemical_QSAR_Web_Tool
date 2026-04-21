# Deploy The Chemical QSAR Web Tool On Streamlit Cloud

This project is designed as a web-only Streamlit app. It does not need an EXE file.

## Fast Deployment

1. Open https://share.streamlit.io/.
2. Sign in with the same GitHub account that owns the repository.
3. Click `Create app` or `New app`.
4. Select the repository: `ahmedGSoliman99/Chemical_QSAR_Web_Tool`.
5. Select branch: `main`.
6. Main file path: `app.py`.
7. Click `Deploy`.

After deployment, Streamlit will give you a public link ending in `.streamlit.app`. Anyone can open that link in a browser.

## Notes

- GitHub Pages can display the landing page only.
- The interactive RDKit and machine-learning app must run on Streamlit Cloud or another Python host.
- If deployment fails while installing RDKit, keep Python at 3.11 and redeploy.
