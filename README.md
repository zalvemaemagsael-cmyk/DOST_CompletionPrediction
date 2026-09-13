# MSME Project Risk Dashboard

A single-page Streamlit app that answers one question on load: **which MSME
projects are most likely to fail, and roughly why.**

It loads the monthly project panel (`MSME_synthetic_panel_data.xlsx`) and the
fitted completion model (`MSME_CompletionModel_panel.pkl`), scores every
project on its most recent monthly record, and surfaces the highest-risk
projects at the top with a plain-English reason derived directly from the
logistic regression's own coefficients.

## Files

- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- `MSME_synthetic_panel_data.xlsx` — the panel data (must stay next to `app.py`)
- `MSME_CompletionModel_panel.pkl` — the fitted model (must stay next to `app.py`)

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Push this folder (including the `.xlsx` and `.pkl` files) to a **public
   GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and point
   it at the repo with `app.py` as the main file.
3. Streamlit Cloud installs `requirements.txt` automatically and deploys the
   app — no other configuration is needed.

## Notes

- The dataset is synthetic and intended for demonstration only (noted in the
  app itself via a small caption).
- Risk is `1 - P(Completed)` from the model's `predict_proba`. Projects are
  *ranked* using the underlying log-odds rather than the raw probability,
  since probabilities can saturate near 0%/100% for a strongly separating
  linear model — the log-odds preserves a meaningful order even then.
- "Why flagged" text is generated from each feature's standardized value
  times its fitted coefficient (no SHAP or extra dependencies needed for a
  linear model).
