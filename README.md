# F1 Prediction (ML + Website)

This repository combines two connected parts of one project:

1. **Machine Learning outputs** for the F1 2026 predictions  
2. **Flask website** that displays those predictions interactively deployed on render
   website: https://f1-prediction-lab8.onrender.com/

The website is directly powered by the ML artifacts stored in this repo, so they are related and intended to live together.

## Project Structure

```text
F1_prediction/
├── app.py                        # Flask web app entry point
├── requirements.txt              # Python dependencies
├── templates/
│   └── index.html                # Frontend UI
├── data/
│   ├── F1_dataset_2024-2026.xlsx
│   ├── X_predict_preprocessed.xlsx
│   └── predictions_2026.xlsx
├── models/
│   ├── classification_model.pkl
│   └── classification_features.pkl
└── scripts/
    └── check_features.py         # Utility check for model feature alignment
```

## How the Website and ML are Connected

- `models/classification_model.pkl` and `models/classification_features.pkl` are loaded by `app.py` for classification groups.
- `data/predictions_2026.xlsx` provides the predicted finishing position per driver.
- `data/X_predict_preprocessed.xlsx` provides model input features.
- `data/F1_dataset_2024-2026.xlsx` provides driver/team mapping for the UI.
- The Flask API returns prediction data, and `templates/index.html` renders it as the website.

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
python app.py
```

3. Open in browser:

```text
http://127.0.0.1:5000
```
