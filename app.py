from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import joblib
import requests

app = Flask(__name__)

# ── Load classification pkl files ─────────────────────────────────────────────
classification_model    = joblib.load("classification_model.pkl")
classification_features = joblib.load("classification_features.pkl")

# ── Load 2026 prediction features ─────────────────────────────────────────────
X_predict = pd.read_excel("X_predict_preprocessed.xlsx")

# ── Load regression lookup ────────────────────────────────────────────────────
reg_df = pd.read_excel("predictions_2026.xlsx")

# ── Load raw dataset ──────────────────────────────────────────────────────────
raw_df = pd.read_excel("F1_dataset_2024-2026.xlsx")
df_2026 = raw_df[raw_df['Year'] == 2026][['Driver', 'Constructor']].drop_duplicates()
driver_team = {row['Driver']: row['Constructor'] for _, row in df_2026.iterrows()}

# ── Run classification ────────────────────────────────────────────────────────
drivers_2026 = raw_df[raw_df['Year'] == 2026]['Driver'].reset_index(drop=True)
X_predict_aligned = X_predict[classification_features]
clf_predictions   = classification_model.predict(X_predict_aligned)

clf_results = pd.DataFrame({"Driver": drivers_2026, "Group": clf_predictions})
clf_lookup = {}
for driver, group_df in clf_results.groupby("Driver"):
    clf_lookup[driver] = group_df["Group"].value_counts().idxmax()

# ── Build regression lookup ───────────────────────────────────────────────────
reg_lookup = {}
for _, row in reg_df.iterrows():
    reg_lookup[row["Driver"]] = int(row["Predicted_Position"])

# ── Final driver data ─────────────────────────────────────────────────────────
DRIVERS = sorted(set(clf_lookup.keys()) & set(reg_lookup.keys()))
DRIVER_DATA = {}
for d in DRIVERS:
    DRIVER_DATA[d] = {
        "team":    driver_team.get(d, "Unknown"),
        "reg_pos": reg_lookup[d],
        "group":   clf_lookup[d],
    }

# ── Exact Wikipedia page titles ───────────────────────────────────────────────
# ── Official F1 racing suit image URLs (media.formula1.com) ──────────────────
F1_IMAGE_URLS = {
    "Alexander Albon":   "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/williams/alealb01/2026williamsalealb01right.png",
    "Arvid Lindblad":    "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/racingbulls/arvlin01/2026racingbullsarvlin01right.png",
    "Carlos Sainz":      "https://media.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png",
    "Charles Leclerc":   "https://media.formula1.com/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png",
    "Esteban Ocon":      "https://media.formula1.com/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png",
    "Fernando Alonso":   "https://media.formula1.com/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png",
    "Franco Colapinto":  "https://media.formula1.com/content/dam/fom-website/drivers/F/FRACOL01_Franco_Colapinto/fracol01.png",
    "Gabriel Bortoleto": "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/audi/gabbor01/2026audigabbor01right.png",
    "George Russell":    "https://media.formula1.com/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png",
    "Isack Hadjar":      "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/redbullracing/isahad01/2026redbullracingisahad01right.png",
    "Kimi Antonelli":    "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/mercedes/andant01/2026mercedesandant01right.png",
    "Lance Stroll":      "https://media.formula1.com/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png",
    "Lando Norris":      "https://media.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png",
    "Lewis Hamilton":    "https://media.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png",
    "Liam Lawson":       "https://media.formula1.com/content/dam/fom-website/drivers/L/LIALAW01_Liam_Lawson/lialaw01.png",
    "Max Verstappen":    "https://media.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png",
    "Nico Hulkenberg":   "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/audi/nichul01/2026audinichul01right.png",
    "Oliver Bearman":    "https://media.formula1.com/content/dam/fom-website/drivers/O/OLIBEA01_Oliver_Bearman/olibea01.png",
    "Oscar Piastri":     "https://media.formula1.com/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png",
    "Pierre Gasly":      "https://media.formula1.com/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png",
    "Sergio Perez":      "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/cadillac/serper01/2026cadillacserper01right.png",
    "Valtteri Bottas":   "https://media.formula1.com/image/upload/c_thumb,g_face,z_0.6,w_720,h_720/q_auto/v1740000001/common/f1/2026/cadillac/valbot01/2026cadillacvalbot01right.png",
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", driver_data=DRIVER_DATA)

@app.route("/proxy_image")
def proxy_image():
    """Proxy F1 official images through Flask to avoid browser hotlink blocking."""
    driver = request.args.get("driver", "")
    url = F1_IMAGE_URLS.get(driver, "")
    if not url:
        return "", 404
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer":    "https://www.formula1.com/",
                "Accept":     "image/png,image/*,*/*",
            },
            stream=True,
        )
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "image/png")
            return Response(resp.content, status=200, content_type=content_type)
        return "", resp.status_code
    except Exception:
        return "", 502

@app.route("/predict", methods=["POST"])
def predict():
    data        = request.get_json()
    driver_name = data.get("driver", "")
    if driver_name not in DRIVER_DATA:
        return jsonify({"error": "Driver not found"}), 400
    info = DRIVER_DATA[driver_name]
    return jsonify({
        "driver":   driver_name,
        "position": info["reg_pos"],
        "group":    info["group"],
        "team":     info["team"],
    })

if __name__ == "__main__":
    app.run(debug=True)