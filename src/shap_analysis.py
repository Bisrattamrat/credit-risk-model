import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

MODEL_PATH = "best_model.pkl"
DATA_PATH = "data/processed/processed_data.csv"

os.makedirs("images", exist_ok=True)

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Remove target column if it exists
if "is_high_risk" in df.columns:
    X = df.drop(columns=["is_high_risk"])
else:
    X = df

# Keep only numeric features
X = X.select_dtypes(include=["int64", "float64"])

print("Generating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Saving summary plot...")

plt.figure()

if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X, show=False)
else:
    shap.summary_plot(shap_values, X, show=False)

plt.tight_layout()
plt.savefig("images/shap_summary.png", dpi=300)

print("Done!")
print("Image saved to images/shap_summary.png")