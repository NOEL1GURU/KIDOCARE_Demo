import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# STEP 1: Load Dataset
data = pd.read_excel(r"C:\Users\Noel\Demo\ImputedData.xlsx")
print("Original columns:", data.columns.tolist())

# STEP 2: Clean & Standardize Column Names
data.columns = data.columns.str.strip().str.upper()

# Rename to match desired names
data.rename(columns={
    "SODIUM LEVEL": "SODIUM",
    "POTASSIUM LEVEL": "POTASSIUM",
    "WT": "WEIGHT",
    "HT_(CM)": "HEIGHT"
}, inplace=True)

print("Cleaned columns:", data.columns.tolist())

# STEP 3: Classification Function for eGFR
def classify_egfr(egfr):
    if pd.isnull(egfr): return "Unknown"
    elif egfr > 90: return "Normal"
    elif egfr > 60: return "Early"
    elif egfr >= 30: return "Middle"
    else: return "End"

# STEP 4: Apply Classification
data["egfr_stage"] = data["EGFR"].apply(classify_egfr)

# STEP 5: Filter valid rows
data = data[data["egfr_stage"] != "Unknown"]

# STEP 6: Keep only the 5 features you want
X = data[["AGE", "SEX", "HEIGHT", "WEIGHT", "EGFR"]]
y = data["egfr_stage"]

# STEP 7: Encode SEX
X = pd.get_dummies(X, columns=["SEX"])

# STEP 8: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# STEP 9: Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# STEP 10: Save Model
joblib.dump(rf_model, "kidocare_rf_model.pkl")

print("Model trained and saved as kidocare_rf_model.pkl")
print("Training features:", list(X.columns))