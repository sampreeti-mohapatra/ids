"""
Intrusion Detection System Using Machine Learning
--------------------------------------------------
Dataset : NSL-KDD (KDDTrain+.txt)
Task    : Binary classification of network connections (normal vs attack)
          plus a per-attack-category breakdown (DoS, Probe, R2L, U2R)
Models  : Random Forest, Linear SVM, Logistic Regression

Run:
    pip install -r requirements.txt
    python IDS.py
Outputs are written to the results/ folder.
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # save figures to files; works without a display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

DATA_FILE = "KDDTrain+.txt"
RESULTS_DIR = "results"
RANDOM_STATE = 42

# ==============================
# STEP 1: LOAD DATASET
# ==============================
# The NSL-KDD file has no header row: 41 features, the attack label,
# and a "difficulty" score (how hard the record is to classify).
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Map each specific attack to one of the four standard NSL-KDD categories.
ATTACK_CATEGORY = {
    "normal": "normal",
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    # Probing / scanning
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "named": "R2L", "sendmail": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "worm": "R2L",
    "xlock": "R2L", "xsnoop": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}

if not os.path.exists(DATA_FILE):
    sys.exit(f"Dataset not found: {DATA_FILE}. Place it next to IDS.py.")

data = pd.read_csv(DATA_FILE, header=None, names=COLUMNS)
print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns\n")

# ==============================
# STEP 2: PREPROCESSING
# ==============================
data = data.drop_duplicates().dropna()
data["category"] = data["label"].map(ATTACK_CATEGORY).fillna("other")
data["is_attack"] = (data["label"] != "normal").astype(int)

print("Class distribution:")
print(data["category"].value_counts().to_string(), "\n")

# "difficulty" is metadata about the record, not a network feature,
# so it must not be used as an input.
X = data.drop(columns=["label", "difficulty", "category", "is_attack"])
y = data["is_attack"]

# Constant columns carry no information.
X = X.loc[:, X.nunique() > 1]
numeric = [c for c in X.columns if c not in CATEGORICAL]

# ==============================
# STEP 3: TRAIN-TEST SPLIT
# ==============================
# Split BEFORE scaling so no information from the test set leaks into training.
X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
    X, y, data["category"], test_size=0.2, random_state=RANDOM_STATE,
    stratify=y,
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

# One-hot encode categorical columns, scale numeric columns.
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), numeric),
])

# ==============================
# STEP 4: TRAIN AND EVALUATE MODELS
# ==============================
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
    "SVM (Linear)": LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(max_iter=2000),
}

os.makedirs(RESULTS_DIR, exist_ok=True)
results = []
fitted = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    start = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = pipe.predict(X_test)
    fitted[name] = (pipe, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Train time (s)": train_time,
    })
    print(f"=== {name} (trained in {train_time:.1f}s) ===")
    print(classification_report(y_test, y_pred,
                                target_names=["normal", "attack"], digits=4))

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"),
                  index=False, float_format="%.4f")
print("Model comparison:")
print(results_df.to_string(index=False, float_format="%.4f"), "\n")

# ==============================
# STEP 5: DETECTION RATE PER ATTACK CATEGORY
# ==============================
# How many attacks of each type did each model catch?
rows = []
for name, (_, y_pred) in fitted.items():
    pred = pd.Series(y_pred, index=cat_test.index)
    for cat in ["DoS", "Probe", "R2L", "U2R"]:
        mask = cat_test == cat
        if mask.any():
            rows.append({"Model": name, "Category": cat,
                         "Samples": int(mask.sum()),
                         "Detection rate": pred[mask].mean()})
per_cat = pd.DataFrame(rows).pivot(index="Category", columns="Model",
                                   values="Detection rate")
per_cat.to_csv(os.path.join(RESULTS_DIR, "detection_by_category.csv"),
               float_format="%.4f")
print("Detection rate by attack category:")
print(per_cat.to_string(float_format="%.4f"), "\n")

# ==============================
# STEP 6: PLOTS
# ==============================
# Model comparison (real scores, computed above)
metrics = ["Accuracy", "Precision", "Recall", "F1"]
ax = results_df.set_index("Model")[metrics].plot(kind="bar", figsize=(9, 5))
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Score")
ax.set_title("Comparison of ML Models for Intrusion Detection (NSL-KDD)")
ax.tick_params(axis="x", rotation=0)
ax.set_xlabel("")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
plt.close()

# Confusion matrix of the best model
best = results_df.iloc[0]["Model"]
ConfusionMatrixDisplay.from_predictions(
    y_test, fitted[best][1], display_labels=["normal", "attack"],
    cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - {best}")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Top features from Random Forest
rf_pipe = fitted["Random Forest"][0]
names = rf_pipe.named_steps["prep"].get_feature_names_out()
importances = pd.Series(
    rf_pipe.named_steps["clf"].feature_importances_, index=names
).sort_values(ascending=False).head(15)
importances.index = [n.split("__", 1)[1] for n in importances.index]
importances[::-1].plot(kind="barh", figsize=(8, 6))
plt.title("Top 15 Features (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ==============================
# STEP 7: CLASSIFY A SINGLE CONNECTION
# ==============================
sample = X_test.iloc[[0]]
pred = fitted[best][0].predict(sample)[0]
print(f"Sample connection (true type: {cat_test.iloc[0]}) -> "
      f"{'MALICIOUS' if pred else 'NORMAL'} (predicted by {best})")
print(f"\nResults and plots saved to '{RESULTS_DIR}/'")
