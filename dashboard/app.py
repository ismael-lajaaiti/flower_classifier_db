import streamlit as st
import pandas as pd
import duckdb
import sys
import os

# Add the src/ folder to Python's import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

DB_PATH = "data/flowers.db"

st.title("Flower classifier")

# Join predictions and images and compute misclassifications
query = f"""
SELECT 
    *,
    CASE WHEN p.true_label != p.predicted_label THEN 1 ELSE 0 END AS is_misclassified
FROM sqlite_scan('{DB_PATH}', 'predictions') p
JOIN sqlite_scan('{DB_PATH}', 'images') i
ON p.image_id = i.id
"""

df = duckdb.sql(query).df()

# Overall misclassification rate.
overall_mis_rate = df["is_misclassified"].mean() * 100
st.metric("Overall Misclassification Rate", f"{overall_mis_rate:.2f}%")

# Select a flower.
labels = sorted(df["true_label"].unique())
selected_label = st.selectbox("Choose a flower class", labels)

# False negatives (true = selected, predicted != selected).
true_class_df = df[df["true_label"] == selected_label]
fn_count = (true_class_df["predicted_label"] != selected_label).sum()
fn_rate = fn_count / len(true_class_df) * 100

# False positives (true != selected, predicted = selected).
pred_class_df = df[df["predicted_label"] == selected_label]
fp_count = (pred_class_df["true_label"] != selected_label).sum()
fp_rate = fp_count / len(pred_class_df) * 100


st.subheader(f"Model performance on *{selected_label}*")

col1, col2 = st.columns(2)
col1.metric(
    "False Negative Rate",
    f"{fn_rate:.2f}%",
    delta=f"{fn_rate - overall_mis_rate:+.2f}%",
    delta_color="inverse",
)
col2.metric(
    "False Positive Rate",
    f"{fp_rate:.2f}%",
    delta=f"{fp_rate - overall_mis_rate:+.2f}%",
    delta_color="inverse",
)
