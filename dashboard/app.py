import streamlit as st
import pandas as pd
import duckdb
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Add the src/ folder to Python's import path -- for deployment.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

DB_PATH = "data/flowers.db"

st.title("Performance of the flower classifier")

st.write(
    """
    The goal of the neural network presented therein is to classify correctly flower images.
    As this project is intended to be small, we focus on a rather small dataset that contain
    only 5 different flower species. 
    Choosing a small dataset and a simple task allows us to focus on the tools (packages, etc.) rather
    idiosyncratic details of a given dataset.

    Our classifier is based on a [ResNet 18](https://pytorch.org/hub/pytorch_vision_resnet/) trained on the 
    [tf_flower](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset. 

    Below are the training logs of the model.
    """
)

query = f"""
SELECT 
    epoch,
    train_loss AS training,
    val_loss AS validation
    FROM sqlite_scan('{DB_PATH}', 'training_logs')
"""
df_log = duckdb.sql(query).df()

query = f"""
SELECT 
    p.true_label, 
    p.predicted_label, 
    p.epoch,
    p.confidence
FROM sqlite_scan('{DB_PATH}', 'predictions') p
"""
df = duckdb.sql(query).df()
epoch_list = sorted(df["epoch"].unique())
selected_epoch = st.selectbox("**Select an epoch**", epoch_list)
df_epoch = df[df["epoch"] == selected_epoch]

# Accuracy metrics.
true = df_epoch["true_label"]
pred = df_epoch["predicted_label"]
labels = sorted(df["true_label"].unique())
accuracy = (true == pred).mean()
misclassified_pct = 1 - accuracy
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")
with col2:
    st.metric("Misclassification Rate", f"{misclassified_pct:.2%}")

# Training logs.
fig = go.Figure()  # Empty canva.
fig.add_trace(
    go.Scatter(
        x=df_log["epoch"],
        y=df_log["training"],
        mode="lines+markers",
        name="Training Loss",
    )
)
fig.add_trace(
    go.Scatter(
        x=df_log["epoch"],
        y=df_log["validation"],
        mode="lines+markers",
        name="Validation Loss",
    )
)
fig.add_vline(
    x=selected_epoch,
    line_width=2,
    line_color="white",
)
fig.update_layout(
    xaxis_title="Epoch",
    yaxis_title="Loss",
    legend=dict(
        x=0.80,
        y=0.98,
    ),
    margin=dict(l=40, r=40, t=40, b=40),
    height=400,
)
st.plotly_chart(fig)


# Confusion matrix.
cm = confusion_matrix(true, pred, labels=labels)
fig_cm = px.imshow(
    cm,
    x=labels,
    y=labels,
    labels=dict(x="Predicted", y="True", color="Count"),
    text_auto=True,
    title="Confusion Matrix (validation data)",
)
st.plotly_chart(fig_cm)


# Add a column to flag whether prediction was correct
df_epoch["correct"] = df_epoch["true_label"] == df_epoch["predicted_label"]

# Map to friendly labels
df_epoch["result"] = df_epoch["correct"].map({True: "Correct", False: "Incorrect"})

# Plot histogram
fig_hist = px.histogram(
    df_epoch,
    x="confidence",
    color="result",
    barmode="overlay",  # overlay or group depending on your preference
    nbins=20,
    histnorm="probability density",  # Show density instead of count
    labels={"confidence": "Model Confidence", "result": "Prediction Result"},
    title="Confidence Distribution: Correct vs. Incorrect Predictions",
    opacity=0.6,
)

st.plotly_chart(fig_hist)
