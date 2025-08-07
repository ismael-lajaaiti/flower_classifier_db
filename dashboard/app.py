import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import duckdb
import sys
import os

# Add the src/ folder to Python's import path -- for deployment.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

DB_PATH = "data/flowers.db"

st.title("Performance of the flower classifier")

st.write(
    """
    The goal of the neural network presented therein is to classify correctly flower images.
    As this project is intended to be small, we focus on a rather small dataset that contain
    only 6 different flower species. 
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

df = duckdb.sql(query).df()

# Create the plot.
fig = go.Figure()  # Empty canva.
fig.add_trace(
    go.Scatter(
        x=df["epoch"], y=df["training"], mode="lines+markers", name="Training Loss"
    )
)
fig.add_trace(
    go.Scatter(
        x=df["epoch"], y=df["validation"], mode="lines+markers", name="Validation Loss"
    )
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
