import streamlit as st
import pandas as pd
from flower_classifier_db.database import (
    get_session,
    get_engine,
    ImageMetadata,
    Prediction,
)
from sqlalchemy import select
from PIL import Image
from pathlib import Path

engine = get_engine()
session = get_session(engine)

st.title("Flower classifier")

total_preds = session.query(Prediction).count()
st.metric("Total predictions", total_preds)

# Query misclassified predictions
misclassified = (
    session.execute(
        select(Prediction).where(Prediction.true_label != Prediction.predicted_label)
    )
    .scalars()
    .all()
)


misclassified_data = []
for pred in misclassified:
    image = session.query(ImageMetadata).filter_by(id=pred.image_id).first()
    if image:
        misclassified_data.append(
            {
                "path": image.path,
                "true_label": pred.true_label,
                "predicted_label": pred.predicted_label,
                "confidence": pred.confidence,
            }
        )

df = pd.DataFrame(misclassified_data)

data = []
images = session.query(ImageMetadata).all()
for image in images:
    data.append(
        {
            "label": image.label,
            "path": image.path,
        }
    )


st.subheader("Missclassified images")

n_show = st.slider("Number of examples", min_value=1, max_value=10, value=5)

for i, row in df.head(n_show).iterrows():
    st.image(str(Path("data") / row["path"]), width=200)
    st.write(
        f"**True label**: {row['true_label']} — **Predicted**: {row['predicted_label']} — **Confidence**: {row['confidence']:.2f}"
    )
    st.write(row.path)
    st.markdown("---")
