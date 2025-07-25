import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import func
import matplotlib.pyplot as plt

ds, info = tfds.load("tf_flowers", with_info=True, as_supervised=True)
train_ds = ds["train"]  # tf.data.Dataset object

labels = info.features["label"].names

metadata = []
for i, (image, label) in enumerate(train_ds):
    label_idx = int(label.numpy())
    metadata.append(
        {
            "id": i,
            "label_id": label_idx,
            "label_name": labels[label_idx],
            "image_shape": image.shape.as_list(),
        }
    )

df_metadata = pd.DataFrame(metadata)
# print(df_metadata.head())

Base = declarative_base()


class ImageMeta(Base):
    __tablename__ = "image_meta"
    id = Column(Integer, primary_key=True)
    label_id = Column(Integer)
    label_name = Column(String)
    image_shape = Column(String)  # store as JSON string or string repr


# Define DB (DataBase).
engine = create_engine("sqlite:///tf_flowers_metadata.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Fill the DB.
for row in metadata:
    img_meta = ImageMeta(
        id=row["id"],
        label_id=row["label_id"],
        label_name=row["label_name"],
        image_shape=str(row["image_shape"]),
    )
    session.merge(img_meta)  # merge to avoid duplicates if rerunning

session.commit()

counts = (
    session.query(ImageMeta.label_name, func.count(ImageMeta.id))
    .group_by(ImageMeta.label_name)
    .all()
)
