from flower_classifier_db.database import get_engine, get_session, ImageMetadata
from sqlalchemy import distinct


def get_label_to_id():
    engine = get_engine()
    session = get_session(engine)
    labels = session.query(distinct(ImageMetadata.label)).all()
    label_list = [label[0] for label in labels]
    label_list.sort()
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    return label_to_id


def get_id_to_label():
    label_to_id = get_label_to_id()
    id_to_label = {v: k for k, v in label_to_id.items()}
    return id_to_label
