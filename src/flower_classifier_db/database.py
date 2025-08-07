from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class ImageMetadata(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    split = Column(String)


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    train_accuracy = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=False)
    val_accuracy = Column(Float, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    true_label = Column(String, nullable=False)
    predicted_label = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    epoch = Column(Integer, nullable=True)


def get_engine(db_path="data/flowers.db"):
    return create_engine(f"sqlite:///{db_path}", echo=False)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
