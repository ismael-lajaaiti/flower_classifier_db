from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class ImageMetadata(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)
    width = Column(Integer)
    height = Column(Integer)


def get_engine(db_path="data/flowers.db"):
    return create_engine(f"sqlite:///{db_path}", echo=False)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
