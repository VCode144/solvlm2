from sqlalchemy import create_engine, Column, Integer, Float, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class FrameMetadata(Base):
    __tablename__ = "frame_metadata"
    id = Column(Integer, primary_key=True)
    timestamp = Column(Float)
    frame_number = Column(Integer)
    description = Column(Text)
    alert = Column(String, nullable=True)
    video_id = Column(String)  


# DB setup
engine = create_engine("sqlite:///metadata.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
