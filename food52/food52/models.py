from sqlalchemy import create_engine, Column, DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from sqlalchemy.types import ARRAY

import settings

DeclarativeBase = declarative_base()

def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(URL(**settings.DATABASE))

def create_recipes_table(engine):
    DeclarativeBase.metadata.create_all(engine)

class RecipeModel(DeclarativeBase):
    """
    Sqlalchemy recipes model
    """
    __tablename__ = "recipes"

    url = Column('id', String, primary_key=True)
    title = Column(String)
    author = Column(String)
    author_url = Column(String)
    date = Column(DateTime)
    n_comments = Column(Integer)
    comment_dates = Column(ARRAY(String), nullable=True)
    rating = Column(Float, nullable=True)
    n_ratings = Column(Integer, nullable=True)
    n_faves = Column(Integer)
    ingredients = Column(String)
    tags = Column(ARRAY(String), nullable=True)
