# movie_recommender_api/database.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, JSONB # Use JSONB for better performance with JSON data
from datetime import datetime, timezone # Import timezone for timezone-aware datetime
from typing import Generator

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Crucial: Check if DATABASE_URL is set
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please set it in your .env file.")

# Create the SQLAlchemy engine
# pool_pre_ping=True helps maintain healthy connections in the pool
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# MetaData object is a container for Table objects
metadata = MetaData()

# Define tables using SQLAlchemy Core's Table object
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, index=True, autoincrement=True),
    Column("username", String, unique=True, index=True, nullable=False),
)

user_profiles_table = Table(
    "user_profiles",
    metadata,
    Column("id", Integer, primary_key=True, index=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), unique=True, nullable=False),
    # ARRAY(Float) for storing embedding vectors (NumPy arrays)
    Column("profile_vector", ARRAY(Float), nullable=True),
)

user_memory_entries_table = Table(
    "user_memory_entries",
    metadata,
    Column("id", Integer, primary_key=True, index=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("movie_title", String, nullable=False),
    # ARRAY(Float) for storing movie embeddings in memory
    Column("movie_embedding", ARRAY(Float), nullable=False),
    # Use timezone-aware datetime for robust timestamps
    Column("timestamp", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False),
)

user_genre_preferences_table = Table(
    "user_genre_preferences",
    metadata,
    Column("id", Integer, primary_key=True, index=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), unique=True, nullable=False),
    # JSONB for storing the genre preferences dictionary (more efficient than JSON)
    Column("preferences_data", JSONB, nullable=True),
)

def get_db_connection() -> Generator:
    """
    Dependency to provide a database connection to API endpoints.
    Ensures the connection is closed after the request.
    """
    with engine.connect() as connection:
        yield connection

def create_db_tables():
    """Creates all defined tables in the database if they don't exist."""
    print("Creating database tables if they don't exist...")
    metadata.create_all(engine)
    print("Database table creation process complete.")

# Optional: Function to drop all tables (for development/testing)
def drop_db_tables():
    """Drops all defined tables in the database."""
    print("Dropping all database tables...")
    metadata.drop_all(engine)
    print("Database tables dropped.")