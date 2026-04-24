from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///./researchmind.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    import db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if "sessions" in inspector.get_table_names():
        session_columns = {column["name"] for column in inspector.get_columns("sessions")}
        if "current_version" not in session_columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE sessions ADD COLUMN current_version INTEGER NOT NULL DEFAULT 1")
                )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
