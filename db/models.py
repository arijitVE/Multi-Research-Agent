from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from db.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    topic = Column(String, nullable=False)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False)
    query_type = Column(String, nullable=True)
    report = Column(Text, nullable=True)
    feedback = Column(Text, nullable=True)
    current_version = Column(Integer, default=1, nullable=False)

    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    versions = relationship(
        "ReportVersion",
        back_populates="session",
        order_by="ReportVersion.version_number",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("Session", back_populates="messages")


class ReportVersion(Base):
    __tablename__ = "report_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    report = Column(Text, nullable=False)
    feedback = Column(Text, nullable=True)
    improvement_prompt = Column(Text, nullable=True)
    evaluator_score = Column(Float, nullable=True)
    evaluator_passed = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    session = relationship("Session", back_populates="versions")
