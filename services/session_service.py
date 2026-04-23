from sqlalchemy.orm import selectinload

from db.database import SessionLocal
from db.models import Message, Session as ResearchSession


def create_session(topic, query_type):
    db = SessionLocal()
    try:
        session = ResearchSession(topic=topic, query_type=query_type)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    finally:
        db.close()


def update_session(session_id, report=None, feedback=None):
    db = SessionLocal()
    try:
        session = db.query(ResearchSession).filter(ResearchSession.id == session_id).first()
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        if report is not None:
            session.report = report
        if feedback is not None:
            session.feedback = feedback
        db.commit()
        db.refresh(session)
        return session
    finally:
        db.close()


def save_message(session_id, role, content):
    db = SessionLocal()
    try:
        message = Message(session_id=session_id, role=role, content=content)
        db.add(message)
        db.commit()
        db.refresh(message)
        return message
    finally:
        db.close()


def get_all_sessions():
    db = SessionLocal()
    try:
        return (
            db.query(ResearchSession)
            .options(selectinload(ResearchSession.messages))
            .order_by(ResearchSession.ts.desc())
            .all()
        )
    finally:
        db.close()


def get_session(session_id):
    db = SessionLocal()
    try:
        return (
            db.query(ResearchSession)
            .options(selectinload(ResearchSession.messages))
            .filter(ResearchSession.id == session_id)
            .first()
        )
    finally:
        db.close()


def delete_session(session_id):
    db = SessionLocal()
    try:
        session = db.query(ResearchSession).filter(ResearchSession.id == session_id).first()
        if session is None:
            return False
        db.delete(session)
        db.commit()
        return True
    finally:
        db.close()


def get_chat_history(session_id):
    db = SessionLocal()
    try:
        messages = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return [{"role": message.role, "content": message.content} for message in messages]
    finally:
        db.close()


def clear_chat_history(session_id):
    db = SessionLocal()
    try:
        db.query(Message).filter(Message.session_id == session_id).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()
