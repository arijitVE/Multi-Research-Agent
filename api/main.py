import asyncio
import json

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from db.database import get_db, init_db
from services.research_service import improve_report, run_research, send_chat_message
from services.session_service import delete_session, get_all_sessions, get_report_versions, get_session


app = FastAPI(title="ResearchMind API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


class ResearchRequest(BaseModel):
    topic: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    report: str


class ImproveRequest(BaseModel):
    session_id: str
    instructions: str = ""
    use_critic: bool = False


@app.post("/research")
async def research_endpoint(req: ResearchRequest):
    async def event_generator():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def callback(step, status, extra=None):
            data = {"step": step, "status": status}
            if extra:
                data.update(extra)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"event": "progress", "data": data},
            )

        async def run():
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: run_research(req.topic, progress_callback=callback),
                )
                await queue.put(
                    {
                        "event": "done",
                        "data": {
                            "session_id": result["session_id"],
                            "query_type": result["state"].get("query_type", "RESEARCH"),
                        },
                    }
                )
            except Exception as e:
                await queue.put({"event": "error", "data": {"message": str(e)}})

        yield f"data: {json.dumps({'event': 'start', 'data': {'topic': req.topic}})}\n\n"
        asyncio.create_task(run())

        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["event"] in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions")
def list_sessions(db: DBSession = Depends(get_db)):
    del db
    sessions = get_all_sessions()
    return [
        {
            "id": session.id,
            "topic": session.topic,
            "ts": session.ts.isoformat(),
            "query_type": session.query_type,
            "current_version": session.current_version,
            "message_count": len(session.messages),
        }
        for session in sessions
    ]


@app.get("/sessions/{session_id}")
def fetch_session(session_id: str, db: DBSession = Depends(get_db)):
    del db
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": session.id,
        "topic": session.topic,
        "ts": session.ts.isoformat(),
        "query_type": session.query_type,
        "current_version": session.current_version,
        "report": session.report,
        "feedback": session.feedback,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in session.messages
        ],
    }


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str, db: DBSession = Depends(get_db)):
    del db
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


@app.post("/chat")
def chat_endpoint(req: ChatRequest, db: DBSession = Depends(get_db)):
    del db
    reply = send_chat_message(req.session_id, req.question, req.report)
    return {"reply": reply}


@app.post("/sessions/{session_id}/improve")
def improve_endpoint(session_id: str, req: ImproveRequest, db: DBSession = Depends(get_db)):
    del db
    try:
        result = improve_report(
            session_id=session_id,
            instructions=req.instructions,
            use_critic=req.use_critic,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/versions")
def list_versions(session_id: str, db: DBSession = Depends(get_db)):
    del db
    versions = get_report_versions(session_id)
    return [
        {
            "version_number": version.version_number,
            "evaluator_score": version.evaluator_score,
            "evaluator_passed": version.evaluator_passed,
            "improvement_prompt": version.improvement_prompt,
            "created_at": version.created_at.isoformat(),
            "report_preview": version.report[:200] + "..." if version.report else "",
        }
        for version in versions
    ]


@app.get("/health")
def health(db: DBSession = Depends(get_db)):
    del db
    return {"status": "ok", "version": "2.0"}
