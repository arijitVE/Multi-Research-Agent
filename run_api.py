import uvicorn
import sys

is_windows = sys.platform == "win32"

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["*.log", "*.db", "__pycache__"],
        workers=1
    )