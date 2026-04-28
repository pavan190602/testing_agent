"""
AutoAgent — FastAPI Backend
REST endpoints for the ML pipeline.
"""

import os
import tempfile
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="AutoAgent API",
    description="Autonomous ML Engineering System — REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineResult(BaseModel):
    success: bool
    eda_report: Optional[dict] = None
    feature_report: Optional[dict] = None
    model_report: Optional[dict] = None
    messages: list = []
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "AutoAgent API"}


@app.post("/run", response_model=PipelineResult)
async def run_pipeline(
    file: UploadFile = File(...),
    problem_description: str = Form(...),
    target_column: str = Form(...),
    task_type: str = Form("classification"),
):
    """
    Run the full AutoAgent pipeline on the uploaded dataset.
    Returns EDA report, feature engineering report, and model selection report.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")

    content = await file.read()
    suffix = "." + file.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from orchestrator import run_pipeline as _run

        final_state = _run(
            file_path=tmp_path,
            problem_description=problem_description,
            target_column=target_column,
            task_type=task_type,
        )

        messages = [m.get("content", "") for m in final_state.get("messages", [])]

        return PipelineResult(
            success=not bool(final_state.get("error")),
            eda_report=final_state.get("eda_report"),
            feature_report=final_state.get("feature_report"),
            model_report=final_state.get("model_report"),
            messages=messages,
            error=final_state.get("error"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)


@app.get("/")
def root():
    return {
        "name": "AutoAgent",
        "description": "Autonomous ML Engineering System",
        "docs": "/docs",
        "endpoints": ["/health", "/run"],
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
