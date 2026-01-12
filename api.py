"""
FastAPI application for Hopular run_train and inference functionality

This API provides endpoints to:
1. Run the complete pipeline: Fetch data from Supabase, preprocess, and train
2. Make predictions using a trained model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import subprocess
import sys
import os
import uuid
import json
from datetime import datetime
import threading
from enum import Enum
import pandas as pd

from typing import List, Dict, Any, Optional
from bin.inference import HopularInference

app = FastAPI(
    title="Hopular API",
    description="API for running the train and inference pipeline",
    version="1.1.0"
)

# ---------------------------------------------
# Models
# ---------------------------------------------
class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunTrainRequest(BaseModel):
    target_column: str
    data: Optional[str] = "data/data.csv"
    output_csv: Optional[str] = "data/processed_data.csv"
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 32
    patience: Optional[int] = 10
    test_size: Optional[float] = 0.2


class RunTrainResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    message: str
    created_at: datetime
    data: Optional[str] = None
    output_csv: Optional[str] = None
    model_path: Optional[str] = None
    metadata_path: Optional[str] = None


class RunTrainStatusResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    message: str
    progress: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    data: Optional[str] = None
    output_csv: Optional[str] = None
    model_path: Optional[str] = None
    metadata_path: Optional[str] = None


class InferenceRequest(BaseModel):
    model_path: Optional[str] = "best_hopular_model.pt"
    metadata_path: Optional[str] = "metadata.pkl"
    input_data: List[Dict[str, Any]]  # List of records to predict
    target_column: Optional[str] = None  # Column to exclude from prediction


class InferenceResponse(BaseModel):
    predictions: List[Any]
    input_count: int
    model_path: str
    metadata_path: str
    processed_at: datetime


# ---------------------------------------------
# Global storage and cache
# ---------------------------------------------
jobs = {}
_INFERENCE_CACHE = {}


# ---------------------------------------------
# Async command runner
# ---------------------------------------------
def run_command_async(cmd: list, description: str, job_id: str):
    try:
        jobs[job_id]["status"] = PipelineStatus.RUNNING
        jobs[job_id]["message"] = f"Running: {description}"

        print(f"Job {job_id}: {description}")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        jobs[job_id]["message"] = f"Completed: {description}"
        jobs[job_id]["progress"] = description

        if result.stdout:
            print(f"Output:\n{result.stdout}")

        return True
    except subprocess.CalledProcessError as e:
        jobs[job_id]["status"] = PipelineStatus.FAILED
        jobs[job_id]["message"] = f"Failed: {description} - Error: {e.stderr}"
        print(f"Job {job_id}: Command failed with return code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        jobs[job_id]["status"] = PipelineStatus.FAILED
        jobs[job_id]["message"] = f"Failed: {description} - Command not found: {cmd[0]}"
        print(f"Job {job_id}: Command not found: {cmd[0]}")
        return False


# ---------------------------------------------
# Async train pipeline
# ---------------------------------------------
def run_train_pipeline_async(request: RunTrainRequest, job_id: str):
    jobs[job_id]["status"] = PipelineStatus.RUNNING
    jobs[job_id]["message"] = "Pipeline started"

    try:
        # Step 1: Fetch data
        fetch_cmd = [
            sys.executable, "bin/fetch_from_supabase.py",
            "--output", request.data
        ]
        if not run_command_async(fetch_cmd, "Fetching data from Supabase", job_id):
            jobs[job_id]["status"] = PipelineStatus.FAILED
            return

        # Step 2: Preprocess
        preprocess_cmd = [
            sys.executable, "bin/preprocessing.py",
            "--input", request.data,
            "--output", request.output_csv
        ]
        if not run_command_async(preprocess_cmd, "Preprocessing data", job_id):
            jobs[job_id]["status"] = PipelineStatus.FAILED
            return

        # Step 3: Train
        train_cmd = [
            sys.executable, "bin/trainer.py",
            "--data", request.output_csv,
            "--target", request.target_column,
            "--epochs", str(request.epochs),
            "--batch", str(request.batch_size),
            "--patience", str(request.patience),
            "--test_size", str(request.test_size),
        ]
        if not run_command_async(train_cmd, "Training the Hopular model", job_id):
            jobs[job_id]["status"] = PipelineStatus.FAILED
            return

        # Update job info
        jobs[job_id].update({
            "data": request.data,
            "output_csv": request.output_csv,
            "model_path": "output/best_hopular_model.pt",
            "metadata_path": "output/metadata.pkl",
            "status": PipelineStatus.COMPLETED,
            "message": "Pipeline completed successfully!",
            "completed_at": datetime.now()
        })
        print(f"Job {job_id}: Pipeline completed successfully!")

    except Exception as e:
        jobs[job_id]["status"] = PipelineStatus.FAILED
        jobs[job_id]["message"] = f"Pipeline failed with exception: {str(e)}"
        print(f"Job {job_id}: Exception occurred: {str(e)}")


# ---------------------------------------------
# Inference
# ---------------------------------------------
def run_inference(
    input_data: List[Dict[str, Any]],
    model_path: str,
    metadata_path: str,
    target_column: Optional[str] = None
):
    """
    Run inference using HopularInference directly (NO subprocess)
    Only return one recommendation with score >= 0.5
    """
    try:
        cache_key = f"{model_path}|{metadata_path}"
        if cache_key not in _INFERENCE_CACHE:
            _INFERENCE_CACHE[cache_key] = HopularInference(
                model_path=model_path,
                metadata_path=metadata_path
            )
        inference_model = _INFERENCE_CACHE[cache_key]

        predictions = inference_model.predict_with_recommendations(input_data)

        # Filter score >= 0.5
        filtered = [p for p in predictions if p.get("kecocokan", 0) >= 0.5]

        # Ambil 1 tertinggi
        if filtered:
            filtered.sort(key=lambda x: x.get("kecocokan", 0), reverse=True)
            filtered = [filtered[0]]
        else:
            filtered = []

        return {
            "predictions": filtered,
            "input_count": len(input_data),
            "success": True
        }

    except Exception as e:
        return {
            "predictions": [],
            "input_count": len(input_data),
            "success": False,
            "error": str(e)
        }


# ---------------------------------------------
# Endpoints
# ---------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Hopular API is running", "version": "1.1.0"}


@app.post("/run_train", response_model=RunTrainResponse, status_code=202)
async def run_train(request: RunTrainRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": PipelineStatus.PENDING,
        "message": "Pipeline queued",
        "created_at": datetime.now(),
        "progress": None,
        "completed_at": None,
        "data": None,
        "output_csv": None,
        "model_path": None,
        "metadata_path": None
    }
    thread = threading.Thread(
        target=run_train_pipeline_async,
        args=(request, job_id)
    )
    thread.start()
    return RunTrainResponse(
        job_id=job_id,
        status=PipelineStatus.PENDING,
        message="Pipeline queued successfully",
        created_at=jobs[job_id]["created_at"]
    )


@app.get("/run_train/{job_id}", response_model=RunTrainStatusResponse)
async def get_run_train_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return RunTrainStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job["message"],
        progress=job.get("progress"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        data=job.get("data"),
        output_csv=job.get("output_csv"),
        model_path=job.get("model_path"),
        metadata_path=job.get("metadata_path")
    )


@app.post("/inference", response_model=InferenceResponse)
async def run_inference_endpoint(request: InferenceRequest):
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
    if not os.path.exists(request.metadata_path):
        raise HTTPException(status_code=404, detail=f"Metadata file not found: {request.metadata_path}")
    if not request.input_data:
        raise HTTPException(status_code=400, detail="Input data is required")

    result = run_inference(
        input_data=request.input_data,
        model_path=request.model_path,
        metadata_path=request.metadata_path,
        target_column=request.target_column
    )

    if not result["success"]:
        raise HTTPException(status_code=500, detail=f"Inference failed: {result.get('error', 'Unknown error')}")

    return InferenceResponse(
        predictions=result["predictions"],
        input_count=result["input_count"],
        model_path=request.model_path,
        metadata_path=request.metadata_path,
        processed_at=datetime.now()
    )


# ---------------------------------------------
# Run as standalone
# ---------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
