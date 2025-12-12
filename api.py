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
import asyncio
from enum import Enum
import pandas as pd


app = FastAPI(
    title="Hopular API",
    description="API for running the complete Supabase fetch, preprocess, train and inference pipeline",
    version="1.0.0"
)


# Request/Response Models
class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunTrainRequest(BaseModel):
    """
    Request model for the run_train pipeline
    """
    supabase_url: str
    supabase_key: str
    table_name: str
    target_column: str
    data: Optional[str] = "data.csv"
    output_csv: Optional[str] = "processed_data.csv"
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 32
    patience: Optional[int] = 10
    test_size: Optional[float] = 0.2
    min_class_samples: Optional[int] = 2


class RunTrainResponse(BaseModel):
    """
    Response model for the run_train pipeline
    """
    job_id: str
    status: PipelineStatus
    message: str
    created_at: datetime
    data: Optional[str] = None
    output_csv: Optional[str] = None
    model_path: Optional[str] = None
    metadata_path: Optional[str] = None


class RunTrainStatusResponse(BaseModel):
    """
    Response model for checking pipeline status
    """
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
    """
    Request model for inference
    """
    model_path: Optional[str] = "best_hopular_model.pt"
    metadata_path: Optional[str] = "metadata.pkl"
    input_data: List[Dict[str, Any]]  # List of records to predict
    target_column: Optional[str] = None  # Column to exclude from prediction


class InferenceResponse(BaseModel):
    """
    Response model for inference
    """
    predictions: List[Any]
    input_count: int
    model_path: str
    metadata_path: str
    processed_at: datetime


# In-memory storage for job status (in production, use a database)
jobs = {}


def run_command_async(cmd: list, description: str, job_id: str):
    """
    Run a subprocess command asynchronously and update job status
    """
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


def run_train_pipeline_async(request: RunTrainRequest, job_id: str):
    """
    Execute the full pipeline asynchronously using subprocess
    """
    jobs[job_id]["status"] = PipelineStatus.RUNNING
    jobs[job_id]["message"] = "Pipeline started"

    try:
        # Step 1: Fetch data from Supabase
        fetch_cmd = [
            sys.executable, "fetch_from_supabase.py",
            "--url", request.supabase_url,
            "--key", request.supabase_key,
            "--table", request.table_name,
            "--output", request.data
        ]

        if request.filters:
            fetch_cmd.extend(["--filters", request.filters])
        if request.limit:
            fetch_cmd.extend(["--limit", str(request.limit)])

        fetch_success = run_command_async(fetch_cmd, "Fetching data from Supabase", job_id)

        if not fetch_success:
            jobs[job_id]["status"] = PipelineStatus.FAILED
            jobs[job_id]["message"] = "Failed to fetch data from Supabase"
            return

        # Step 2: Preprocess the data
        preprocess_cmd = [
            sys.executable, "preprocessing.py",
            "--input", request.data,
            "--output", request.output_csv
        ]

        preprocess_success = run_command_async(preprocess_cmd, "Preprocessing data", job_id)

        if not preprocess_success:
            jobs[job_id]["status"] = PipelineStatus.FAILED
            jobs[job_id]["message"] = "Failed to preprocess data"
            return

        # Step 3: Train the model
        train_cmd = [
            sys.executable, "trainer.py",
            "--data", request.output_csv,
            "--target", request.target_column,
            "--epochs", str(request.epochs),
            "--batch", str(request.batch_size),
            "--patience", str(request.patience),
            "--test_size", str(request.test_size),
            "--min_class_samples", str(request.min_class_samples)
        ]

        train_success = run_command_async(train_cmd, "Training the Hopular model", job_id)

        if not train_success:
            jobs[job_id]["status"] = PipelineStatus.FAILED
            jobs[job_id]["message"] = "Failed to train the model"
            return

        # Update job with output files
        jobs[job_id]["data"] = request.data
        jobs[job_id]["output_csv"] = request.output_csv
        jobs[job_id]["model_path"] = "best_hopular_model.pt"
        jobs[job_id]["metadata_path"] = "metadata.pkl"
        jobs[job_id]["status"] = PipelineStatus.COMPLETED
        jobs[job_id]["message"] = "Pipeline completed successfully!"
        jobs[job_id]["completed_at"] = datetime.now()

        print(f"Job {job_id}: Pipeline completed successfully!")

    except Exception as e:
        jobs[job_id]["status"] = PipelineStatus.FAILED
        jobs[job_id]["message"] = f"Pipeline failed with exception: {str(e)}"
        print(f"Job {job_id}: Exception occurred: {str(e)}")


def run_inference(input_data: List[Dict[str, Any]], model_path: str, metadata_path: str, target_column: Optional[str] = None):
    """
    Run inference on the input data using the trained model
    """
    try:
        # Import the inference module
        from bin.inference import HopularInference
        import pandas as pd

        # Create a temporary CSV file to store input data
        temp_input_file = f"temp_inference_input_{uuid.uuid4()}.csv"

        # Convert input_data to DataFrame
        df = pd.DataFrame(input_data)

        # If target column is specified, remove it from the input data for prediction
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])

        # Save to temporary CSV file
        df.to_csv(temp_input_file, index=False)

        # Prepare output file name
        temp_output_file = f"temp_inference_output_{uuid.uuid4()}.csv"

        # Create inference instance
        hopular_inf = HopularInference(
            model_path=model_path,
            metadata_path=metadata_path
        )

        # Make predictions
        predictions = hopular_inf.predict(df)

        # Clean up temporary input file
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)

        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            "input_count": len(input_data),
            "success": True
        }

    except Exception as e:
        # Clean up temporary files if they exist
        if 'temp_input_file' in locals() and os.path.exists(temp_input_file):
            os.remove(temp_input_file)

        return {
            "predictions": [],
            "input_count": 0,
            "success": False,
            "error": str(e)
        }


@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running
    """
    return {"message": "Hopular API is running", "version": "1.0.0"}


@app.post("/run_train", response_model=RunTrainResponse, status_code=202)
async def run_train(request: RunTrainRequest):
    """
    Start the run_train pipeline asynchronously
    """
    job_id = str(uuid.uuid4())

    # Store job info
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

    # Run the pipeline in a separate thread
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
    """
    Check the status of a run_train job
    """
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
    """
    Run inference on input data using a trained model
    """
    # Check if model and metadata files exist
    if not os.path.exists(request.model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {request.model_path}"
        )

    if not os.path.exists(request.metadata_path):
        raise HTTPException(
            status_code=404,
            detail=f"Metadata file not found: {request.metadata_path}"
        )

    if not request.input_data:
        raise HTTPException(
            status_code=400,
            detail="Input data is required"
        )

    # Run inference
    result = run_inference(
        input_data=request.input_data,
        model_path=request.model_path,
        metadata_path=request.metadata_path,
        target_column=request.target_column
    )

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {result.get('error', 'Unknown error')}"
        )

    return InferenceResponse(
        predictions=result["predictions"],
        input_count=result["input_count"],
        model_path=request.model_path,
        metadata_path=request.metadata_path,
        processed_at=datetime.now()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)