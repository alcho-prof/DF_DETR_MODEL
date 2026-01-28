"""
RF-DETR Defect Detection API
=============================

FastAPI backend for manufacturing defect detection using:
- DETR (DEtection TRansformer) for feature extraction and object identification
- Random Forest for defect/non-defect classification
- Comprehensive preprocessing for robustness against adversarial attacks and lighting changes

API Endpoints:
--------------
- GET  /                : Web interface (Live Camera + Upload)
- GET  /health          : Health check endpoint
- GET  /model-info      : Get information about the loaded model
- POST /predict         : Classify an image as defect/non-defect
- POST /quality-control : Alias for /predict (as per task specification)

Sample Usage:
-------------
    curl -X POST "http://localhost:8000/predict" \\
         -H "Content-Type: multipart/form-data" \\
         -F "file=@image.jpg"

Response Format:
----------------
    {
        "prediction": "defect",
        "prediction_label": "Defected",
        "object_name": "bottle",
        "confidence": 0.95,
        "is_anomaly": false,
        "explanation": "The system detected a DEFECT in the bottle...",
        "recommendations": ["Flag product for manual inspection"],
        "status": "Success"
    }

Author: RF-DETR Defect Detection System
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import numpy as np
import cv2
import uvicorn
import traceback

from app.inference import get_inference_service

# Get the directory of this file
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")


# ============================================================
# FastAPI Application Setup
# ============================================================

app = FastAPI(
    title="RF-DETR Defect Detection API",
    description="""
    Manufacturing Defect Detection System using DETR + Random Forest.
    
    ## Features
    - **Object Detection**: Identifies objects in images using DETR
    - **Defect Classification**: Classifies products as Defect/Non-Defect
    - **Adversarial Defense**: Robust against adversarial attacks
    - **Environment Handling**: Handles varying lighting and conditions
    - **Anomaly Detection**: Detects unusual inputs and low-confidence predictions
    
    ## Workflow
    1. Upload an image of a product
    2. The system preprocesses the image for robustness
    3. DETR extracts features and identifies the object
    4. Random Forest classifies as defect or non-defect
    5. Results are interpreted with confidence analysis
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load the inference service
_service = None


def get_service():
    """Get or create the inference service (lazy loading)."""
    global _service
    if _service is None:
        _service = get_inference_service()
    return _service


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", tags=["General"], response_class=HTMLResponse)
def read_root():
    """
    Serve the web interface.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
            <html>
                <body style="font-family: sans-serif; background: #1e293b; color: white; display: flex; align-items: center; justify-content: center; height: 100vh;">
                    <div style="text-align: center;">
                        <h1>RF-DETR Defect Detection API</h1>
                        <p>Web interface not found. Use the API endpoints directly.</p>
                        <p><a href="/docs" style="color: #818cf8;">API Documentation</a></p>
                    </div>
                </body>
            </html>
        """)


@app.get("/health", tags=["General"])
def health_check():
    """
    Health check endpoint for monitoring.
    """
    service = get_service()
    model_loaded = service.model is not None
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "message": "Service is running" if model_loaded else "Model not loaded - run train.py"
    }


@app.get("/api", tags=["General"])
def api_info():
    """
    API information endpoint (JSON format).
    """
    return {
        "name": "RF-DETR Defect Detection API",
        "version": "2.0.0",
        "description": "Manufacturing defect detection using DETR + Random Forest",
        "endpoints": {
            "/": "GET - Web interface (Live Camera + Upload)",
            "/api": "GET - This API info (JSON)",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information",
            "/predict": "POST - Classify an image",
            "/quality-control/": "POST - Alias for /predict",
            "/docs": "GET - Swagger API documentation",
            "/redoc": "GET - ReDoc API documentation"
        },
        "features": [
            "Live camera capture",
            "Image upload",
            "Adversarial attack defense",
            "Dynamic environment handling",
            "Anomaly detection",
            "Defect severity assessment"
        ]
    }


@app.get("/model-info", tags=["Model"])
def get_model_info():
    """
    Get information about the loaded model and configuration.
    """
    service = get_service()
    return service.get_model_info()


@app.post("/predict", tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    detailed: bool = Query(True, description="Return detailed interpretation")
):
    """
    Classify an image as Defected or Non-Defected.
    
    **Process:**
    1. Image is preprocessed for robustness
    2. Features are extracted using DETR
    3. Random Forest classifies the image
    4. Results are interpreted and returned
    
    **Returns:**
    - prediction: Class name (e.g., "defect", "good")
    - prediction_label: Human-readable label ("Defected" or "Non-Defected")
    - object_name: Detected object type
    - confidence: Prediction confidence (0-1)
    - is_anomaly: Whether an anomaly was detected
    - explanation: Human-readable explanation
    - recommendations: List of suggested actions
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        contents = await file.read()
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR format
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please ensure the file is a valid image."
            )
        
        # Run prediction
        service = get_service()
        result = service.predict(img, return_detailed=detailed)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/quality-control/", tags=["Prediction"])
async def quality_control(
    file: UploadFile = File(..., description="Image file to classify")
):
    """
    Quality control endpoint (alias for /predict).
    
    This endpoint matches the sample layout specification:
    
    Input: image
    Output: Defected/Non-defected
    
    **Sample Response:**
    ```json
    {
        "quality_result": {
            "prediction_label": "Defected",
            "confidence": 0.95,
            "object_name": "bottle",
            "is_anomaly": false
        }
    }
    ```
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image bytes
        contents = await file.read()
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )
        
        # Run prediction
        service = get_service()
        result = service.predict(img, return_detailed=True)
        
        # Format as per sample specification
        return {
            "quality_result": {
                "prediction_label": result.get("prediction_label", "Unknown"),
                "prediction": result.get("prediction", "unknown"),
                "confidence": result.get("confidence", 0),
                "object_name": result.get("object_name", "unknown"),
                "is_anomaly": result.get("is_anomaly", False),
                "defect_severity": result.get("defect_severity", "none"),
                "explanation": result.get("explanation", ""),
                "recommendations": result.get("recommendations", [])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status": "Error"
        }
    )


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting RF-DETR Defect Detection API Server")
    print("=" * 60)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
