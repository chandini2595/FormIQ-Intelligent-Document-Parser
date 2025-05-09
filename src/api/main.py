from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from typing import Dict, Any
import json

from src.models.layoutlm import FormIQModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FormIQ API",
    description="Intelligent Document Parser API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = FormIQModel()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "FormIQ API"}

@app.post("/extract")
async def extract_information(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    document_type: str = "invoice"
) -> Dict[str, Any]:
    """Extract information from uploaded document.
    
    Args:
        file: Uploaded document image
        confidence_threshold: Minimum confidence score for predictions
        document_type: Type of document being processed
        
    Returns:
        Dictionary containing extracted fields and metadata
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process image
        extraction_results = model.predict(
            image=image,
            confidence_threshold=confidence_threshold
        )
        
        # Validate extraction
        validation_results = model.validate_extraction(
            extracted_fields=extraction_results,
            document_type=document_type
        )
        
        # Combine results
        response = {
            "extraction": extraction_results,
            "validation": validation_results,
            "metadata": {
                "document_type": document_type,
                "confidence_threshold": confidence_threshold
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the current model."""
    return {
        "model_name": model.model.config.model_type,
        "device": model.device,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 