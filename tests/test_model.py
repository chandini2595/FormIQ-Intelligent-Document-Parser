import pytest
import torch
from PIL import Image
import numpy as np
from src.models.layoutlm import FormIQModel

@pytest.fixture
def model():
    """Create a model instance for testing."""
    return FormIQModel(device="cpu")

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a random image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image_array)

def test_model_initialization(model):
    """Test model initialization."""
    assert model.device == "cpu"
    assert model.model is not None
    assert model.processor is not None

def test_preprocess_image(model, sample_image):
    """Test image preprocessing."""
    processed = model.preprocess_image(sample_image)
    
    # Check if all required keys are present
    assert "input_ids" in processed
    assert "attention_mask" in processed
    assert "bbox" in processed
    assert "pixel_values" in processed
    
    # Check tensor types and shapes
    assert isinstance(processed["input_ids"], torch.Tensor)
    assert isinstance(processed["attention_mask"], torch.Tensor)
    assert isinstance(processed["bbox"], torch.Tensor)
    assert isinstance(processed["pixel_values"], torch.Tensor)

def test_predict(model, sample_image):
    """Test prediction functionality."""
    results = model.predict(sample_image, confidence_threshold=0.5)
    
    # Check result structure
    assert "fields" in results
    assert "metadata" in results
    assert isinstance(results["fields"], list)
    assert isinstance(results["metadata"], dict)
    
    # Check metadata
    assert "confidence_scores" in results["metadata"]
    assert "model_version" in results["metadata"]

def test_validate_extraction(model):
    """Test field validation."""
    # Create sample extraction results
    sample_extraction = {
        "fields": [
            {"label": "amount", "confidence": 0.95, "value": "100.00"},
            {"label": "date", "confidence": 0.85, "value": "2024-03-20"}
        ]
    }
    
    # Test validation
    validation_results = model.validate_extraction(
        sample_extraction,
        document_type="invoice"
    )
    
    # Check validation results structure
    assert "is_valid" in validation_results
    assert "validation_errors" in validation_results
    assert "confidence_score" in validation_results
    
    # Check types
    assert isinstance(validation_results["is_valid"], bool)
    assert isinstance(validation_results["validation_errors"], list)
    assert isinstance(validation_results["confidence_score"], float)

def test_error_handling(model):
    """Test error handling."""
    # Test with invalid image
    with pytest.raises(Exception):
        model.predict(Image.new("RGB", (0, 0)))
    
    # Test with invalid confidence threshold
    with pytest.raises(Exception):
        model.predict(Image.new("RGB", (224, 224)), confidence_threshold=2.0)
    
    # Test with invalid document type
    with pytest.raises(Exception):
        model.validate_extraction({}, document_type="invalid_type") 