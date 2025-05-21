# FormIQ: Intelligent Document Parser

FormIQ is an advanced document parsing system that leverages foundation models for intelligent information extraction from semi-structured documents like invoices, receipts, and academic forms.

## Features

- Layout-aware document understanding using LayoutLMv3
- Zero-shot field validation with GPT-4 Turbo
- Comprehensive MLOps pipeline with monitoring and versioning
- Real-time document processing and validation
- Interactive web interface for document upload and results visualization

## Architecture

### Core Components
- **Document Processing Pipeline**: LayoutLMv3 for layout-aware understanding
- **Validation Engine**: GPT-4 Turbo for intelligent field validation
- **MLOps Infrastructure**: DVC, Hydra, MLflow/W&B integration
- **Deployment**: AWS SageMaker endpoints with auto-scaling
- **Monitoring**: Evidently AI for drift detection and model monitoring

### Tech Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Cloud**: AWS (S3, SageMaker, CloudWatch)
- **MLOps**: DVC, Hydra, MLflow/W&B, GitHub Actions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/formiq.git
cd formiq
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Project Structure

```
formiq/
├── src/
│   ├── models/           # ML model implementations
│   ├── api/             # FastAPI backend
│   ├── frontend/        # Streamlit UI
│   ├── utils/           # Utility functions
│   └── config/          # Hydra configurations
├── tests/               # Unit and integration tests
├── notebooks/           # Jupyter notebooks for exploration
├── data/               # Dataset storage (DVC managed)
├── mlruns/             # MLflow tracking
└── docs/               # Documentation
```

## Usage

1. Start the backend server:
```bash
uvicorn src.api.main:app --reload
```

2. Launch the frontend:
```bash
streamlit run src/frontend/app.py
```

3. Access the web interface at `http://localhost:8501`

## Model Training

1. Prepare your dataset:
```bash
python src/scripts/prepare_dataset.py
```

2. Train the model:
```bash
python src/scripts/train.py
```

3. Monitor training:
```bash
tensorboard --logdir=logs/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LayoutLMv3 team for the base model
- Roboflow for annotation tools
- AWS for cloud infrastructure 
>>>>>>> 8a53702 (Initial commit of FormIQ Intelligent Document Parser)
