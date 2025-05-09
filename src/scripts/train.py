import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_dataset
import mlflow
import wandb
from pathlib import Path
import logging
from typing import Dict, Any
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormIQTrainer:
    def __init__(self, config: DictConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = torch.device(config.model.device)
        
        # Initialize model and processor
        self.processor = LayoutLMv3Processor.from_pretrained(config.model.name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            config.model.name,
            num_labels=config.model.num_labels
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup MLflow and W&B logging."""
        if self.config.logging.mlflow.enabled:
            mlflow.set_tracking_uri(self.config.logging.mlflow.tracking_uri)
            mlflow.set_experiment(self.config.logging.mlflow.experiment_name)
            
        if self.config.logging.wandb.enabled:
            wandb.init(
                project=self.config.logging.wandb.project,
                entity=self.config.logging.wandb.entity,
                config=OmegaConf.to_container(self.config, resolve=True)
            )
    
    def prepare_dataset(self):
        """Prepare the dataset for training."""
        # TODO: Implement dataset preparation
        # This is a placeholder implementation
        return None, None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            correct_predictions += (predictions == batch["labels"]).sum().item()
            total_predictions += batch["labels"].numel()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "accuracy": correct_predictions / total_predictions
            })
        
        # Calculate epoch metrics
        metrics = {
            "train_loss": total_loss / len(train_loader),
            "train_accuracy": correct_predictions / total_predictions
        }
        
        return metrics
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            eval_loader: DataLoader for evaluation data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update metrics
                total_loss += loss.item()
                predictions = outputs.logits.argmax(-1)
                correct_predictions += (predictions == batch["labels"]).sum().item()
                total_predictions += batch["labels"].numel()
        
        # Calculate evaluation metrics
        metrics = {
            "eval_loss": total_loss / len(eval_loader),
            "eval_accuracy": correct_predictions / total_predictions
        }
        
        return metrics
    
    def train(self):
        """Train the model."""
        # Prepare datasets
        train_loader, eval_loader = self.prepare_dataset()
        
        # Training loop
        best_eval_loss = float('inf')
        for epoch in range(self.config.training.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            eval_metrics = self.evaluate(eval_loader)
            
            # Log metrics
            metrics = {**train_metrics, **eval_metrics}
            if self.config.logging.mlflow.enabled:
                mlflow.log_metrics(metrics, step=epoch)
            if self.config.logging.wandb.enabled:
                wandb.log(metrics, step=epoch)
            
            # Save best model
            if eval_metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]
                self.save_model("best_model")
            
            # Save checkpoint
            self.save_model(f"checkpoint_epoch_{epoch + 1}")
    
    def save_model(self, name: str):
        """Save the model.
        
        Args:
            name: Name of the saved model
        """
        save_path = Path(self.config.model.save_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        if self.config.logging.mlflow.enabled:
            mlflow.log_artifacts(str(save_path), f"models/{name}")

@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    """Main training function."""
    trainer = FormIQTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 