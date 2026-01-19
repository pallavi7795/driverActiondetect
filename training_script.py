import os
import sys
import json
import argparse
import logging
import yaml
import torch
import tarfile
from pathlib import Path

# Import sagemaker's log_metric utility
from sagemaker.session import Session
from time import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define SageMaker paths
BASE_PATH = '/opt/ml'
INPUT_PATH = os.path.join(BASE_PATH, 'input')
MODEL_PATH = os.path.join(BASE_PATH, 'model')
CONFIG_PATH = os.path.join(INPUT_PATH, 'config')

# Data paths - Updated for correct structure
DATA_PATH = os.path.join(INPUT_PATH, 'data')
IMAGES_PATH = os.path.join(DATA_PATH, 'training/images')
LABELS_PATH = os.path.join(DATA_PATH, 'training/labels')

# Model paths
PRETRAINED_PATH = os.path.join(MODEL_PATH, 'pretrained')
OUTPUT_PATH = os.path.join(MODEL_PATH, 'output')
ARTIFACTS_PATH = os.path.join(MODEL_PATH, 'artifacts')

# Expected directory structure should be:
"""
/opt/ml/
├── input/
│   ├── config/
│   │   ├── aws_config.yaml
│   │   └── config.yaml
│   └── data/
│       ├── images/
│       │   └── train/
│       │   └── val/
│       │   ├── test/
│       ├── labels/
│           └── train/
└── model/
    ├── pretrained/
    │   └── yolov8l.pt
    ├── output/
    └── artifacts/
"""

# Get SageMaker environment variables
training_data_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
pretrained_model_dir = os.environ.get('SM_CHANNEL_PRETRAINED', '/opt/ml/input/data/pretrained')
config_dir = os.environ.get('SM_CHANNEL_CONFIG', '/opt/ml/input/data/config')
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
output_dir = os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output')
#config_file= os.environ.get('SM_CHANNEL_CONFIG', '/opt/ml/input/data/config/config.yaml')

def install_requirements():
    """Install required packages"""
    try:
        import subprocess
        logger.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        logger.info("Successfully installed requirements")
    except Exception as e:
        logger.error(f"Error installing requirements: {str(e)}")
        raise

def load_configs():
    """Load both AWS and model configurations"""
    try:
        # Load AWS config
        aws_config_path = os.path.join(config_dir, 'aws_config.yaml')
        with open(aws_config_path, 'r') as f:
            aws_config = yaml.safe_load(f)
            logger.info("Loaded AWS config")

        # Load model config
        model_config_path = os.path.join(config_dir, 'config.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            logger.info("Loaded model config")

        return aws_config, model_config
    except Exception as e:
        logger.error(f"Error loading configs: {str(e)}")
        raise

def create_data_yaml(model_config):
    """Create YOLO training configuration file"""
    try:
        logger.info("Creating data.yaml for YOLO training...")
        aws_config, model_config = load_configs()
        
        data_yaml = {
            'path': training_data_dir,
            'train': os.path.join('images'),
            'val': os.path.join('images'),  # Same as train for now
            'test': os.path.join('images'),  # Same as train for now
            'nc': model_config['nc'],
            'names': model_config['names']
        }

        data_yaml_path = os.path.join(DATA_PATH, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        logger.info(f"Created data.yaml at: {data_yaml_path}")
        return data_yaml_path
    except Exception as e:
        logger.error(f"Error creating data.yaml: {str(e)}")
        raise

def verify_dataset_structure():
    """Verify dataset directory structure"""
    try:
        dataset_info = {'images': {}, 'labels': {}}
        splits = ['train', 'validation', 'test']
        
        # Verify base directories exist
        if not os.path.exists(IMAGES_PATH):
            raise FileNotFoundError(f"Images directory not found at: {IMAGES_PATH}")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels directory not found at: {LABELS_PATH}")
        
        # Check each split
        for split in splits:
            # Check split directories
            img_split_path = os.path.join(IMAGES_PATH, split)
            label_split_path = os.path.join(LABELS_PATH, split)
            
            if not os.path.exists(img_split_path):
                raise FileNotFoundError(f"Images {split} directory not found at: {img_split_path}")
            if not os.path.exists(label_split_path):
                raise FileNotFoundError(f"Labels {split} directory not found at: {label_split_path}")
            
            # Count files
            img_files = [f for f in os.listdir(img_split_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            label_files = [f for f in os.listdir(label_split_path) 
                          if f.endswith('.txt')]
            
            dataset_info['images'][split] = len(img_files)
            dataset_info['labels'][split] = len(label_files)
            
            logger.info(f"{split} split: {len(img_files)} images, {len(label_files)} labels")
            
            # Verify matching files
            img_basenames = {os.path.splitext(f)[0] for f in img_files}
            label_basenames = {os.path.splitext(f)[0] for f in label_files}
            
            missing_labels = img_basenames - label_basenames
            missing_images = label_basenames - img_basenames
            
            if missing_labels:
                logger.warning(f"Missing labels for images in {split}: {missing_labels}")
            if missing_images:
                logger.warning(f"Missing images for labels in {split}: {missing_images}")
        
        return dataset_info
    except Exception as e:
        logger.error(f"Dataset verification failed: {str(e)}")
        raise

def save_model_artifacts(model_path):
    """Save model artifacts in SageMaker format"""
    try:
        # Ensure artifacts directory exists
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)
        
        # Create model archive
        output_path = os.path.join(ARTIFACTS_PATH, 'model.tar.gz')
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(model_path, arcname=os.path.basename(model_path))
        logger.info(f"Model artifacts saved to: {output_path}")
        
    except Exception as e:
        
        logger.error(f"Error saving model artifacts: {str(e)}")
        
        raise
        
def save_metrics(results):
    """Save metrics using mean_results method"""
    try:
        # Get results as list using mean_results()
        mean_vals = results.mean_results()
        
        # Map the values to metrics (based on YOLO's output order)
        metrics = {
            'precision': float(mean_vals[0]),  # First value is precision
            'recall': float(mean_vals[1]),     # Second value is recall
            'mAP50': float(mean_vals[2]),      # Third value is mAP50
            'mAP50-95': float(mean_vals[3])    # Fourth value is mAP50-95
        }
        
        # Log metrics in SageMaker format
        for name, value in metrics.items():
            print(f"metrics/{name}(B):{value:.4f};")
            logger.info(f"{name}: {value:.4f}")
        
        # Save for SageMaker metrics tab
        metrics_path = os.path.join('/opt/ml/output/metrics/sagemaker', 'metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        formatted_metrics = {
            "AutoML": {
                "_timestamp": int(time()),
                **metrics
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(formatted_metrics, f)
        
        # Save to artifacts
        artifacts_path = os.path.join(ARTIFACTS_PATH, 'metrics.json')
        with open(artifacts_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
        
    except Exception as e:
        # This block appears twice, which could cause confusion
        logger.error(f"Error saving metrics: {str(e)}")
        try:
            if hasattr(results, 'mean_results'):
                mean_vals = results.mean_results()
                logger.error(f"Mean results: {mean_vals}")
            if hasattr(results, 'results_dict'):
                logger.error(f"Results dict: {results.results_dict}")
        except:
            pass
        raise
        

def train_yolo(args, aws_config, model_config):
    """Train YOLOv8 model"""
    try:
        from ultralytics import YOLO

        # Verify dataset structure
        logger.info("Verifying dataset structure...")
        dataset_info = verify_dataset_structure()

        # Create YOLO data configuration
        data_yaml_path = create_data_yaml(model_config)
        
        # Load pretrained model
        pretrained_model_path = os.path.join(pretrained_model_dir, 'yolov8l.pt')
        logger.info(f"Loading pretrained model from: {pretrained_model_path}")
        model = YOLO(pretrained_model_path)
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Train model
        logger.info("\nStarting training...")
        results = model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            imgsz=args.image_size,
            batch=args.batch_size,
            device=device,
            workers=args.workers,
            project=OUTPUT_PATH,
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer=args.optimizer,
            lr0=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            dropout=args.dropout,
            patience=args.patience,
            save=True,
            save_period=args.save_period,
        )

        # Save the best model and metrics
        best_model_path = Path(OUTPUT_PATH) / 'train' / 'weights' / 'best.pt'
        if best_model_path.exists():
            save_model_artifacts(str(best_model_path))
            metrics = save_metrics(results)
            
            logger.info("\n Training Results:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            raise FileNotFoundError("Best model weights not found after training")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 model in SageMaker')
    
    # Training parameters
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save-period', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--num-classes', type=int, default=12)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    args = parser.parse_args()

    try:
        # Install requirements
        install_requirements()
        
        # Load configurations
        logger.info("Loading configurations...")
        aws_config, model_config = load_configs()
        
        # Print training configuration
        logger.info("\nTraining Configuration:")
        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")
        
        # Train model
        logger.info("\nStarting model training...")
        
        train_yolo(args, aws_config, model_config)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()