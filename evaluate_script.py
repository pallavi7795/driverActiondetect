import os
import sys
import json
import logging
import argparse
import subprocess
import yaml
from pathlib import Path
import torch
import tarfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_metrics_for_sagemaker(results):
    """Format metrics in SageMaker-compatible structure"""
    # Check if results is a list (from predict) or has results_dict (from val)
    if isinstance(results, list):
        # Handle prediction results which don't have metrics
        # In this case, we'll set default values
        metrics = {
            "object_detection_metrics": {  #binary_classification_metrics
                "precision": {
                    "value": 0.0,
                    "standard_deviation": "NaN"
                },
                "recall": {
                    "value": 0.0,
                    "standard_deviation": "NaN"
                },
                "mAP50": {
                    "value": 0.0,
                    "standard_deviation": "NaN"
                },
                "mAP50-95": {
                    "value": 0.0,
                    "standard_deviation": "NaN"
                }
            }
        }
    else:
        # Handle validation results with metrics
        metrics = {
            "object_detection_metrics": {
                "precision": {
                    "value": float(results.results_dict.get('metrics/precision(B)', 0)),
                    "standard_deviation": "NaN"
                },
                "recall": {
                    "value": float(results.results_dict.get('metrics/recall(B)', 0)),
                    "standard_deviation": "NaN"
                },
                "mAP50": {
                    "value": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    "standard_deviation": "NaN"
                },
                "mAP50-95": {
                    "value": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    "standard_deviation": "NaN"
                }
            }
        }
    return metrics


def install_requirements():
    """Install required packages"""
    try:
        # Install required packages
        packages = [
            "opencv-python-headless",
            "ultralytics"
        ]
        for package in packages:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--no-cache-dir", 
                package
            ])
        logger.info("Successfully installed all required packages")
    except Exception as e:
        logger.error(f"Error installing requirements: {str(e)}")
        raise


def prepare_model_path(model_path):
    """Prepare model path by handling different scenarios"""
    
    logger.info(f"Original model path: {model_path}")
    
    # If directory, find first .tar.gz or .pt file
    if os.path.isdir(model_path):
        files = [f for f in os.listdir(model_path) if f.endswith(('.tar.gz', '.pt'))]
        if not files:
            raise FileNotFoundError("No .tar.gz or .pt files found in directory")
        model_path = os.path.join(model_path, files[0])
    
    # If tar.gz, extract
    if model_path.endswith('.tar.gz'):
        extract_dir = os.path.join(os.path.dirname(model_path), 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        with tarfile.open(model_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        # Find .pt file
        pt_files = list(Path(extract_dir).rglob('*.pt'))
        if not pt_files:
            raise FileNotFoundError("No .pt file found in archive")
        model_path = str(pt_files[0])
    
    logger.info(f"Prepared model path: {model_path}")
    return model_path


def load_config():
    """Load model configuration from config file"""
    try:
        config_path = '/opt/ml/processing/input/config/config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Successfully loaded model config")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise


def evaluate_model():
    """Main evaluation function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-images', type=str, default='/opt/ml/processing/input/images/test')
    parser.add_argument('--test-labels', type=str, default='/opt/ml/processing/input/labels/test')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/evaluation')
    args = parser.parse_args()
    
    try:
        from ultralytics import YOLO
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load config file
        config = load_config()
        logger.info(f"Loaded config with {config['nc']} classes")

        # Prepare model path and load model
        model_path = prepare_model_path(args.model_path)
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        logger.info("Model loaded successfully")

        # Prepare data paths
        base_path = os.path.dirname(args.test_images)
        
        # ----------------------------------------------------------------
        # IMPORTANT NOTE:
        # The YOLO library requires 'train' and 'val' keys in the data.yaml
        # file even when only doing evaluation (no training). This is just
        # a technical requirement of the library.
        #
        # We are NOT using any training data during evaluation. 
        # We are ONLY evaluating the model on your test data.
        #
        # 'train' path points to an empty directory (not used)
        # 'val' path points to your test data (used for evaluation)
        # ----------------------------------------------------------------
        
        data_yaml = {
            'path': base_path,
            'train': os.path.join(os.path.dirname(args.test_images), 'train'),  # Required by YOLO but NOT USED
            'val': os.path.basename(args.test_images),  # This is your TEST data used for evaluation
            'test': os.path.basename(args.test_images),  # Also points to your test data
            'nc': config['nc'],
            'names': config['names']
        }
        
        # Create empty train directory to satisfy YOLO requirement
        # No actual training happens here - this is just for YOLO's API
        os.makedirs(os.path.join(base_path, 'train'), exist_ok=True)
        
        logger.info(f"Creating data.yaml with config: {json.dumps(data_yaml, indent=2)}")
        logger.info("NOTE: 'train' path is only included because YOLO requires it, but NO TRAINING is happening")
        
        data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # First try with validation
        try:
            logger.info("Running evaluation on test data (using 'val' key because of YOLO requirement)")
            results = model.val(
                data=data_yaml_path,
                split='val',  # Points to your test data
                device=device,
                save_json=True,
                save_txt=True
            )
            eval_method = "validation"
            logger.info("Evaluation completed successfully")
        except Exception as e:
            logger.warning(f"Validation failed, falling back to prediction: {e}")
            # Fall back to prediction
            logger.info("Running direct prediction on test data")
            results = model.predict(
                source=args.test_images,
                device=device,
                save_json=True,
                save_txt=True
            )
            eval_method = "prediction"
            logger.info("Prediction completed successfully")
        
        # Format and save metrics for SageMaker
        metrics = format_metrics_for_sagemaker(results)

        # For prediction mode, extract detection results for manual evaluation
        detection_summary = {}
        if eval_method == "prediction":
            try:
                # Get detection counts from results
                detection_summary = {
                    "images_processed": len(results),
                    "detection_counts": {}
                }
                
                # Extract class detections from results
                for r in results:
                    if hasattr(r, 'boxes') and hasattr(r.boxes, 'cls'):
                        classes = r.boxes.cls.cpu().numpy()
                        for c in classes:
                            c_name = model.names[int(c)]
                            if c_name in detection_summary["detection_counts"]:
                                detection_summary["detection_counts"][c_name] += 1
                            else:
                                detection_summary["detection_counts"][c_name] = 1
            except Exception as e:
                logger.error(f"Error extracting detection summary: {e}")

        # Save detailed metrics
        detailed_metrics = {
            'sagemaker_metrics': metrics,
            'evaluation_method': eval_method,
            'detection_summary': detection_summary
        }
        
        # If we have validation results, add them
        if eval_method == "validation":
            detailed_metrics['raw_metrics'] = {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
            }
        
        # Save metrics to both files
        metrics_path = os.path.join(args.output_dir, 'evaluation.json')
        detailed_metrics_path = os.path.join(args.output_dir, 'detailed_metrics.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        with open(detailed_metrics_path, 'w') as f:
            json.dump(detailed_metrics, f, indent=4)
        
        logger.info(f"Evaluation completed using {eval_method}. Results saved to: {metrics_path}")
        logger.info(f"Detailed metrics saved to: {detailed_metrics_path}")
        logger.info(f"Metrics: {json.dumps(detailed_metrics, indent=2)}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    try:
        # Install requirements
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "ultralytics"])
        except subprocess.CalledProcessError:
            logger.warning("Failed to install via pip, attempting alternative installation...")
            try:
                # Try conda if pip fails
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "ultralytics"])
            except subprocess.CalledProcessError as e:
                logger.error("All installation attempts failed")
                raise e
        
        # Run evaluation
        evaluate_model()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)