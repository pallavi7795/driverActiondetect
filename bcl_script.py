print("Eval step")

import os
import sys
import json
import logging
import argparse
import subprocess
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import tarfile
#import os
import pip


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    try:
        import subprocess

        # Check if we have sudo rights
        subprocess.run(['sudo', '-n', 'true'], check=True)

        # Update package list
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        
        # # Install system dependencies for OpenCV
        # subprocess.check_call(["apt-get", "update"])
        # subprocess.check_call(["apt-get", "install", "-y", "libgl1-mesa-glx"])
        
        # # Install Python packages
        # subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "ultralytics"])
        # logger.info("Successfully installed requirements")
        
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
        # '/opt/ml/processing/input/config/config.yaml'  # config/config.yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Successfully loaded model config")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def evaluate_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-images', type=str, default='/opt/ml/processing/input/images/test')
    parser.add_argument('--test-labels', type=str, default='/opt/ml/processing/input/labels/test')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/evaluation')
    #'/opt/ml/processing/evaluation' #training/outputs
    args = parser.parse_args()
    
    try:
        from ultralytics import YOLO
        # Prepare output directory
        #os.makedirs(args.output_dir, exist_ok=True)
        
        # Load config file
        config = load_config()
        logger.info(f"Loaded config with {config['nc']} classes")

        # Prepare model path
        model_path = prepare_model_path(args.model_path)
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        logger.info("Model loaded successfully")

        # # If directory, find first .tar.gz or .pt file
        # if os.path.isdir(model_path):
        #     files = [f for f in os.listdir(model_path) if f.endswith(('.tar.gz', '.pt'))]
        #     if not files:
        #         raise FileNotFoundError("No .tar.gz or .pt files found in directory")
        #     model_path = os.path.join(model_path, files[0])
    
        # # If tar.gz, extract
        # if model_path.endswith('.tar.gz'):
        #     extract_dir = os.path.join(os.path.dirname(model_path), 'extracted')
        #     os.makedirs(extract_dir, exist_ok=True)
            
        #     with tarfile.open(model_path, 'r:gz') as tar:
        #         tar.extractall(extract_dir)
            
        #     # Find .pt file
        #     pt_files = list(Path(extract_dir).rglob('*.pt'))
        #     if not pt_files:
        #         raise FileNotFoundError("No .pt file found in archive")
        #     model_path = str(pt_files[0])

        # # Prepare model path
        # model_path = prepare_model_path(args.model_path)
        # logger.info(f"Contents of model directory: {os.listdir(os.path.dirname(args.model_path))}")

        # #--------------
        # # Ensure model path is a file, not a directory
        # if os.path.isdir(args.model_path):
        #     # Find .tar.gz file in the directory
        #     model_files = [f for f in os.listdir(args.model_path) if f.endswith('.tar.gz')]
        #     if not model_files:
        #         raise FileNotFoundError("No .tar.gz file found in the directory")
        #     model_path = os.path.join(args.model_path, model_files[0])
        # else:
        #     model_path = args.model_path
        # #--------------
        
        # # Extract model if it's a tar.gz file
        # if args.model_path.endswith('.tar.gz'):
            
        #     extract_dir = os.path.join(os.path.dirname(args.model_path), 'extracted')
        #     os.makedirs(extract_dir, exist_ok=True)
            
        #     logger.info(f"Extracting model from {args.model_path} to {extract_dir}")
            
        #     with tarfile.open(args.model_path, 'r:gz') as tar:
        #         tar.extractall(extract_dir)
            
        #     # Find the .pt file
        #     model_files = list(Path(extract_dir).rglob('*.pt'))
        #     if not model_files:
        #         raise FileNotFoundError("No .pt file found in model archive")
        #     model_path = str(model_files[0])
        # else:
        #     model_path = args.model_path
        #     print("eval model_path:",model_path)
            
        # # Load model
        # logger.info(f"Loading model from {model_path}")
        # model = YOLO(model_path)
        # logger.info("Model loaded successfully")

        # Prepare data paths
        base_path = os.path.dirname(args.test_images)
        
        # Create data.yaml for test data using config values           
        data_yaml = {
            'path': base_path,
            'train': os.path.join(os.path.dirname(args.test_images), 'train'),
            'val': os.path.join(os.path.dirname(args.test_images), 'test'),
            'test': os.path.basename(args.test_images),
            'nc': config['nc'],
            'names': config['names']
        }
        
        # Ensure paths exist
        os.makedirs(data_yaml['train'], exist_ok=True)
        os.makedirs(data_yaml['val'], exist_ok=True)
        
        logger.info(f"Creating data.yaml with config: {json.dumps(data_yaml, indent=2)}")
        
        data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
         # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Run validation on test data
        try:
            results = model.val(
                data=data_yaml_path,
                split='test',
                device=device,
                save_json=True,
                save_txt=True
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            results = model.predict(
                source=args.test_images,
                device=device,
                save_json=True,
                save_txt=True
            )
            
        
        # Extract metrics

        metrics = {
           'metrics': {
               'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
               'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
               'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
               'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
           }
       }
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'evaluation.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation completed. Results saved to: {metrics_path}")
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    try:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "ultralytics"])
        except subprocess.CalledProcessError:
            logger.warning("Failed to install via pip, attempting alternative installation...")
        try:
            # Try conda if pip fails
            #subprocess.check_call(["conda", "install", "-y", "opencv"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "ultralytics"])
        except subprocess.CalledProcessError as e:
            logger.error("All installation attempts failed")
            raise e
        
        #install_requirements()
        
        
        # Then run evaluation
        evaluate_model()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)



# First install requirements
# subprocess.run(['apt-get', 'update', '-y'])
# subprocess.run(['apt-get', 'install', '-y', 'libgl1-mesa-glx', 'libglib2.0-0'])
# subprocess.run(['pip3', 'install', 'ultralytics'])

# First install requirements
# subprocess.check_call(["apt-get", "update", "-y"])
# subprocess.check_call(["apt-get", "install", "-y", "libgl1-mesa-glx", "libglib2.0-0"])