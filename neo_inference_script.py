# optimization/neo_inference_script.py
import os
import json
import torch
import tarfile
import logging
from pathlib import Path

# # Import the ultralytics package for YOLOv8
# try:
#     from ultralytics import YOLO
# except ImportError:
#     # Install ultralytics if not available
#     import subprocess
#     subprocess.check_call(["pip", "install", "ultralytics"])
#     from ultralytics import YOLO

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def model_fn(model_dir):
    """Load the YOLOv8 model for inference"""
    logger.info(f"Loading model from {model_dir}")
    
    # List all files in model directory to debug
    logger.info(f"Model directory contents: {os.listdir(model_dir)}")
    
    # Extract the model archive if it exists
    model_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(model_path):
        # Try to find a .pt file
        pt_files = list(Path(model_dir).glob("**/*.pt"))
        if pt_files:
            model_path = str(pt_files[0])
            logger.info(f"Found PT model at {model_path}")
        else:
            # Look for tar.gz file and extract it
            tar_path = os.path.join(model_dir, "model.tar.gz")
            if os.path.exists(tar_path):
                logger.info(f"Extracting model archive from {tar_path}")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=model_dir)
                
                # Look for .pt file again after extraction
                pt_files = list(Path(model_dir).glob("**/*.pt"))
                if pt_files:
                    model_path = str(pt_files[0])
                    logger.info(f"Found PT model at {model_path} after extraction")
                else:
                    logger.error(f"No .pt files found after extracting {tar_path}")
                    raise FileNotFoundError(f"No YOLOv8 model file found in {model_dir}")
            else:
                logger.error(f"No model file found in {model_dir}")
                raise FileNotFoundError(f"No YOLOv8 model file found in {model_dir}")
    
    # Load the model
    logger.info(f"Loading YOLOv8 model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = YOLO(model_path)
    model.to(device)
    
    return model

def input_fn(request_body, request_content_type):
    """Convert the incoming request data into a format for YOLOv8 inference"""
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/x-image':
        # Process binary image data
        from PIL import Image
        import io
        import numpy as np
        
        # Convert binary data to PIL Image
        image = Image.open(io.BytesIO(request_body))
        logger.info(f"Loaded image with size: {image.size}")
        
        # Convert to numpy array
        img_array = np.array(image)
        return img_array
    
    elif request_content_type == 'application/json':
        # Parse JSON input
        input_data = json.loads(request_body)
        
        # Handle different JSON formats
        if isinstance(input_data, dict) and 'url' in input_data:
            # This is a URL to an image
            import urllib.request
            from PIL import Image
            import io
            import numpy as np
            
            logger.info(f"Downloading image from URL: {input_data['url']}")
            with urllib.request.urlopen(input_data['url']) as response:
                image_data = response.read()
            
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image)
            return img_array
        
        elif isinstance(input_data, dict) and 'image' in input_data:
            # This is base64 encoded image data
            import base64
            from PIL import Image
            import io
            import numpy as np
            
            logger.info("Decoding base64 image data")
            image_data = base64.b64decode(input_data['image'])
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image)
            return img_array
        
        else:
            logger.info("JSON data format not recognized")
            return input_data
    
    else:
        logger.info(f"Unsupported content type: {request_content_type}, treating as JSON")
        return json.loads(request_body)

def predict_fn(input_data, model):
    """Perform prediction with the loaded model"""
    logger.info(f"Running inference with input data of type {type(input_data)}")
    
    # YOLOv8 inference
    results = model(input_data, size=640)
    logger.info(f"Inference completed, results type: {type(results)}")
    
    return results

def output_fn(prediction_output, response_content_type):
    """Convert the prediction output to the expected response format"""
    logger.info(f"Formatting output to content type: {response_content_type}")
    
    # Default to JSON format for response
    if response_content_type is None or 'application/json' in response_content_type:
        results = []
        for result in prediction_output:
            # Extract detection boxes
            boxes = result.boxes.cpu().numpy()
            
            # Format predictions
            pred_results = {
                "boxes": boxes.xyxy.tolist() if hasattr(boxes, 'xyxy') else [],  # x1, y1, x2, y2
                "scores": boxes.conf.tolist() if hasattr(boxes, 'conf') else [],  # confidence scores
                "classes": boxes.cls.tolist() if hasattr(boxes, 'cls') else [],   # class indices
            }
            
            # Add class names if available
            if hasattr(result, 'names') and result.names:
                pred_results["class_names"] = [
                    result.names[int(cls_idx)] for cls_idx in pred_results["classes"]
                ]
            
            results.append(pred_results)
        
        return json.dumps(results)
    
    # If output format is not supported, return string representation
    return str(prediction_output)