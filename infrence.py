import os
import json
import numpy as np
import torch
import cv2

def model_fn(model_dir):
    """
    Load the model for inference
    For Neo-compiled models
    """
    try:
        # Find the model file (Neo compiled models don't always have .pt extension)
        files = os.listdir(model_dir)
        model_files = [f for f in files if os.path.isfile(os.path.join(model_dir, f)) and not f.startswith('.')]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Log available files for debugging
        print(f"Available files in model directory: {files}")
        
        # Use the first model file found
        model_path = os.path.join(model_dir, model_files[0])
        print(f"Loading model from: {model_path}")
        
        # Load the model - for Jetson, using TensorRT optimized model
        # We'll assume it's a YOLOv8 model
        try:
            # First try using YOLO interface if ultralytics is available
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.model.eval()
            return model
        except (ImportError, Exception) as e:
            print(f"Failed to load with ultralytics: {str(e)}")
            # Fall back to TorchScript if YOLO loading fails
            return torch.jit.load(model_path)
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """
    Process incoming request data for YOLOv8
    """
    if request_content_type == "application/x-image":
        # For raw image bytes
        image_as_bytes = np.frombuffer(request_body, dtype=np.uint8)
        # Decode image
        img = cv2.imdecode(image_as_bytes, cv2.IMREAD_COLOR)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions for post-processing
        orig_height, orig_width = img.shape[:2]
        
        # Preprocess for YOLOv8
        # 1. Resize to model input size (640x640)
        img_resized = cv2.resize(img, (640, 640))
        # 2. Convert to float32 and normalize to 0-1
        img_normalized = img_resized.astype(np.float32) / 255.0
        # 3. Convert to NCHW format
        img = img_normalized.transpose(2, 0, 1)  # HWC to CHW
        # 4. Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return {
            'input': img, 
            'orig_size': (orig_height, orig_width)
        }
        
    elif request_content_type == "application/json":
        # For preprocessed tensors in JSON format
        data = json.loads(request_body)
        input_data = np.array(data['input'], dtype=np.float32)
        
        # Ensure we have the original dimensions for scaling later
        orig_size = data.get('orig_size', (640, 640))
        
        return {
            'input': input_data,
            'orig_size': orig_size
        }
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    try:
        img = input_data['input']
        orig_size = input_data['orig_size']
        
        # Detect objects
        if hasattr(model, 'predict') and callable(model.predict):
            # Use YOLO interface if available
            results = model.predict(img, conf=0.25, iou=0.45)
            return {
                'results': results,
                'orig_size': orig_size
            }
        else:
            # For TorchScript models
            with torch.no_grad():
                if isinstance(img, np.ndarray):
                    # Convert numpy array to tensor
                    img_tensor = torch.from_numpy(img).float()
                else:
                    img_tensor = img
                
                # Run inference
                outputs = model(img_tensor)
                
                # For TorchScript models, we'll need to process the outputs
                # based on the specific model structure
                return {
                    'outputs': outputs,
                    'orig_size': orig_size
                }
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def output_fn(prediction_output, accept):
    """
    Serialize and prepare the prediction output
    """
    if accept == 'application/json':
        # Different handling based on prediction output type
        results = []
        
        if 'results' in prediction_output:
            # YOLO native Results object
            yolo_results = prediction_output['results']
            orig_h, orig_w = prediction_output['orig_size']
            
            for pred in yolo_results:
                boxes = pred.boxes
                
                # Get all detections
                detections = []
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].tolist()  # Get boxes in [x1, y1, x2, y2] format
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    
                    # Map class to name if available
                    class_name = pred.names.get(cls, f"class_{cls}") if hasattr(pred, 'names') else f"class_{cls}"
                    
                    detection = {
                        'bbox': box,
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': class_name
                    }
                    detections.append(detection)
                
                results.append(detections)
        
        elif 'outputs' in prediction_output:
            # TorchScript model outputs
            outputs = prediction_output['outputs']
            orig_h, orig_w = prediction_output['orig_size']
            
            # Process outputs based on model architecture
            # This is a simplified example and would need to be adapted
            # to your specific model's output format
            detections = []
            
            # Example processing for YOLO-style outputs
            # Assuming outputs[0] contains detection boxes
            if isinstance(outputs, (list, tuple)):
                # Process multiple outputs (typical for object detection models)
                if len(outputs) >= 3:
                    # Assuming format: [boxes, scores, classes]
                    boxes = outputs[0].cpu().numpy()
                    scores = outputs[1].cpu().numpy()
                    classes = outputs[2].cpu().numpy()
                    
                    for i in range(len(boxes)):
                        # Scale boxes to original image size
                        box = boxes[i]
                        x1, y1, x2, y2 = box
                        
                        # Scale to original image size
                        x1 = float(x1 * orig_w / 640)
                        y1 = float(y1 * orig_h / 640)
                        x2 = float(x2 * orig_w / 640)
                        y2 = float(y2 * orig_h / 640)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(scores[i]),
                            'class_id': int(classes[i]),
                            'class_name': f"class_{int(classes[i])}"
                        }
                        detections.append(detection)
                else:
                    # Single output tensor with all information
                    # Format: [batch, num_detections, 6] where last dim is [x1, y1, x2, y2, confidence, class]
                    detection_tensor = outputs[0].cpu().numpy()
                    
                    for detection in detection_tensor[0]:  # First batch
                        x1, y1, x2, y2, confidence, class_id = detection
                        
                        # Scale to original image size
                        x1 = float(x1 * orig_w / 640)
                        y1 = float(y1 * orig_h / 640)
                        x2 = float(x2 * orig_w / 640)
                        y2 = float(y2 * orig_h / 640)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': int(class_id),
                            'class_name': f"class_{int(class_id)}"
                        }
                        detections.append(detection)
            
            results.append(detections)
        
        return json.dumps(results)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")