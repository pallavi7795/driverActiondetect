import os
import sys
import logging
import argparse
import json
import boto3
import sagemaker
from sagemaker.model_metrics import MetricsSource, ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-package-group-name', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--evaluation-path', type=str, required=True)
    args = parser.parse_args()
    
    try:
        session = sagemaker.Session()
        
        # Create model metrics
        with open(args.evaluation_path) as f:
            metrics = json.load(f)
        
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=args.evaluation_path,
                content_type='application/json'
            )
        )
        
        # Register model
        model_package = session.register_model(
            model_package_group_name=args.model_package_group_name,
            model_data=args.model_path,
            content_types=['application/x-image'],
            response_types=['application/json'],
            inference_instances=['ml.g4dn.xlarge'],
            transform_instances=['ml.g4dn.xlarge'],
            model_metrics=model_metrics,
            approval_status='PendingManualApproval'
        )
        
        logger.info(f"Model registered successfully. Model package ARN: {model_package}")
        
    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    register_mode