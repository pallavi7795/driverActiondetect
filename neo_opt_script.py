import os
import sys
import logging
import argparse
import json
import torch
from pathlib import Path
import sagemaker
from sagemaker.neo import NeoCompilerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--target-instance', type=str, default='ml_g4dn')
    args = parser.parse_args()
    
    try:
        # Configure Neo compilation
        compiler_config = NeoCompilerConfig(
            target_platform_os='LINUX',
            target_platform_arch='X86_64',
            compiler_options={
                'dtype': 'float32',
                'optimization_level': 3
            }
        )
        
        # Compile model
        sagemaker_session = sagemaker.Session()
        compiled_model = sagemaker_session.compile_model(
            model_path=args.model_dir,  #trained model from s3 URI (model.tar.gz)
            target_instance_family=args.target_instance,
            input_shape={'images': [1, 3, 640, 640]},
            framework='PYTORCH',
            framework_version='2.0.1',
            output_path=args.output_dir,  #
            compiler_config=compiler_config
        )
        
        logger.info(f"Model optimized successfully. Output path: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Model optimization failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    optimize_model()