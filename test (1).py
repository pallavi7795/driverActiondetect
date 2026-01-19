import os
import sys
import logging
import argparse
import json
import torch
from pathlib import Path
import sagemaker
from sagemaker.compiler.config import CompilerConfig
from sagemaker.exceptions import SageMakerException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_paths(model_dir: str, output_dir: str) -> tuple[Path, Path]:
    """
    Validate input and output paths.
    
    Args:
        model_dir: Input model directory path
        output_dir: Output directory path
        
    Returns:
        Tuple of validated Path objects
    """
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return model_path, output_path

def get_compiler_config(target_instance: str) -> CompilerConfig:
    """
    Create compiler configuration.
    
    Args:
        target_instance: Target instance type
        
    Returns:
        CompilerConfig object
    """
    return CompilerConfig(
        target_platform_os='LINUX',
        target_platform_arch='X86_64',
        compiler_options={
            'dtype': 'float32',
            'optimization_level': 3
        }
    )

def optimize_model():
    """Main function to optimize PyTorch model using SageMaker compilation."""
    parser = argparse.ArgumentParser(description='Optimize PyTorch model using SageMaker compilation')
    parser.add_argument('--model-dir', type=str, required=True,
                      help='Input directory containing the model')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for optimized model')
    parser.add_argument('--target-instance', type=str, default='ml_g4dn',
                      help='Target instance type (default: ml_g4dn)')
    parser.add_argument('--framework-version', type=str, default='2.0.1',
                      help='PyTorch framework version (default: 2.0.1)')
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        model_path, output_path = validate_paths(args.model_dir, args.output_dir)
        logger.info(f"Model directory: {model_path}")
        logger.info(f"Output directory: {output_path}")
        
        # Get compiler configuration
        compiler_config = get_compiler_config(args.target_instance)
        
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        
        # Compile model
        logger.info("Starting model compilation...")
        compiled_model = sagemaker_session.compile_model(
            model_path=str(model_path),
            target_instance_family=args.target_instance,
            input_shape={'images': [1, 3, 640, 640]},  # Consider making this configurable
            framework='PYTORCH',
            framework_version=args.framework_version,
            output_path=str(output_path),
            compiler_config=compiler_config
        )
        
        logger.info(f"Model optimized successfully")
        logger.info(f"Output path: {output_path}")
        
        # Save compilation metadata
        metadata = {
            'framework_version': args.framework_version,
            'target_instance': args.target_instance,
            'input_shape': {'images': [1, 3, 640, 640]},
            'compiler_options': compiler_config.compiler_options
        }
        
        metadata_path = output_path / 'compilation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved compilation metadata to: {metadata_path}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except SageMakerException as e:
        logger.error(f"SageMaker compilation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during model optimization: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    optimize_model()