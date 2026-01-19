
import time
import json
import boto3
import argparse
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-data', type=str, required=True)
    parser.add_argument('--compilation-output-path', type=str, required=True)
    parser.add_argument('--target-platform-os', type=str, default='LINUX')
    parser.add_argument('--target-platform-arch', type=str, default='X86_64')
    parser.add_argument('--target-device', type=str, default=None)
    parser.add_argument('--framework', type=str, default='PYTORCH')
    
    args = parser.parse_args()
    
    # Set up the SageMaker client
    sm_client = boto3.client('sagemaker')
    
    # Generate a unique job name
    compilation_job_name = f'yolov8-neo-{int(time.time())}'
    
    # Configure input for YOLOv8
    data_input_config = {
        'data': [1, 3, 640, 640]  # Standard YOLOv8 input dimensions
    }
    
    # Set up input config
    input_config = {
        'S3Uri': args.model_data,
        'DataInputConfig': json.dumps(data_input_config),
        'Framework': args.framework
    }
    
    # Set up output config
    output_config = {
        'S3OutputLocation': args.compilation_output_path
    }
    
    # Target platform can be either a device or platform
    if args.target_device and args.target_device.lower() != "none":
        output_config['TargetDevice'] = args.target_device
    else:
        output_config['TargetPlatform'] = {
            'Os': args.target_platform_os,
            'Arch': args.target_platform_arch
        }
    
    # Configure the compilation job
    compilation_job_config = {
        'CompilationJobName': compilation_job_name,
        'RoleArn': os.environ['SAGEMAKER_ROLE'],
        'InputConfig': input_config,
        'OutputConfig': output_config,
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 900
        }
    }
    
    # Start the compilation job
    logger.info(f"Starting compilation job: {compilation_job_name}")
    logger.info(f"Compilation config: {json.dumps(compilation_job_config, default=str)}")
    
    try:
        sm_client.create_compilation_job(**compilation_job_config)
        
        # Monitor the job
        status = None
        while status not in ['COMPLETED', 'FAILED', 'STOPPED']:
            response = sm_client.describe_compilation_job(
                CompilationJobName=compilation_job_name
            )
            status = response['CompilationJobStatus']
            
            if status == 'FAILED':
                failure_reason = response.get('FailureReason', 'Unknown reason')
                logger.error(f"Compilation job failed: {failure_reason}")
                raise Exception(f"Compilation job failed: {failure_reason}")
            
            logger.info(f"Compilation job status: {status}")
            
            if status not in ['COMPLETED', 'FAILED', 'STOPPED']:
                logger.info("Waiting for compilation job to complete...")
                time.sleep(30)
        
        # If job completed successfully, get the output artifact
        if status == 'COMPLETED':
            model_artifact = response['ModelArtifacts']['S3ModelArtifacts']
            logger.info(f"Compilation completed successfully. Model artifact: {model_artifact}")
            
            # Write the artifact location to output file for downstream consumption
            with open('/opt/ml/processing/output/compilation_output.json', 'w') as f:
                json.dump({
                    'compilation_job_name': compilation_job_name,
                    'compiled_model_artifact': model_artifact
                }, f)
            
            logger.info("Compilation job results saved to output file")
        else:
            logger.error(f"Compilation job did not complete successfully. Status: {status}")
            raise Exception(f"Compilation job did not complete successfully. Status: {status}")
    
    except Exception as e:
        logger.error(f"Error in compilation job: {str(e)}")
        # Create output file even on error to prevent pipeline failure
        with open('/opt/ml/processing/output/compilation_output.json', 'w') as f:
            json.dump({
                'error': str(e),
                'compilation_job_name': compilation_job_name
            }, f)
        raise

if __name__ == '__main__':
    main()