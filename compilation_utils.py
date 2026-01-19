# utils/compilation_utils.py
import time
import boto3
import logging
import sagemaker
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_latest_approved_model_location(sm_client, model_package_group_name):
    """Get the latest approved model package S3 location"""
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )
        
        if not response['ModelPackageSummaryList']:
            logger.warning(f"No approved model packages found in group {model_package_group_name}")
            return None
        
        model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
        logger.info(f"Found latest approved model package: {model_package_arn}")
        
        model_details = sm_client.describe_model_package(ModelPackageName=model_package_arn)
        model_data_url = model_details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        
        logger.info(f"Model data URL: {model_data_url}")
        return model_data_url
        
    except Exception as e:
        logger.error(f"Error getting latest approved model: {str(e)}")
        raise

def run_neo_compilation_job(role_arn, model_s3_uri, compilation_output_path, 
                           target_platform=None, target_device=None, framework="PYTORCH"):
    """
    Run a SageMaker Neo compilation job for YOLOv8 model
    
    Args:
        role_arn (str): IAM role ARN for SageMaker
        model_s3_uri (str): S3 URI to the model artifacts
        compilation_output_path (str): S3 URI for the compilation output
        target_platform (dict, optional): Dict with 'Os' and 'Arch' keys, e.g., {'Os': 'LINUX', 'Arch': 'X86_64'}
        target_device (str, optional): Target device (e.g., 'ml_c5', 'jetson_xavier')
        framework (str): ML framework, use 'PYTORCH' for YOLOv8
        
    Returns:
        str: S3 URI of the compiled model artifacts
    """
    sm_client = boto3.client('sagemaker')
    
    # Create a unique compilation job name
    compilation_job_name = f'yolov8-neo-{int(time.time()*1000)}'
    
    # YOLOv8 input configuration
    input_config = {
        'S3Uri': model_s3_uri,
        'DataInputConfig': json.dumps({"data": [1, 3, 640, 640]}),
        'Framework': framework
    }
    
    # Output configuration
    output_config = {
        'S3OutputLocation': compilation_output_path
    }
    
    # Set either target platform or target device
    if target_device:
        output_config['TargetDevice'] = target_device
    elif target_platform:
        output_config['TargetPlatform'] = target_platform
    else:
        # Default to Linux x86_64
        output_config['TargetPlatform'] = {
            'Os': 'LINUX',
            'Arch': 'X86_64'
        }
    
    # Create the compilation job configuration
    compilation_job_config = {
        'CompilationJobName': compilation_job_name,
        'RoleArn': role_arn,
        'InputConfig': input_config,
        'OutputConfig': output_config,
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 900
        }
    }
    
    # Start the compilation job
    logger.info(f"Creating compilation job: {compilation_job_name}")
    logger.info(f"Job config: {json.dumps(compilation_job_config, default=str, indent=2)}")
    
    try:
        sm_client.create_compilation_job(**compilation_job_config)
        
        # Poll the status of the job
        logger.info(f'Started compilation job {compilation_job_name}')
        print('Started compilation job .', end='')
        
        while True:
            resp = sm_client.describe_compilation_job(CompilationJobName=compilation_job_name)
            status = resp['CompilationJobStatus']
            
            if status in ['STARTING', 'INPROGRESS']:
                print('.', end='')
            else:
                print(f"\n{status}: {compilation_job_name}")
                break
            time.sleep(10)
        
        if status == 'COMPLETED':
            s3_compiled_model_artifact = resp['ModelArtifacts']['S3ModelArtifacts']
            logger.info(f'Compiled artifact location in S3: {s3_compiled_model_artifact}')
            return s3_compiled_model_artifact
        else:
            error_msg = f"Compilation job failed with status: {status}"
            if 'FailureReason' in resp:
                error_msg += f"\nReason: {resp['FailureReason']}"
            
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    except Exception as e:
        logger.error(f"Error in compilation job: {str(e)}")
        raise

def deploy_neo_compiled_model(role, compiled_model_uri, instance_type, endpoint_name=None,
                              framework_version="1.13.1"):
    """
    Deploy a Neo-compiled YOLOv8 model to a SageMaker endpoint
    
    Args:
        role (str): IAM role ARN
        compiled_model_uri (str): S3 URI to the compiled model
        instance_type (str): Instance type for deployment
        endpoint_name (str, optional): Custom endpoint name
        framework_version (str): PyTorch framework version
        
    Returns:
        sagemaker.predictor.Predictor: Predictor for the endpoint
    """
    try:
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        
        # Create a unique endpoint name if not provided
        if not endpoint_name:
            endpoint_name = f'yolov8-endpoint-{int(time.time())}'
        
        # Get the appropriate container for Neo-compiled model
        region = boto3.session.Session().region_name
        
        # Determine if deploying to GPU or CPU instance
        if any(gpu_type in instance_type for gpu_type in ['ml.p', 'ml.g']):
            image_type = "pytorch-inference-neo-gpu"
        else:
            image_type = "pytorch-inference-neo"
        
        # Get the Neo container URI
        from sagemaker.image_uris import retrieve
        try:
            container_uri = retrieve(
                framework=image_type,
                region=region,
                version=framework_version
            )
        except Exception as e:
            logger.warning(f"Could not retrieve Neo container URI: {str(e)}. Using PyTorch container instead.")
            
            # Fallback to regular PyTorch container
            if any(gpu_type in instance_type for gpu_type in ['ml.p', 'ml.g']):
                container_uri = retrieve(
                    framework="pytorch-inference",
                    region=region,
                    version=framework_version,
                    image_scope="inference",
                    instance_type=instance_type
                )
            else:
                container_uri = retrieve(
                    framework="pytorch-inference",
                    region=region,
                    version=framework_version,
                    image_scope="inference"
                )
        
        logger.info(f"Using container URI: {container_uri}")
        
        # Create a SageMaker model using the Neo-compiled artifacts
        from sagemaker.model import Model
        
        model = Model(
            image_uri=container_uri,
            model_data=compiled_model_