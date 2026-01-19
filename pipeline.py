import os
import sys
import logging
import boto3
import yaml
import sagemaker
from datetime import datetime
from pathlib import Path
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CacheConfig,
    #CompilationStep  # Correct import for CompilationStep
)
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.functions import Join
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.retry import RetryPolicy
from sagemaker.workflow.step_collections import RegisterModel

from sagemaker.processing import FrameworkProcessor
from sagemaker.pytorch import PyTorch

from sagemaker.workflow.functions import JsonGet

#-----------------------------------------------------------------------------
import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
#-----------

from sagemaker.pytorch import PyTorchModel
#from sagemaker.workflow.compilation_step import CompilationStep, CompilationInput

#------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv8Pipeline:
    def __init__(self, aws_config_path, model_config_path=None):
        """Initialize pipeline with AWS and model configurations"""
        # Load AWS configuration
        self.aws_config = self._load_config(aws_config_path, 'aws')
        
        # Load model configuration if provided
        self.model_config = None
        if model_config_path:
            self.model_config = self._load_config(model_config_path, 'model')
        
        self.clean_s3_paths()
        
        # Set AWS configurations
        self.role = self.aws_config['aws']['role_arn']
        self.region = self.aws_config['aws']['region']
        self.bucket = self.aws_config['aws']['bucket']
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        # Set training configurations
        self.instance_type = self.aws_config['training']['instance_type']
        self.instance_count = self.aws_config['training']['instance_count']
        self.framework_version = self.aws_config['training']['framework_version']
        self.py_version = self.aws_config['training']['py_version']
        
        # Initialize sessions
        self.boto_session = boto3.Session(region_name=self.region)
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.boto_session,
            default_bucket=self.bucket
        )
        self.pipeline_session = PipelineSession(
            boto_session=self.boto_session,
            sagemaker_client=self.sagemaker_session.sagemaker_client
        )

        # Set retry policy
        self.step_retry = RetryPolicy(
            backoff_rate=1.5,
            interval_seconds=60,
            max_attempts=3
        )

        # Cache configuration
        self.cache_config = CacheConfig(
            enable_caching=True,
            expire_after="P1D"
        )

    def _load_s3_config(self, s3_uri, config_type):
        """Load configuration from S3"""
        try:
            # Parse S3 URI
            s3_path = s3_uri.replace('s3://', '')
            bucket = s3_path.split('/')[0]
            key = '/'.join(s3_path.split('/')[1:])
            
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Download file from S3
            response = s3_client.get_object(Bucket=bucket, Key=key)
            config_content = response['Body'].read().decode('utf-8')
            
            # Parse YAML content
            config = yaml.safe_load(config_content)
            logger.info(f"Loaded {config_type} config from: {s3_uri} ") #config: {config} 
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading {config_type} config from S3: {str(e)}")
            raise

    def _load_config(self, config_path, config_type):
        """Load configuration from file or S3"""
        try:
            if config_path.startswith('s3://'):
                config = self._load_s3_config(config_path, config_type)
            else:
                # Local file loading
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded {config_type} config from local path: {config_path}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading {config_type} config: {str(e)}")
            raise

    def clean_s3_paths(self):
        """Clean S3 paths in configuration"""
        for prefix_key in ['input_prefix', 'output_prefix', 'model_prefix']:
            if prefix_key in self.aws_config['s3']:
                self.aws_config['s3'][prefix_key] = '/'.join(
                    filter(None, self.aws_config['s3'][prefix_key].split('/'))
                )
        logger.info("Cleaned S3 paths in configuration")

    def clean_s3_uri(self, bucket, prefix):
        """Clean S3 URI"""
        clean_prefix = '/'.join(filter(None, prefix.strip('/').split('/')))
        return f"s3://{bucket}/{clean_prefix}"

    def get_pipeline_parameters(self):
        """Define pipeline parameters"""
        return {
            'input_prefix': ParameterString(
                name="InputPrefix",
                default_value=self.aws_config['s3']['input_prefix']
            ),
            'output_prefix': ParameterString(
                name="OutputPrefix",
                default_value=self.aws_config['s3']['output_prefix']
            ),
            'model_prefix': ParameterString(
                name="ModelPrefix",
                default_value=self.aws_config['s3']['model_prefix']
            ),
            'model_eval': ParameterString(
                name="ModelEval",
                default_value=self.aws_config['s3']['model_eval']
            ),
            'image_size': ParameterInteger(
                name="ImageSize",
                default_value=640
            ),
            'num_classes': ParameterInteger(
                name="NumClasses",
                default_value=12
            ),
            'epochs': ParameterInteger(
                name="Epochs",
                default_value=100
            ),
            'batch_size': ParameterInteger(
                name="BatchSize",
                default_value=16
            ),
            'learning_rate': ParameterFloat(
                name="LearningRate",
                default_value=0.001
            ),
            'patience': ParameterInteger(
                name="Patience",
                default_value=50
            ),
            'map_threshold': ParameterFloat(
                name="MAPThreshold",
                default_value=0.5
            )
        }

    def create_preprocessing_step(self, parameters):
        """Create preprocessing step"""
        processor = ScriptProcessor(
            image_uri=f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-training:{self.framework_version}-gpu-{self.py_version}",
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            base_job_name="yolov8-preprocess",
            command=["python3"],
            sagemaker_session=self.sagemaker_session
        )

        input_data_uri = self.clean_s3_uri(self.bucket, f"{parameters['input_prefix'].default_value}")
        output_data_uri = self.clean_s3_uri(self.bucket, parameters['output_prefix'].default_value)

        print(" input_data_uri -->",input_data_uri)
        print(" output_data_uri -->",output_data_uri)
        
        return ProcessingStep(
            name="PreprocessData",
            processor=processor,
            inputs=[
                ProcessingInput(
                source=f"{input_data_uri}/images",
                destination="/opt/ml/processing/input/images",  # Changed from /opt/ml/input/data/images
                s3_data_type="S3Prefix"
            ),
            ProcessingInput(
                source=f"{input_data_uri}/labels",
                destination="/opt/ml/processing/input/labels",  # Changed from /opt/ml/input/data/labels
                s3_data_type="S3Prefix"
            ),
            ProcessingInput(
                source=f"s3://{self.bucket}/config",
                destination="/opt/ml/processing/input/config",  # Changed from /opt/ml/input/config
                s3_data_type="S3Prefix"
            )
            ],
            outputs=[
            ProcessingOutput(
                output_name="training_data",  # This name must match what we reference
                source="/opt/ml/processing/output",
                destination=output_data_uri  #s3 path
            )
            ],
            code="preprocessing/preprocessing.py",
            #cache_config=self.cache_config
        )

    def create_training_step(self, parameters, preprocessing_step):
        """Create training step"""

        pretrained_model_uri = "s3://pw-fleet/pw-pretrained-model/"
        
        # metrics_definitions = [
        #     {'Name': 'map', 'Regex': 'metrics/mAP50-95\\(B\\):(.*?);'},
        #     {'Name': 'map50', 'Regex': 'metrics/mAP50\\(B\\):(.*?);'},
        #     {'Name': 'precision', 'Regex': 'metrics/precision\\(B\\):(.*?);'},
        #     {'Name': 'recall', 'Regex': 'metrics/recall\\(B\\):(.*?);'},
        #     {'Name': 'train_accuracy', 'Regex': 'Train_Accuracy:\\s*([-+]?\\d*\\.?\\d+)'},
        #     {'Name': 'val_accuracy', 'Regex': 'Val_Accuracy:\\s*([-+]?\\d*\\.?\\d+)'}
        #     ]
        # In your pipeline's create_training_step method:
        
        metrics_definitions = [
            { 'Name': 'precision', 'Regex': 'metrics/precision\\(B\\):\\s*([-+]?\\d*\\.?\\d+);'},
            { 'Name': 'recall', 'Regex': 'metrics/recall\\(B\\):\\s*([-+]?\\d*\\.?\\d+);'},
            { 'Name': 'mAP50', 'Regex': 'metrics/mAP50\\(B\\):\\s*([-+]?\\d*\\.?\\d+);'},
            { 'Name': 'mAP50-95', 'Regex': 'metrics/mAP50-95\\(B\\):\\s*([-+]?\\d*\\.?\\d+);'}
            
        ]   
            # Add new accuracy metrics
            # {'Name': 'accuracy', 'Regex': 'metrics/accuracy\\(B\\):(.*?);'},
            # {'Name': 'val_accuracy', 'Regex': 'metrics/val_accuracy\\(B\\):(.*?);'},
            # {'Name': 'train_accuracy', 'Regex': 'metrics/train_accuracy\\(B\\):(.*?);'}

        
        # metrics_definitions = [
        #     {"Name": "train:loss", "Regex": "loss: ([0-9\\.]+)"},
        #     {"Name": "train:accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
        #     {"Name": "validation:loss", "Regex": "val_loss: ([0-9\\.]+)"},
        #     {"Name": "validation:accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"},
        # ]

        training_data_uri = preprocessing_step.properties.ProcessingOutputConfig.Outputs["training_data"].S3Output.S3Uri
       
        #where model
        model_output_path = self.clean_s3_uri(self.bucket,f"{parameters['model_prefix'].default_value}/artifacts")
        
        #print("model_output_path:",model_output_path)
        
        config_uri = f"s3://{self.bucket}/config/"

        estimator = PyTorch(
            entry_point="training_script.py",
            source_dir="training",
            role=self.role,
            framework_version=self.framework_version,
            py_version=self.py_version,
            instance_count=self.instance_count, #1
            instance_type=self.instance_type,
            volume_size=100,
            max_run=432000,
            metric_definitions=metrics_definitions,
            enable_sagemaker_metrics= True,
            hyperparameters={
                "image-size": parameters['image_size'],
                "epochs": parameters['epochs'],
                "batch-size": parameters['batch_size'],
                "workers": 8,
                "patience": parameters['patience'],
                "save-period": 10,
                "optimizer": "Adam",
                "learning-rate": parameters['learning_rate'],
                "weight-decay": 0.0005,
                "num-classes": parameters['num_classes'],
                "label-smoothing": 0.0,
                "dropout": 0.0
            },
            environment={
                "SM_CHANNEL_TRAINING": "/opt/ml/input/data/training",
                "SM_CHANNEL_PRETRAINED": "/opt/ml/input/data/pretrained",
                "SM_CHANNEL_CONFIG": "/opt/ml/input/data/config",
                "SM_OUTPUT_DIR": "/opt/ml/model",
                "SM_MODEL_DIR": "/opt/ml/model"
            },
            output_path=model_output_path,
            sagemaker_session=self.sagemaker_session
        )

        return TrainingStep(
            name="TrainYOLOv8Model",
            estimator=estimator,
            inputs={
                "training": TrainingInput(   #Changed from train to training
                    s3_data=training_data_uri,
                    distribution="FullyReplicated",
                    s3_data_type="S3Prefix",
                    input_mode="File",
                    content_type="application/x-recordio"
                ),
                "pretrained": TrainingInput(
                    s3_data=pretrained_model_uri,
                    distribution="FullyReplicated",
                    s3_data_type="S3Prefix", #folder
                    input_mode="File"
                ),
                "config": TrainingInput(
                    s3_data=config_uri,
                    distribution="FullyReplicated",
                    s3_data_type="S3Prefix",
                    input_mode="File"
                )
            },
            
            cache_config=self.cache_config
        )

    def create_evaluation_step(self, training_step, parameters):
        """Create evaluation step"""
        processor = ScriptProcessor(
            image_uri=f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-training:{self.framework_version}-gpu-{self.py_version}",
            role=self.role,
            instance_count=1,
            instance_type=self.instance_type,
            base_job_name="yolov8-evaluate",
            command=["python3"], #"evaluate_script.py"
            sagemaker_session=self.sagemaker_session
        )

        # Define the metrics property file
        metrics_property = PropertyFile(
            name="EvaluationMetrics",
            output_name="evaluation", 
            path="evaluation.json"
        )
        
        evaluation_report = PropertyFile(   # not
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json"
        )

        input_data_uri = self.clean_s3_uri(self.bucket, parameters['input_prefix'].default_value)
        model_eval_path = parameters['model_eval'].default_value

        # Construct full S3 destination path
        evaluation_output_uri = f"s3://{self.bucket}/{model_eval_path}/evaluation"

        return ProcessingStep(
            name="EvaluateModel",
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"   #/model.tar.gz" # Changed from /opt/ml/model/artifacts
                ),
                ProcessingInput(
                    source=f"{input_data_uri}/images/test", 
                    destination="/opt/ml/processing/input/images/test"  # Changed to match script
                ),
                ProcessingInput(
                    source=f"{input_data_uri}/labels/test",
                    destination="/opt/ml/processing/input/labels/test"
                ),
                ProcessingInput(
                source=f"s3://{self.bucket}/config",
                destination="/opt/ml/processing/input/config"
                ),
                ProcessingInput(
                source="evaluation/requirements.txt",
                destination="/opt/ml/processing/input/requirements"
                )
            ],
            
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",  # Fixed path
                    destination= evaluation_output_uri
                )
            ],
            #code="evaluation/evaluate_script.py",
            #code="evaluate_script.py",
            code="evaluation/evaluate_script.py",
            property_files=[metrics_property]
            #source_dir="evaluation",
            #property_files=[evaluation_report],
            #cache_config=self.cache_config
        )

    def create_register_model_step(self, training_step, evaluation_step):
        """Create model registration step"""
        return RegisterModel(
            name="RegisterModel",
            estimator=training_step.estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/x-image"],
            response_types=["application/json"],
            inference_instances=["ml.g4dn.xlarge"],
            transform_instances=["ml.g4dn.xlarge"],
            model_package_group_name=f"YOLOv8Model-{self.timestamp}",
            approval_status="PendingManualApproval",
            model_metrics=ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=Join(
                        on="/",
                        values=[
                            evaluation_step.properties.ProcessingOutputConfig.Outputs[
                                "evaluation"
                            ].S3Output.S3Uri,
                            "evaluation.json"
                        ]
                    ),
                    content_type="application/json"
                )
            )
        )

    # def create_compilation_step_02(self, training_step, parameters):
    #     """Create a SageMaker Neo compilation step for the trained YOLOv8 model using CompilationStep"""

        
    #     # Set up the SageMaker session
    #     role = self.role
    #     pipeline_session = self.pipeline_session
    #     bucket = self.bucket
        
    #     # Get model from training step
    #     model_data = training_step.properties.ModelArtifacts.S3ModelArtifacts
        
    #     # Define compilation output path
    #     compilation_output_path = self.clean_s3_uri(
    #         self.bucket,
    #         f"{parameters['model_prefix'].default_value}/compiled"
    #     )
        
    #     # Create a PyTorch model
    #     pytorch_model = PyTorchModel(
    #         model_data=model_data,  # Use model data from training step
    #         role=role,
    #         entry_point="inference.py",  # Make sure this file exists in your code directory
    #         source_dir="optimization",   # Directory containing inference.py and any other needed files
    #         framework_version=self.framework_version,
    #         py_version=self.py_version,
    #         sagemaker_session=pipeline_session
    #     )
        
    #     # Create compilation inputs for Jetson
    #     compilation_inputs = CompilationInput(
    #         target_instance_type="jetson_tx2",  # Use jetson_tx2 for Jetson TX2
    #         input_shape=json.dumps({"data": [1, 3, 640, 640]}),  # YOLOv8 input shape
    #         output_path=compilation_output_path,
    #         framework="pytorch",
    #         framework_version=self.framework_version,
    #         compile_max_run=3600,  # 1 hour
    #         job_name=f"yolov8-jetson-compilation-{self.timestamp}",
    #         compiler_options=json.dumps({
    #             "dtype": "float32",                  # Use float32 for higher accuracy
    #             "optimization_level": 3,             # Highest optimization level
    #             "target_platform_os": "LINUX",       # Target OS
    #             "target_platform_arch": "ARM64",     # Jetson architecture
    #             "target_platform_accelerator": "NVIDIA",  # For Jetson GPU
    #             "mcpu": "cortex-a57+crypto"          # CPU architecture for Jetson TX2
    #         })
    #     )
        
    #     # Create compilation step
    #     return CompilationStep(
    #         name="CompileYOLOv8ForJetson",
    #         estimator=None,                   # Not needed when model is provided
    #         model=pytorch_model,
    #         inputs=compilation_inputs,
    #         depends_on=[training_step.name],
    #         display_name="YOLO Model Compilation for Jetson",
    #         description="Compile YOLOv8 model for optimized inference on NVIDIA Jetson"
    #     )

    def create_compilation_step(self, training_step, parameters):
        """Create a SageMaker Neo compilation step for the trained YOLOv8 model for Jetson"""
        # Get model from training step
        model_data = training_step.properties.ModelArtifacts.S3ModelArtifacts
        
        # Define compilation output path
        compilation_output_path = self.clean_s3_uri(
            self.bucket,
            f"{parameters['model_prefix'].default_value}/compiled"
        )
        
        # Create a PyTorch model
        pytorch_model = PyTorchModel(
            model_data=model_data,
            role=self.role,
            entry_point="compilation_script.py",
            source_dir="optimization",
            framework_version=self.framework_version,
            py_version=self.py_version,
            sagemaker_session=self.pipeline_session
        )
        
         #Create compilation step with the correct parameters for SageMaker SDK 2.237.1
        # return CompilationStep(
        #     name="CompileYOLOv8ForJetson",
        #     model=pytorch_model,
        #     target_device="jetson_tx2",   # For Jetson TX2
        #     target_platform_os="LINUX",
        #     target_platform_arch="ARM64",
        #     target_platform_accelerator="NVIDIA",
        #     compiler_options={"dtype": "float32", "optimization_level": 3},
        #     input_shape={"data": [1, 3, 640, 640]},  # YOLOv8 input shape
        #     job_name=f"yolov8-jetson-compilation-{self.timestamp}",
        #     output_path=compilation_output_path,
        #     depends_on=[training_step.name],
        #     description="Compile YOLOv8 model for optimized inference on NVIDIA Jetson"
        #)


    def create_pipeline(self):
        """Create complete pipeline"""
        parameters = self.get_pipeline_parameters()
        # Create pipeline steps
        preprocessing_step = self.create_preprocessing_step(parameters) 
        training_step = self.create_training_step(parameters, preprocessing_step)
        training_step.add_depends_on([preprocessing_step])
        evaluation_step = self.create_evaluation_step(training_step, parameters)
        compilation_step = self.create_compilation_step(training_step, parameters) #adding compilation step

        # Define metrics property file to extract the metrics
        metrics_property = PropertyFile(
            name="EvaluationMetrics",
            output_name="evaluation",
            path="evaluation.json"
        )

        # Create condition for model registration
        condition = ConditionStep(
            name="CheckMAPThreshold",
            conditions=[ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                step_name=evaluation_step.name,
                property_file=metrics_property,
                json_path="object_detection_metrics.mAP50-95.value"
            ),
            right= parameters['map_threshold']
                
                # left=Join(
                #     on="/",
                #     values=[
                #         evaluation_step.properties.ProcessingOutputConfig.Outputs[
                #             "evaluation"
                #         ].S3Output.S3Uri,
                #         "map50"
                #     ]
                # ),
                # right=parameters['map_threshold']
            )],
            if_steps=[self.create_register_model_step(training_step, evaluation_step)],
            else_steps=[]
        )

        # Create pipeline
        pipeline_name = f"YOLOv8DriverActionPipeline"
        
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[param for param in parameters.values()],
            steps=[
                preprocessing_step,
                training_step,
                evaluation_step,
                condition,
                compilation_step
            ],
            sagemaker_session=self.pipeline_session
        )

        return pipeline

def main():
    try:
        # Config paths from S3
        aws_config_path = "s3://pw-fleet/config/aws_config.yaml"
        #print(f"Loading config from: {aws_config_path}")  # Add this line
        model_config_path = "s3://pw-fleet/config/config.yaml"
        
        logger.info("Initializing YOLOv8 Pipeline...")
        pipeline_instance = YOLOv8Pipeline(
            aws_config_path=aws_config_path,
            model_config_path=model_config_path
        )
        
        # Create pipeline
        logger.info("Creating pipeline...")
        pipeline = pipeline_instance.create_pipeline()
        
        # Upsert pipeline
        logger.info("Upserting pipeline...")
        pipeline.upsert(role_arn=pipeline_instance.role)
        
        # Start pipeline execution
        logger.info("Starting pipeline execution...")
        execution = pipeline.start()
        
        logger.info(f"Pipeline started successfully. Execution ARN: {execution.arn}")
        
    except Exception as e:
        logger.error(f"Pipeline creation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()