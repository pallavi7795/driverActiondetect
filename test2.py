import boto3
import sagemaker
import json
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CompilationStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.inputs import CompilationInput



# Set up the SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession()
bucket = sagemaker_session.default_bucket()

# Define pipeline parameters
model_path = ParameterString(
    name="ModelPath",
    default_value=f"s3://{bucket}/yolo-models/model.tar.gz"
)

target_instance_type = ParameterString(
    name="TargetInstanceType",
    default_value="ml_c5"  # Options: ml_c5, ml_p3, ml_inf1, etc.
)

framework = ParameterString(
    name="Framework",
    default_value="pytorch"  # Note: lowercase for CompilationInput
)

framework_version = ParameterString(
    name="FrameworkVersion",
    default_value="1.8"
)

output_path = ParameterString(
    name="OutputPath",
    default_value=f"s3://{bucket}/yolo-models/compiled-model/"
)

# Create a PyTorch estimator
pytorch_estimator = PyTorch(
    entry_point="inference.py",
    source_dir="code",
    role=role,
    framework_version="1.8",
    py_version="py3",
    instance_count=1,
    instance_type="ml.c5.xlarge",
    sagemaker_session=pipeline_session
)

# Create a PyTorch model 
pytorch_model = PyTorchModel(
    model_data=model_path, #tar.gz
    role=role,
    entry_point="inference.py", 
    framework_version="1.8",
    py_version="py3",
    sagemaker_session=pipeline_session
)

# Create compilation inputs with correct parameters
compilation_inputs = CompilationInput(
    target_instance_type=target_instance_type,
    input_shape=json.dumps({"input0": [1, 3, 416, 416]}),  # YOLO input shape as JSON string
    output_path=output_path,
    framework=framework,
    framework_version=framework_version,
    compile_max_run=900,  # 15 minutes
    job_name=f"yolo-compilation-{sagemaker_session.default_bucket_prefix(minutes=5, seconds=5)}",
    compiler_options=json.dumps({
        "trt-ver": "7.1.3",
        "gpu-code": "sm_70"  # For Tesla V100 GPUs (if using ml_p3)
    })
)

# Create compilation step
compilation_step = CompilationStep(
    name="YoloCompilation",
    estimator=pytorch_estimator,
    model=pytorch_model,
    inputs=compilation_inputs,
    depends_on=[],
    display_name="YOLO Model Compilation",
    description="Compile YOLO model for optimized inference"
)

# Create the pipeline
pipeline = Pipeline(
    name="YoloCompilationPipeline",
    parameters=[
        model_path,
        target_instance_type,
        framework,
        framework_version,
        output_path
    ],
    steps=[compilation_step],
    sagemaker_session=pipeline_session
)

# Submit the pipeline definition
pipeline.upsert(role_arn=role)

# Start the pipeline execution
execution = pipeline.start()

# Monitor the execution
print(f"Pipeline execution started with ARN: {execution.arn}")

#---------------------------------------------------------------------bck
def create_compilation_step_old(self, training_step, parameters):
        """Create a SageMaker Neo compilation step for the trained YOLOv8 model"""
        # Create a processor for running the compilation job
        processor = ScriptProcessor(
            image_uri=f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-training:{self.framework_version}-cpu-{self.py_version}",
            role=self.role,
            instance_count=1,
            instance_type="ml.c5.xlarge",  # Compilation doesn't need GPU
            base_job_name="yolov8-neo-compilation",
            command=["python3"],
            sagemaker_session=self.pipeline_session
        )
        
        # Define the compilation output path
        compilation_output_path = self.clean_s3_uri(
            self.bucket, 
            f"{parameters['model_prefix'].default_value}/compiled"
        )
        
        # Set up the processing step
        return ProcessingStep(
            name="CompileYOLOv8Model",
            processor=processor,
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/input/model"
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    output_name="compilation_output",
                    source="/opt/ml/processing/output",
                    destination=compilation_output_path
                )
            ],
            code="optimization/compile_neo.py",
            job_arguments=[
                "--model-data", training_step.properties.ModelArtifacts.S3ModelArtifacts,
                "--compilation-output-path", compilation_output_path,
                "--target-platform-os", "LINUX",
                "--target-platform-arch", "X86_64",
                "--framework", "PYTORCH"
            ],
            environment={
                "SAGEMAKER_ROLE": self.role
            },
            depends_on=[training_step.name]
        )