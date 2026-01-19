METRICS_PATH = f's3://{MODEL_EVALUTION_PATH}/{EVALUATION_METRICS_FOLDER}/metrics.json'

register_estimator = Estimator(image_uri=PROCESSING_IMAGE,
                      role=ROLE,
                      instance_count=1,
                      instance_type='ml.t3.medium',
                      sagemaker_session=SAGEMAKER_SESSION
                     )

register_model_step = RegisterModel(
    name="YOLOTrainedModel",
    estimator=register_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=model_package_name,
    approval_status="PendingManualApproval",
    description=f"Model metrics available at {METRICS_PATH}"
)

condition_check = ConditionEquals(
    left=lambda_step.properties.Outputs['result'], 
    right=True
)

registry_condition_step = ConditionStep(
    name="YOLO-ModelRegistration",
    conditions=[condition_check],
    if_steps=[register_model_step],
    else_steps=[]
)