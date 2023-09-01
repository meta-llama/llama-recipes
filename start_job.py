import datetime
from sagemaker.pytorch import PyTorch
import sagemaker
import os
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='...')['Role']['Arn']
print(role)

volume_size = 500
pytorch_estimator = PyTorch(
    entry_point="llama_finetuning.py", # the name of the script
    instance_type="ml.g5.12xlarge", 
    instance_count=2, # this determines the number of p4d instances
    source_dir=os.getcwd(),
    framework_version="1.11.0",
    py_version="py38",
    volume_size=volume_size,
    # dependencies=[''],
    region='us-west-2',
)
pytorch_estimator.fit(
    job_name='FSDP' + '-' + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
