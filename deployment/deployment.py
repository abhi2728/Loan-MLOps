from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.webservice import AciWebservice, webservice
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
import traceback
import logging



def deploymodel():
    '''
    Deploy the LOAN ML Model to Azure ACI
    '''
    try:
        run = Run.get_context()
        ws = run.experiment.workspace
        env = Environment.from_conda_specification(name="loan", file_path="conda_dependencies.yml")
        inference_config = InferenceConfig(entry_script="score.py", environment=env)

        aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, description="Loan classification")
        service = Model.deploy(ws, "loan-uat", Model.list(ws), inference_config, aciconfig, overwrite=True)
        service.wait_for_deployment(True)
    except Exception as err:
        # traceback.print_exc()
        logging.error("Deployment Exception")