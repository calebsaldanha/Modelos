from azureml.core import Workspace, Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment

# Carregar dados
ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, name='diabetes_dataset')
train_data = dataset.to_pandas_dataframe()

# Configuração do AutoML
automl_config = AutoMLConfig(
    task='classification',
    primary_metric='accuracy',
    X=train_data.drop(columns=['target']),
    y=train_data['target'],
    n_cross_validations=5
)

# Executar o experimento
experiment = Experiment(ws, 'diabetes_experiment')
run = experiment.submit(automl_config)
