# Projeto de Classificação com Azure Machine Learning AutoML

## Descrição do Projeto

Este projeto tem como objetivo a construção de um modelo de classificação utilizando o **Azure Machine Learning AutoML** para prever a ocorrência de diabetes com base em um conjunto de dados médicos. O Azure ML AutoML foi utilizado para explorar diferentes modelos de machine learning, avaliá-los e selecionar o melhor modelo para previsão.

A principal vantagem do AutoML é a automação do processo de treinamento, onde o Azure experimenta várias técnicas e hiperparâmetros para otimizar o desempenho do modelo. Isso reduz a necessidade de intervenção manual no ajuste de parâmetros e seleção de algoritmos, permitindo um desenvolvimento mais rápido e eficiente de modelos preditivos.

## Estrutura do Projeto

A organização do projeto está dividida nas seguintes pastas e arquivos principais:

- **model/**: Contém o modelo treinado, que foi salvo em formato `.pkl` após o treinamento.
- **src/**:
  - **train_model.py**: Contém o código responsável pelo treinamento do modelo utilizando o AutoML.
  - **inference.py**: Código utilizado para fazer previsões utilizando o modelo treinado.
- **environment.yml**: Especifica o ambiente Conda necessário para rodar o projeto, incluindo todas as bibliotecas necessárias.
- **README.md**: Documento que contém a explicação do projeto, objetivos e como ele foi realizado.

## Conceitos Abordados

### AutoML (Automated Machine Learning)

O **AutoML** é uma ferramenta que automatiza as etapas de treinamento de um modelo de machine learning, incluindo a escolha de algoritmos, ajuste de hiperparâmetros e avaliação de modelos. Isso permite que desenvolvedores e cientistas de dados se concentrem mais nos problemas de negócios, deixando as tarefas técnicas para a automação.

### Classificação

O modelo criado é do tipo **classificação**, ou seja, ele foi treinado para prever uma variável categórica. No caso deste projeto, a tarefa foi prever se um paciente tem ou não diabetes com base em um conjunto de variáveis médicas.

### Validação Cruzada

A técnica de **validação cruzada** foi utilizada para avaliar a performance dos modelos, garantindo que o modelo escolhido tenha uma boa generalização e não sofra de overfitting. No projeto, foi configurado para realizar 5 iterações de validação cruzada.

### Seleção de Modelos

O **Azure ML AutoML** explora diferentes algoritmos de machine learning, como XGBoost, LightGBM, entre outros, para encontrar o modelo mais adequado com base em métricas de desempenho como acurácia e F1-Score.

## Resultados Obtidos

Durante o processo de treinamento, o Azure ML experimentou diferentes combinações de algoritmos e hiperparâmetros para encontrar a melhor solução. Os principais modelos gerados foram:

- **VotingEnsemble**:
  - **Acurácia**: 95,3%
  - Algoritmos combinados: XGBoostClassifier, LightGBM
  - Este modelo foi o mais eficiente, combinando os pontos fortes de diferentes algoritmos para alcançar o melhor desempenho.

- **StackEnsemble**:
  - **Acurácia**: 95,29%
  - Algoritmos combinados: XGBoostClassifier, LightGBM
  - Este modelo teve desempenho muito próximo ao do VotingEnsemble, mostrando a eficácia de combinar diferentes modelos.

- **Outros Modelos**:
  - O AutoML também experimentou configurações como **MaxAbsScaler + XGBoostClassifier** e **MaxAbsScaler + ExtremeRandomTrees**, mas com desempenhos ligeiramente inferiores, com a maior acurácia atingindo 95,18%.

O modelo final escolhido foi o **VotingEnsemble**, devido à sua maior precisão.

## Explicação do Código

### Código de Treinamento do Modelo

```python
# %% [markdown]
# # Train a classification model with Automated Machine Learning
# 
# Azure Machine Learning enables you to automate the comparison of models trained using different algorithms and preprocessing options. You can use the visual interface in Azure Machine Learning Studio or the Python SDK to leverage this capability. 
# 
# ## Before you start
# You'll need the latest version of the **azure-ai-ml** package to run the code in this notebook.
# 
# If the **azure-ai-ml** package is not installed, run `pip install azure-ai-ml` to install it.
# 

# %% 
pip show azure-ai-ml

# %% [markdown]
# ## Connect to your workspace
# With the required SDK packages installed, now you're ready to connect to your workspace.

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)

# %% [markdown]
# ## Prepare data
# Prepare the diabetes dataset to use as input for the AutoML job.
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:diabetes-training:1")

# %% [markdown]
# ## Configure automated machine learning job
# Here, you configure the AutoML job with the settings for model training, such as the target column and the algorithms to exclude (e.g., LogisticRegression).
from azure.ai.ml import automl

classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True
)

classification_job.set_limits(
    timeout_minutes=60, 
    trial_timeout_minutes=20, 
    max_trials=5,
    enable_early_termination=True,
)

classification_job.set_training(
    blocked_training_algorithms=["LogisticRegression"], 
    enable_onnx_compatible_models=True
)

# %% 
returned_job = ml_client.jobs.create_or_update(classification_job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```

### Código de Inferência

```python
import joblib
from azureml.core import Workspace, Model

# Connect to the Workspace
ws = Workspace.from_config()

# Load the registered model
model_name = 'diabetes_modelo'
model_version = 1

model = Model(ws, name=model_name, version=model_version)

# Download the model to the local directory
model.download(target_dir='./', exist_ok=True)
print(f"Modelo {model_name} versão {model_version} foi baixado com sucesso.")
```

## Considerações Finais

Este projeto demonstra o poder do **AutoML** no Azure Machine Learning para a automação do treinamento de modelos de machine learning. Ao utilizar o AutoML, conseguimos selecionar rapidamente o melhor modelo para o problema de classificação de diabetes, sem a necessidade de ajustes manuais extensivos nos hiperparâmetros e algoritmos.
