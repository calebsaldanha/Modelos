# Projeto de Classificação com Azure Machine Learning AutoML

## 1. Descrição do Projeto
Este projeto utiliza o Azure Machine Learning para treinar um modelo de classificação de dados. O objetivo é prever a ocorrência de diabetes com base em um conjunto de dados com diversas características médicas. Utilizamos o AutoML para explorar diferentes modelos de machine learning, avaliá-los e escolher o mais adequado para a previsão. O processo inclui a automação do treinamento de modelos, validação e a exportação do modelo treinado.

## 2. Estrutura do Projeto
A estrutura do projeto é organizada da seguinte maneira:

```
project/
│
├── model/
│   └── model.pkl         # Modelo treinado (salvo após execução)
│
├── src/
│   ├── train_model.py    # Código para treinamento do modelo
│   └── inference.py      # Código para fazer previsões com o modelo treinado
│
├── environment.yml       # Especificação do ambiente Conda
│
└── README.md             # Documentação do projeto
```

### Descrição dos Arquivos

- **model/**: Contém o arquivo do modelo treinado (`model.pkl`), que é gerado após a execução do treinamento.
- **src/train_model.py**: Contém o script Python que utiliza o Azure Machine Learning AutoML para treinar o modelo.
- **src/inference.py**: Código para realizar previsões utilizando o modelo treinado.
- **environment.yml**: Arquivo Conda que define o ambiente com todas as dependências necessárias.
- **README.md**: Documentação do projeto.

## 3. Como Executar o Projeto

### 3.1 Configuração do Ambiente

Antes de executar o projeto, é necessário configurar o ambiente Conda para garantir que todas as dependências estejam instaladas corretamente.

#### Criar o ambiente Conda:

No terminal, execute o seguinte comando para criar o ambiente Conda utilizando o arquivo `environment.yml`:

```bash
conda env create -f environment.yml
```

Isso criará um ambiente Conda com todas as dependências listadas, incluindo as bibliotecas necessárias como `azureml`, `xgboost`, `prophet`, `scikit-learn`, entre outras.

#### Ativar o ambiente:

Após a criação do ambiente, ative-o com o seguinte comando:

```bash
conda activate project_environment
```

### 3.2 Executando o Treinamento do Modelo

Após configurar o ambiente, você pode treinar o modelo utilizando o script `train_model.py`. Este script realiza as seguintes etapas:

1. **Importação dos dados**: Carrega o conjunto de dados necessário para o treinamento.
2. **Treinamento com AutoML**: Utiliza o serviço Azure Machine Learning AutoML para explorar e treinar múltiplos modelos de machine learning.
3. **Seleção do melhor modelo**: O AutoML seleciona o modelo com melhor performance (acurácia, F1-Score, etc.).
4. **Salvamento do modelo treinado**: O modelo treinado é salvo como `model.pkl`.

Execute o script `train_model.py` com o seguinte comando:

```bash
python src/train_model.py
```

### 3.3 Realizando Previsões

Para realizar previsões com o modelo treinado, utilize o script `inference.py`. Ele permite que você forneça novos dados e obtenha as previsões. O modelo estará disponível após ser treinado, e o código de inferência carrega esse modelo para fazer as previsões.

Execute o script `inference.py` com o seguinte comando:

```bash
python src/inference.py
```

Isso retornará as previsões geradas pelo modelo treinado.

## 4. Resultados de Execução do Modelo

Durante a execução do treinamento, o Azure Machine Learning AutoML experimenta diferentes algoritmos e configurações para encontrar o melhor modelo. Os resultados de desempenho dos modelos podem ser visualizados como segue:

### Modelos Gerados pelo AutoML

- **VotingEnsemble**
  - **Acurácia**: 0.95300
  - **Algoritmos**: ['XGBoostClassifier', 'LightGBM']
  - **Tempo de execução**: 49s

- **StackEnsemble**
  - **Acurácia**: 0.95290
  - **Algoritmos**: ['XGBoostClassifier', 'LightGBM']
  - **Tempo de execução**: 52s

- **MaxAbsScaler, LightGBM**
  - **Acurácia**: 0.95180
  - **Parâmetro**: min_data_in_leaf : 20
  - **Tempo de execução**: 36s

- **MaxAbsScaler, XGBoostClassifier**
  - **Acurácia**: 0.95180
  - **Parâmetro**: tree_method : auto
  - **Tempo de execução**: 41s

- **MaxAbsScaler, ExtremeRandomTrees**
  - **Acurácia**: 0.82810
  - **Parâmetros**: bootstrap : true, class_weight : balanced, criterion : gini, max_features : sqrt, min_samples_leaf : 0.01
  - **Tempo de execução**: 37s

A precisão é o principal indicador de desempenho dos modelos, e o **VotingEnsemble** foi o melhor modelo, com a maior acurácia.

## 5. Detalhamento do Código

### 5.1 Código de Treinamento do Modelo

O script `train_model.py` treina o modelo utilizando o AutoML do Azure. Abaixo, está um exemplo do código:

```python
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
```

### 5.2 Código de Inferência

O script `inference.py` carrega o modelo treinado e faz previsões:

```python
import joblib
import pandas as pd
import numpy as np

# Carregar o modelo
model = joblib.load('model/model.pkl')

# Nova entrada para previsão
input_data = pd.DataFrame({
    "Pregnancies": [1],
    "PlasmaGlucose": [85],
    "DiastolicBloodPressure": [66],
    "TricepsThickness": [29],
    "SerumInsulin": [0],
    "BMI": [26.6],
    "DiabetesPedigree": [0.351],
    "Age": [34]
})

# Realizar previsão
prediction = model.predict(input_data)
print(f'Prediction: {prediction}')
```

## 6. Arquivo Conda (`environment.yml`)

O arquivo `environment.yml` especifica as dependências necessárias para rodar o projeto. Abaixo está o conteúdo do arquivo:

```yaml
name: project_environment
dependencies:
  - python=3.9.21
  - pip:
    - azureml-train-automl-runtime==1.59.0
    - inference-schema
    - xgboost<=1.5.2
    - prophet==1.1.4
    - azureml-interpret==1.59.0
    - azureml-defaults==1.59.0
  - numpy==1.23.5
  - pandas==1.5.3
  - scikit-learn==1.5.1
  - holidays==0.68
  - psutil==5.9.3
channels:
  - anaconda
  - conda-forge
```

## 7. GitHub - Integração com Repositório

### 7.1 Repositório no GitHub

O código do projeto está hospedado no GitHub para facilitar o versionamento, colaboração e a execução do projeto em diferentes ambientes. Aqui estão as etapas para integrar o código com o GitHub:

#### Criar um repositório no GitHub:
1. Vá até [GitHub](https://github.com) e crie um novo repositório chamado `azureml-diabetes-prediction`.

#### Subir o código para o repositório:
Após configurar o repositório, siga as etapas abaixo:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/SEU_USUARIO/azureml-diabetes-prediction.git
git push -u origin master
```

Isso enviará todos os arquivos para o repositório remoto no GitHub.

### 7.2 Requisitos de GitHub Actions (opcional)

Se você quiser automatizar o processo de execução do projeto, pode configurar o GitHub Actions para rodar o treinamento do modelo automaticamente. Crie um arquivo de workflow em `.github/workflows/main.yml` com o seguinte conteúdo:

```yaml
name: AutoML Training Workflow

on:
  push:
    branches:
      - main

jobs:
  train_model:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run model training
      run: |
        python src/train_model.py
```

Isso vai configurar o GitHub Actions para automaticamente treinar o modelo toda vez que uma nova alteração for realizada no repositório.
