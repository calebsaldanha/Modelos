# Projeto de Classificação com Azure Machine Learning AutoML

Este projeto utiliza o Azure Machine Learning para treinar um modelo de classificação de dados. O objetivo é prever a ocorrência de diabetes com base em um conjunto de dados com diversas características médicas. Utilizamos o AutoML para explorar diferentes modelos de machine learning, avaliá-los e escolher o mais adequado para a previsão. O processo inclui a automação do treinamento de modelos, validação e a exportação do modelo treinado.

## Estrutura do Projeto

- **model/**: Contém o arquivo do modelo treinado (`model.pkl`), que é gerado após a execução do treinamento.
- **src/**:
  - **train_model.py**: Script para treinar o modelo utilizando Azure Machine Learning AutoML.
  - **inference.py**: Script para fazer previsões com o modelo treinado.
- **environment.yml**: Especificação do ambiente Conda com dependências necessárias.
- **README.md**: Documentação principal do projeto.

## Como Executar o Projeto

### 1. Configuração do Ambiente

Para configurar o ambiente Conda, execute o comando:

```bash
conda env create -f environment.yml
conda activate project_environment
2. Treinamento do Modelo
Para treinar o modelo, execute o script train_model.py:

bash
Copiar
Editar
python src/train_model.py
O modelo será treinado utilizando o AutoML e salvo como model.pkl.

3. Realizando Previsões
Para realizar previsões, execute o script inference.py:

bash
Copiar
Editar
python src/inference.py
Isso irá carregar o modelo treinado e realizar a previsão com os dados fornecidos.

Resultados de Execução do Modelo
O AutoML gera múltiplos modelos e seleciona o melhor com base em métricas de desempenho, como a acurácia. O melhor modelo gerado foi:

VotingEnsemble

Acurácia: 0.95300

Algoritmos: ['XGBoostClassifier', 'LightGBM']

Tempo de execução: 49s

vbnet
Copiar
Editar

### **5. Modelo Treinado (`model/model.pkl`)**

Este arquivo contém o modelo treinado após a execução do script `train_model.py`. Ele será gerado automaticamente após o treinamento e salvo na pasta `model/`.
