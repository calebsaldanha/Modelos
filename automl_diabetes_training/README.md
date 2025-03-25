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

O treinamento do modelo foi realizado utilizando o **AutoMLConfig** do Azure Machine Learning, que permite configurar parâmetros como o tipo de tarefa (classificação), a métrica principal (acurácia), e o número de validações cruzadas. O modelo treinado foi exportado e salvo em formato `.pkl` utilizando a biblioteca `joblib`.

### Código de Inferência

O script **inference.py** foi criado para permitir que o modelo treinado seja utilizado em novas previsões. Após o treinamento, é possível carregar o modelo salvo e fazer previsões para novos dados. Por exemplo, dados de pacientes com características como **idade**, **nível de glicose no sangue**, e **índice de massa corporal** podem ser passados para o modelo, que retornará a previsão se o paciente tem ou não diabetes.

## Considerações Finais

Este projeto demonstra o poder do **AutoML** no Azure Machine Learning para a automação do treinamento de modelos de machine learning. Ao utilizar o AutoML, conseguimos selecionar rapidamente o melhor modelo para o problema de classificação de diabetes, sem a necessidade de ajustes manuais extensivos nos hiperparâmetros e algoritmos.
