# Análise de Ações da Uber (2019-2025) 

## Metadados
Utilizaremos a base de dados "Uber Stocks Dataset 2025" disponível no Kaggle, que contém informações textuais relacionadas ao desempenho das ações da Uber. Os metadados incluirão a origem dos dados, a estrutura textual, como colunas que representam preços de abertura, fechamento, volume de ações, datas e a descrição dos indicadores. Essa apresentação nos permitirá entender melhor a qualidade e a relevância dos dados para nossa análise.

## Objetivos e Metas
Nosso objetivo é realizar uma análise exploratória abrangente, seguida da aplicação de técnicas de aprendizado de máquina para prever o comportamento das ações da Uber. As metas específicas incluem:

1. **Análise Exploratória:** Examinar os dados para identificar padrões, tendências e possíveis outliers.

2. **Previsão de Preços:** Treinar um modelo de aprendizado de máquina para prever o preço das ações da Uber, utilizando técnicas como regressão linear e modelos de séries temporais.

3. **Identificação de Padrões:** Aplicar métodos de regressão para identificar padrões nos preços históricos das ações e analisar fatores que impactam essas variações.

4. **Classificação de Preços:** Desenvolver um modelo de classificação que prevê se o preço de fechamento das ações será alto ou baixo, com base em variáveis relevantes.

## Estrutura do Projeto

- `database/`: Contém os arquivos de dados utilizados no projeto.
- `scripts/`: Contém os scripts Python para análise e modelagem.
- `requirements.txt`: Lista de dependências do projeto.

## Scripts

### 1. analise_exploratoria.py
local: src\analise_exploratoria\analise_exploratoria.py

Realiza uma análise exploratória detalhada dos dados históricos das ações da Uber.

Funcionalidades:
- Análise de tendências e crescimento
- Estudo de volatilidade e risco
- Identificação de padrões sazonais
- Análise de volume e liquidez

### 2. normalizacao.py
local: src\normalizacao\normalizacao.py

Prepara os dados para modelagem, realizando normalização e tratamento de outliers.

Funcionalidades:
- Tratamento de valores ausentes
- Remoção de outliers
- Normalização de variáveis numéricas
- Visualização de distribuições antes e depois do tratamento

### 3. Aplicação do método analítico.ipynb
local: src\metodo_analitico\Aplicação do método analítico.ipynb

Este notebook implementa um modelo ARIMA para previsão do preço de ações da Uber.

Funcionalidades:
- Carregamento e preparação de dados
- Análise de estacionariedade
- Modelagem ARIMA
- Previsão de preços futuros

### 4. medidas_de_acuracia.py
local:src\metodo_analitico\medidas de acuracia.py

Avalia a qualidade das previsões geradas pelo modelo ARIMA.

Funcionalidades:
- Cálculo de métricas de erro: MAE, RMSE, MAPE
- Avaliação da acurácia direcional
- Visualização de resultados

## Requisitos e Instalação

O arquivo `requirements.txt` contém todas as dependências necessárias para executar os scripts. As principais bibliotecas utilizadas incluem:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- pmdarima

Para instalar as dependências, execute:
pip install -r requirements.txt


## Como Executar

1. Clone o repositório
2. Instale as dependências
3. Execute os scripts na ordem: analise_exploratoria.py, normalizacao.py, "Aplicação do método analítico.ipynb", medidas_de_acuracia.py

## Contribuições

Membros do grupo:
- Gustavo Jose Fermiano
- Kelly Haro Vasconcellos 
- Wesley Rodrigo Dos Santos

## Referências

- Uber Stocks Dataset 2025 (Kaggle)
- Documentação das bibliotecas utilizadas (pandas, scikit-learn, statsmodels, etc.)


## Contribuições
Membros do grupo:

* Gustavo Jose Fermiano
* Kelly Haro Vasconcellos 
* Wesley Rodrigo Dos Santos 



## Referências
- Uber Stocks Dataset 2025 (Kaggle)
- Documentação das bibliotecas utilizadas (pandas, scikit-learn, statsmodels, etc.)