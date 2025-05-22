# Análise de Ações da Uber (2019-2025) 

Este repositório contém dados e análises relacionadas à previsão do comportamento das ações da Uber Technologies Inc. (UBER), desenvolvido como parte do Projeto Integrador II do curso de Tecnologia em Ciência de Dados da Universidade Presbiteriana Mackenzie. A análise visa aplicar métodos estatísticos e de aprendizado de máquina para compreender e prever tendências do mercado de ações da Uber.

# Visão Geral
A Uber é uma das empresas que mais transformaram o setor de mobilidade urbana nas últimas décadas, redefinindo não apenas o transporte, mas também as relações trabalhistas através da "uberização". Este projeto aborda:

- Análise exploratória dos dados históricos das ações da Uber (maio/2019 - fevereiro/2025)
- Modelagem preditiva utilizando técnicas de séries temporais (ARIMA)
- Avaliação de performance através de métricas de acurácia e erro
- Desenvolvimento do produto "Uber Insights" - plataforma de apoio à decisão para investidores

## Objetivos e Metas
Nosso objetivo é realizar uma análise exploratória abrangente, seguida da aplicação de técnicas de aprendizado de máquina para prever o comportamento das ações da Uber. As metas específicas incluem:

1. **Análise Exploratória:** Examinar os dados para identificar padrões, tendências e possíveis outliers.

2. **Previsão de Preços:** Treinar um modelo de aprendizado de máquina para prever o preço das ações da Uber, utilizando técnicas como regressão linear e modelos de séries temporais.

3. **Identificação de Padrões:** Aplicar métodos de regressão para identificar padrões nos preços históricos das ações e analisar fatores que impactam essas variações.

4. **Classificação de Preços:** Desenvolver um modelo de classificação que prevê se o preço de fechamento das ações será alto ou baixo, com base em variáveis relevantes.


# Principais Descobertas
A análise revelou insights importantes sobre o comportamento das ações da Uber:

- Crescimento consistente: Valorização de 55,11% no período analisado, com tendência de alta de $0,015 por dia
- Alta volatilidade: Desvio padrão dos retornos diários de 3,36%, com variações extremas de até 38,26%
- Sazonalidade identificada: Preços médios mais altos em fevereiro ($49,03) e mais baixos em maio ($42,02)
- Boa liquidez: Volume médio diário de 24,3 milhões de ações negociadas

# Modelo de Previsão
O modelo ARIMA (4,1,0) desenvolvido apresentou:

- Acurácia global: 90,08% (MAPE de 9,92%)
- Eficácia em tendências: Boa capacidade para previsões de médio e longo prazo
- Limitações: Dificuldade em capturar oscilações abruptas de curto prazo devido à alta volatilidade do ativo

# Produto Gerado: Uber Insights
Como resultado prático da pesquisa, foi conceptualizado o Uber Insights - uma plataforma de apoio à decisão que oferece:

- Previsão de tendências de preços
- Monitoramento de volatilidade anormal
- Dashboard com análise de risco baseada em dados históricos
- Relatórios preditivos customizados para investidores individuais e institucionais

## Metadados
Utilizaremos a base de dados "Uber Stocks Dataset 2025" disponível no Kaggle, que contém informações textuais relacionadas ao desempenho das ações da Uber. Os metadados incluirão a origem dos dados, a estrutura textual, como colunas que representam preços de abertura, fechamento, volume de ações, datas e a descrição dos indicadores. Essa apresentação nos permitirá entender melhor a qualidade e a relevância dos dados para nossa análise.

## Estrutura do Projeto

- `database/`: Contém os arquivos de dados utilizados no projeto.
- `scripts/`: Contém os scripts Python para análise e modelagem.
- `requirements.txt`: Lista de dependências do projeto.

## Scripts

### 1. analise_exploratoria.py
`local: src\analise_exploratoria\analise_exploratoria.py`

Este script realiza uma análise exploratória detalhada dos dados históricos das ações da Uber.

## Funcionalidades

- Carrega dados históricos das ações da Uber a partir de um arquivo CSV
- Calcula métricas importantes como retornos diários, amplitude de preços e variação percentual
- Realiza análise de tendência utilizando regressão linear
- Testa a estacionariedade da série temporal com o teste Dickey-Fuller
- Identifica outliers no volume de negociação
- Analisa padrões sazonais por mês e ano

## Principais Visualizações

1. Tendência e Crescimento - Mostra a evolução do preço das ações com tendência linear e médias móveis
2. Volatilidade e Risco - Apresenta os retornos diários com destaque para variações extremas
3. Sazonalidade e Padrões - Exibe o preço médio por mês do ano
4. Volume e Liquidez - Mostra o volume de negociações diárias com outliers destacados

O script gera insights detalhados sobre evolução de preço, volatilidade, volume de negociação, estacionariedade e padrões sazonais das ações da Uber.

### 2. normalizacao.py
`local: src\normalizacao\normalizacao.py`

Este script prepara os dados para modelagem, realizando normalização e tratamento de outliers.

## Funcionalidades

- Carrega dados das ações da Uber a partir de um arquivo CSV
- Trata valores ausentes utilizando diferentes estratégias (média, remoção, preenchimento)
- Remove outliers com base no Z-score (valores que se afastam mais de 3 desvios padrões da média)
- Extrai informações temporais da coluna de data (ano, mês, dia)
- Normaliza os dados numéricos utilizando StandardScaler (padronização)
- Visualiza as distribuições antes e depois do tratamento
- Salva os dados tratados em um novo arquivo CSV

O script é fundamental para garantir que os dados estejam limpos e padronizados antes da aplicação de modelos de aprendizado de máquina ou análise estatística.

### 3. Aplicação do método analítico.ipynb
`local: src\metodo_analitico\Aplicação do método analítico.ipynb`

Este notebook implementa um modelo ARIMA para previsão do preço de ações da Uber.

## Funcionalidades

- Instala e importa bibliotecas necessárias (numpy, pandas, matplotlib, statsmodels, pmdarima)
- Carrega dados tratados das ações da Uber
- Verifica a inexistência de valores ausentes
- Plota os valores de fechamento das ações ao longo do tempo
- Realiza teste de estacionariedade utilizando o teste Dickey-Fuller
- Aplica decomposição da série temporal para identificar tendência e sazonalidade
- Utiliza auto_arima para determinar os melhores parâmetros do modelo ARIMA
- Divide os dados em conjuntos de treinamento e teste
- Treina o modelo ARIMA com os parâmetros otimizados
- Realiza previsões e avalia o desempenho do modelo

O notebook implementa uma abordagem completa de análise de séries temporais, desde a preparação dos dados até a avaliação do modelo preditivo.

### 4. medidas_de_acuracia.py
`local:src\metodo_analitico\medidas de acuracia.py`

Este script avalia a qualidade das previsões geradas pelo modelo ARIMA.

## Funcionalidades

- Carrega os dados reais e previstos das ações da Uber
- Mescla os dados por data para facilitar a comparação
- Calcula métricas de erro:
  - MAE (Erro Médio Absoluto)
  - RMSE (Raiz do Erro Quadrático Médio)
  - MAPE (Erro Percentual Absoluto Médio)
  - R² (Coeficiente de Determinação)
- Calcula métricas de acurácia:
  - Acurácia global (100% - MAPE)
  - Acurácia da direção (% de acerto na tendência)
  - Percentual de previsões com erro abaixo de limiares (5% e 10%)
- Gera visualizações:
  - Comparação entre valores reais e previstos
  - Gráfico de dispersão
  - Distribuição das previsões por faixa de erro
- Salva os resultados em arquivo de texto e as visualizações em imagens

O script fornece uma avaliação abrangente do desempenho do modelo, identificando pontos fortes e limitações das previsões realizadas.

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
`pip install -r requirements.txt`


## Como Executar

1. Clone o repositório
2. Instale as dependências
3. Execute os scripts na ordem: analise_exploratoria.py, normalizacao.py, "Aplicação do método analítico.ipynb", medidas_de_acuracia.py

## Contribuições

Membros do grupo:
- Gustavo Jose Fermiano
- Kelly Haro Vasconcellos 
- Wesley Rodrigo Dos Santos

## Links
- Link da apresentação no YouTube
- [Material de Apoio PPT](https://docs.google.com/presentation/d/1TCHCQ4E6fIqW0cbiiqHnzFWHVUIJ_e-Hx0h2g3yDyW8/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45)


## Referências
- Uber 
- Uber Stocks Dataset 2025 (Kaggle)
- ADEBIYI et al. - Stock Price Prediction Using the ARIMA Model (2014)
- IBM - Apresentando os modelos ARIMA (2024)
- Analytics Vidhya - Stock Market Forecasting using Time Series Analysis (2024)
- Alura - Métricas de avaliação para séries temporais (2021)
