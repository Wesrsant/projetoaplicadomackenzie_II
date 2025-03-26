import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Caminho relativo do dataset
file_path = r"../../database/uber_stock_data.csv"


# Função para carregar os dados e tratar exceções
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f'Dados carregados com sucesso de {file_path}')
        print(df.head())
        return df
    except FileNotFoundError:
        print(f'Erro: O arquivo {file_path} não foi encontrado.')
        return None


# Carregar os dados
df = load_data(file_path)
if df is None:
    raise SystemExit('Erro ao carregar os dados. O processo será encerrado.')


# Tratamento de valores ausentes usando diferentes estratégias: 'mean', 'drop', 'forward_fill', 'backward_fill'
def handle_missing_values(df, strategy='mean'):
    print("Valores ausentes antes do tratamento:")
    print(df.isnull().sum())
    
    if strategy == 'mean':
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'drop':
        df = df.dropna()
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill')
    elif strategy == 'backward_fill':
        df = df.fillna(method='bfill')
    
    print("\nValores ausentes depois do tratamento:")
    print(df.isnull().sum())
    return df


df = handle_missing_values(df, strategy='mean')


# Remoção de outliers com base no Z-score (considerando como outliers valores com Z-score > 3)
def remove_outliers(df, columns):
    df_filtered = df.copy()  # Cria uma cópia do dataset
    rows_before = df_filtered.shape[0]
    
    for col in columns:
        if df_filtered[col].dtype in [np.float64, np.int64]:
            mean, std = df_filtered[col].mean(), df_filtered[col].std()
            df_filtered = df_filtered[np.abs(df_filtered[col] - mean) <= (3 * std)]
    
    rows_after = df_filtered.shape[0]
    print(f'Outliers removidos: {rows_before - rows_after} linhas')

    return df_filtered  # Retornar a versão filtrada corretamente


# Definindo as colunas para remoção de outliers (todas as colunas numéricas, exceto 'Year', 'Month' e 'Day')
outlier_columns = df.select_dtypes(include=[np.number]).columns.difference(['Year', 'Month', 'Day'])


# Removendo outliers com base nas colunas definidas
df = remove_outliers(df, outlier_columns)


# Visualizar antes e depois da remoção de outliers
df_no_outliers = remove_outliers(df.copy(), outlier_columns)


for col in outlier_columns:
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    df[col].hist(bins=50, alpha=0.7)
    plt.title(f'Antes da remoção de outliers: {col}')

    plt.subplot(1, 2, 2)
    df_no_outliers[col].hist(bins=50, alpha=0.7)
    plt.title(f'Depois da remoção de outliers: {col}')

    plt.show()


# Convertendo a coluna 'Date' para datetime e extraindo informações numéricas (ano, mês, dia)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day


# Exibição das estatísticas descritivas dos dados antes da normalização para entender a distribuição
print("\nEstatísticas descritivas antes da normalização:")
print(df.describe())


# Normalização dos dados (padronização) - excluindo a coluna 'Date' da normalização
def normalize_data(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(['Year', 'Month', 'Day'])
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# Normalizar os dados
df = normalize_data(df)


# Estatísticas descritivas após a normalização
print("\nEstatísticas descritivas após a normalização:")
print(df.describe())


# Visualização das distribuições antes e depois da normalização
for col in df.select_dtypes(include=[np.number]).columns.difference(['Year', 'Month', 'Day']):
    plt.figure(figsize=(10, 6))
    
    # Antes da normalização
    plt.subplot(1, 2, 1)
    df[col].hist(bins=50, alpha=0.7)
    plt.title(f'Antes da normalização: {col}')
    
    # Depois da normalização
    plt.subplot(1, 2, 2)
    df[col].hist(bins=50, alpha=0.7)
    plt.title(f'Depois da normalização: {col}')
    
    plt.show()


# Salvando os dados tratados em um novo arquivo CSV ('dados_tratados.csv') para uso posterior
if not df.empty:
    df.to_csv("dados_tratados.csv", index=False)
    print("Os dados tratados foram salvos em 'dados_tratados.csv'.")
else:
    print("Erro: DataFrame está vazio após o processamento!")
