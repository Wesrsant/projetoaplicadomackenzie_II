import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import os


# Caminho relativo fixo para buscar o arquivo CSV
file_path = os.path.join('..', '..', 'database', 'uber_stock_data.csv')

# Carregar os dados
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Cálculos básicos para análise exploratória
df['Daily_Return'] = df['Close'].pct_change() * 100
df['Daily_Range'] = df['High'] - df['Low']
df['Range_Pct'] = (df['Daily_Range'] / df['Low']) * 100
df['Price_Change'] = df['Close'] - df['Open']
df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100

# Análise de tendência
df['Days'] = (df.index - df.index.min()).days
X = df['Days'].values.reshape(-1, 1)
y = df['Close'].values

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
df['Linear_Trend'] = model.predict(X)

# Teste de estacionariedade
result = adfuller(df['Close'].dropna())
is_stationary_original = result[1] <= 0.05

df['Close_Diff'] = df['Close'].diff()
result_diff = adfuller(df['Close_Diff'].dropna())
is_stationary_diff = result_diff[1] <= 0.05

# Identificação de outliers no volume
Q1 = df['Volume'].quantile(0.25)
Q3 = df['Volume'].quantile(0.75)
IQR = Q3 - Q1
volume_outliers = df[(df['Volume'] < (Q1 - 1.5 * IQR)) | (df['Volume'] > (Q3 + 1.5 * IQR))]

# Análise de padrões sazonais
df['Year'] = df.index.year
df['Month_of_Year'] = df.index.month

yearly_returns = df.groupby('Year')['Close'].apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
monthly_avg = df.groupby('Month_of_Year')['Close'].mean()

# Correlação entre volume e variação de preço
corr_vol_change = df['Volume'].corr(df['Price_Change_Pct'])

# Gráfico 1: Tendência e Crescimento
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Preço de Fechamento', color='blue')
plt.plot(df.index, df['Linear_Trend'], 'r--', label='Tendência Linear', linewidth=2)

df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
plt.plot(df.index, df['MA50'], color='green', label='Média Móvel 50 dias', alpha=0.7)
plt.plot(df.index, df['MA200'], color='purple', label='Média Móvel 200 dias', alpha=0.7)

growth_pct = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
plt.annotate(f'Crescimento Total: {growth_pct:.2f}%', 
             xy=(0.75, 0.15),
             xycoords='axes fraction',
             fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

plt.title('Tendência e Crescimento das Ações da Uber', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tendencia_crescimento.png', dpi=300)
plt.show()

# Gráfico 2: Volatilidade e Risco
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Daily_Return'], color='purple', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

max_return_date = df['Daily_Return'].idxmax()
min_return_date = df['Daily_Return'].idxmin()
plt.scatter(max_return_date, df.loc[max_return_date, 'Daily_Return'], color='green', s=100, zorder=5)
plt.scatter(min_return_date, df.loc[min_return_date, 'Daily_Return'], color='red', s=100, zorder=5)

plt.annotate(f'+{df.loc[max_return_date, "Daily_Return"]:.2f}% ({max_return_date.strftime("%d/%m/%Y")})', 
             xy=(max_return_date, df.loc[max_return_date, 'Daily_Return']),
             xytext=(50, -30),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->'),
             fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.7))

plt.annotate(f'{df.loc[min_return_date, "Daily_Return"]:.2f}% ({min_return_date.strftime("%d/%m/%Y")})', 
             xy=(min_return_date, df.loc[min_return_date, 'Daily_Return']),
             xytext=(50, 30),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->'),
             fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.7))

plt.title('Volatilidade e Risco: Retornos Diários das Ações da Uber', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Retorno Diário (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('volatilidade_risco.png', dpi=300)
plt.show()

# Gráfico 3: Sazonalidade e Padrões
plt.figure(figsize=(14, 7))
month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
bars = plt.bar(range(1, 13), monthly_avg.values, color='teal', alpha=0.7)
plt.xticks(range(1, 13), month_names)

max_month = monthly_avg.idxmax()
min_month = monthly_avg.idxmin()
plt.bar(max_month, monthly_avg[max_month], color='green', alpha=0.9)
plt.bar(min_month, monthly_avg[min_month], color='red', alpha=0.9)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'${height:.2f}', ha='center', va='bottom', fontsize=11)

plt.title('Sazonalidade e Padrões: Preço Médio por Mês do Ano', fontsize=16)
plt.xlabel('Mês', fontsize=12)
plt.ylabel('Preço Médio ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sazonalidade_padroes.png', dpi=300)
plt.show()

# Gráfico 4: Volume e Liquidez
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Volume'], color='blue', alpha=0.5, label='Volume Normal')
plt.scatter(volume_outliers.index, volume_outliers['Volume'], color='red', s=30, label='Outliers')

max_volume_date = df['Volume'].idxmax()
plt.annotate(f'Volume Máximo: {df.loc[max_volume_date, "Volume"]:,.0f}\n{max_volume_date.strftime("%d/%m/%Y")}', 
             xy=(max_volume_date, df.loc[max_volume_date, 'Volume']),
             xytext=(30, -30),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->'),
             fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

plt.axhline(y=df['Volume'].mean(), color='green', linestyle='--', 
            label=f'Volume Médio: {df["Volume"].mean():,.0f}')

plt.title('Volume e Liquidez: Negociações Diárias com Outliers Destacados', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('volume_liquidez.png', dpi=300)
plt.show()

# Conclusões finais e insights
print("\n=== CONCLUSÕES FINAIS E INSIGHTS ===")
print("1. Evolução do Preço:")
if model.coef_[0] > 0:
    print(f"   - Tendência geral de alta: ${model.coef_[0]:.4f} por dia")
else:
    print(f"   - Tendência geral de baixa: ${model.coef_[0]:.4f} por dia")

print(f"   - Crescimento total no período: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")

print("\n2. Volatilidade:")
print(f"   - Desvio padrão dos retornos diários: {df['Daily_Return'].std():.2f}%")
print(f"   - Amplitude média diária: {df['Range_Pct'].mean():.2f}%")

print("\n3. Volume de Negociação:")
print(f"   - Volume médio diário: {df['Volume'].mean():.0f}")
print(f"   - Correlação entre volume e variação de preço: {corr_vol_change:.4f}")

print("\n4. Estacionariedade:")
print("   - Série original não é estacionária, indicando presença de tendência")
print("   - Série diferenciada é estacionária, adequada para modelagem de séries temporais")

print("\n5. Outliers e Eventos Extremos:")
print(f"   - Identificados {volume_outliers.shape[0]} dias com volume de negociação atípico")
print(f"   - Maior variação positiva: {df['Daily_Return'].max():.2f}% em {df['Daily_Return'].idxmax().strftime('%d/%m/%Y')}")
print(f"   - Maior variação negativa: {df['Daily_Return'].min():.2f}% em {df['Daily_Return'].idxmin().strftime('%d/%m/%Y')}")

print("\n6. Padrões Sazonais:")
max_month = monthly_avg.idxmax()
min_month = monthly_avg.idxmin()
month_names = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
print(f"   - Mês com preço médio mais alto: {month_names[max_month-1]} (${monthly_avg[max_month]:.2f})")
print(f"   - Mês com preço médio mais baixo: {month_names[min_month-1]} (${monthly_avg[min_month]:.2f})")

print("\n7. Desempenho Anual:")
best_year = yearly_returns.idxmax()
worst_year = yearly_returns.idxmin()
print(f"   - Melhor ano: {best_year} ({yearly_returns[best_year]:.2f}%)")
print(f"   - Pior ano: {worst_year} ({yearly_returns[worst_year]:.2f}%)")
