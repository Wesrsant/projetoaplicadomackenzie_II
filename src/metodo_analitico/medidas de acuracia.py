import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Obter o diretório onde o script está localizado
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

# Definir caminhos para os arquivos
caminho_dados_reais = os.path.join(diretorio_atual, 'dados_tratados_sem_normalizacao.csv')
caminho_dados_previstos = os.path.join(diretorio_atual, 'previsao_uber.csv')

# Carregar os dados
df_real = pd.read_csv(caminho_dados_reais)
df_previsto = pd.read_csv(caminho_dados_previstos)

# Converter as colunas de data para o formato datetime
df_real['Date'] = pd.to_datetime(df_real['Date'])
df_previsto['Date'] = pd.to_datetime(df_previsto['Date'])

# Mesclar os dataframes pelo campo de data
df_merged = pd.merge(df_real, df_previsto, on='Date', how='inner')
print(f"Número de datas correspondentes: {len(df_merged)}")

# Calcular as métricas
y_real = df_merged['Close']
y_previsto = df_merged['Close Previsto']

# Métricas de erro
mae = mean_absolute_error(y_real, y_previsto)
mse = mean_squared_error(y_real, y_previsto)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_real - y_previsto) / y_real)) * 100
r2 = r2_score(y_real, y_previsto)

# Métricas de acurácia
acuracia = 100 - mape  # Acurácia baseada no MAPE
direcao_correta = np.mean((y_real.diff() > 0) == (y_previsto.diff() > 0)) * 100  # % de acerto na direção
correlacao = np.corrcoef(y_real, y_previsto)[0, 1]

# Calcular % de previsões com erro abaixo de limiares
erro_percentual = np.abs((y_real - y_previsto) / y_real) * 100
previsoes_erro_5 = np.sum(erro_percentual < 5) / len(erro_percentual) * 100
previsoes_erro_10 = np.sum(erro_percentual < 10) / len(erro_percentual) * 100

# Imprimir os resultados
print("\nMétricas de Avaliação do Modelo ARIMA:")
print("\n1. Métricas de Erro:")
print(f"   - MAE: {mae:.4f}")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - MAPE: {mape:.2f}%")
print(f"   - R²: {r2:.4f}")

print("\n2. Métricas de Acurácia:")
print(f"   - Acurácia global (100% - MAPE): {acuracia:.2f}%")
print(f"   - Acurácia da direção (tendência): {direcao_correta:.2f}%")
print(f"   - Previsões com erro < 5%: {previsoes_erro_5:.2f}%")
print(f"   - Previsões com erro < 10%: {previsoes_erro_10:.2f}%")
print(f"   - Correlação: {correlacao:.4f}")

# Visualizações
plt.figure(figsize=(12, 6))
plt.plot(df_merged['Date'], y_real, label='Valor Real', color='blue')
plt.plot(df_merged['Date'], y_previsto, label='Valor Previsto', color='red')
plt.title('Comparação entre Valores Reais e Previstos - Ações da Uber')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(diretorio_atual, 'comparacao_valores.png'))

# Gráfico de dispersão
plt.figure(figsize=(8, 8))
plt.scatter(y_real, y_previsto, alpha=0.5)
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Previstos')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(diretorio_atual, 'dispersao_valores.png'))

# Gráfico de barras para acurácia por faixas de erro
plt.figure(figsize=(8, 6))
categorias = ['Erro < 5%', 'Erro 5-10%', 'Erro > 10%']
valores = [
    previsoes_erro_5,
    previsoes_erro_10 - previsoes_erro_5,
    100 - previsoes_erro_10
]
cores = ['green', 'orange', 'red']
plt.bar(categorias, valores, color=cores)
plt.title('Distribuição das Previsões por Faixa de Erro')
plt.ylabel('Porcentagem de Previsões (%)')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(valores):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(diretorio_atual, 'distribuicao_acuracia.png'))

# Salvar resultados em um arquivo de texto
with open(os.path.join(diretorio_atual, 'resultados_acuracia.txt'), 'w') as f:
    f.write("Métricas de Avaliação do Modelo ARIMA - Previsão de Ações da Uber\n")
    f.write("=================================================================\n\n")
    
    f.write("1. Métricas de Erro:\n")
    f.write(f"- MAE (Erro Médio Absoluto): {mae:.4f}\n")
    f.write(f"- RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}\n")
    f.write(f"- MAPE (Erro Percentual Absoluto Médio): {mape:.2f}%\n")
    f.write(f"- R² (Coeficiente de Determinação): {r2:.4f}\n\n")
    
    f.write("2. Métricas de Acurácia:\n")
    f.write(f"- Acurácia global (100% - MAPE): {acuracia:.2f}%\n")
    f.write(f"- Acurácia da direção (tendência): {direcao_correta:.2f}%\n")
    f.write(f"- Previsões com erro < 5%: {previsoes_erro_5:.2f}%\n")
    f.write(f"- Previsões com erro < 10%: {previsoes_erro_10:.2f}%\n")
    f.write(f"- Correlação entre valores reais e previstos: {correlacao:.4f}\n\n")
    
    f.write("3. Interpretação dos Resultados:\n")
    f.write(f"- O modelo apresenta uma acurácia global de {acuracia:.2f}%\n")
    f.write(f"- {previsoes_erro_10:.1f}% das previsões têm erro menor que 10%\n")
    f.write(f"- O modelo prevê corretamente a direção do movimento em {direcao_correta:.2f}% dos casos\n\n")
    
    f.write("4. Pontos Fortes e Limitações:\n")
    f.write("Pontos Fortes:\n")
    f.write(f"- Capacidade de prever a direção do movimento em {direcao_correta:.2f}% dos casos\n")
    f.write(f"- {previsoes_erro_10:.1f}% das previsões com erro menor que 10%\n\n")
    
    f.write("Limitações:\n")
    f.write("- Dificuldade em capturar a volatilidade do mercado de ações\n")
    f.write("- Não considera fatores externos como notícias e eventos econômicos\n")

print("\nAnálise concluída. Arquivos gerados no diretório:", diretorio_atual)
