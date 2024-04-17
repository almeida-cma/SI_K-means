# Script para agrupar dados e gerar reulstados dos agrupamentos
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carregar os dados do arquivo CSV
dados_clientes = pd.read_csv("arquivo.csv")

# Selecionar todas as features, exceto 'Grupo'
features = ['Idade', 'Genero', 'Valor_Medio_Compra', 'Num_Medio_Compras_por_Mes']

# Normalizar os dados
scaler = StandardScaler()
dados_clientes_normalizados = scaler.fit_transform(dados_clientes[features])

# Aplicar o algoritmo K-means
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(dados_normalizados)
labels = kmeans.labels_

# Adicionar os rótulos dos clusters ao DataFrame original
dados_clientes['Grupo_Kmeans'] = labels

# Salvar os resultados em um arquivo CSV
dados_clientes.to_csv("resultados_agrupamento.csv", index=False)

# ---------------------------------------------------------------------------------

# Scrip para representar graficamente os centróides
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carregar os dados do arquivo CSV
dados_clientes = pd.read_csv("resultados_agrupamento.csv")

# Separar os atributos dos clientes
X = dados_clientes[['Idade', 'Valor_Medio_Compra']]

# Inicializar e ajustar o modelo K-means
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)

# Adicionar os rótulos dos clusters ao DataFrame original
dados_clientes['Grupo_Kmeans'] = kmeans.labels_

# Plotar o gráfico
plt.figure(figsize=(10, 6))

# Cores para cada grupo
colors = ['blue', 'green', 'red']

# Plotar os pontos para cada grupo
for grupo, color in zip(dados_clientes['Grupo_Kmeans'].unique(), colors):
    subset = dados_clientes[dados_clientes['Grupo_Kmeans'] == grupo]
    plt.scatter(subset['Idade'], subset['Valor_Medio_Compra'], label=f'Grupo {grupo}', color=color, alpha=0.5)

# Plotar os centróides dos clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centróides')

plt.title('Agrupamento de Clientes pelo Algoritmo K-means')
plt.xlabel('Idade')
plt.ylabel('Valor Médio de Compra')
plt.legend()
plt.grid(True)
plt.show()
