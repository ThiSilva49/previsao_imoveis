import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregar os dados
df = pd.read_csv("housing.csv")

# Preencher valores ausentes com a mediana da coluna
df.fillna(df.median(numeric_only=True), inplace=True)

# Transformar variáveis categóricas em numéricas
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Separar features e target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Verificar se ainda há valores NaN
print("Valores ausentes após tratamento:\n", X.isnull().sum().sum())  # Deve imprimir 0

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos a serem testados
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Regressão Linear": LinearRegression()
}

# Dicionário para armazenar os resultados
resultados = {}

# Treinar e avaliar os modelos
for nome, modelo in models.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Raiz quadrada do MSE manualmente

    resultados[nome] = {"MAE": mae, "RMSE": rmse}

# Converter para DataFrame
resultados_df = pd.DataFrame(resultados).T

# Exibir os resultados no console
print("\n📊 Comparação de Modelos:")
print(resultados_df)

# Criar gráfico de comparação de erros
plt.figure(figsize=(10, 5))

# Plotar as barras de MAE e RMSE, com diferentes posições no eixo X
bar_width = 0.35  # Largura das barras
index = resultados_df.index

# Plotar MAE (usando azul)
plt.bar(index, resultados_df["MAE"], color="blue", width=bar_width, label="MAE")

# Plotar RMSE (usando vermelho) deslocando um pouco para a direita
plt.bar([x + bar_width for x in range(len(index))], resultados_df["RMSE"], color="red", width=bar_width, label="RMSE")

# Adicionar título e rótulos
plt.title("Comparação de Modelos - Erros")
plt.ylabel("Erro")
plt.xticks([x + bar_width / 2 for x in range(len(index))], index)  # Ajustar as posições das labels no eixo X

# Adicionar a legenda
plt.legend(title="Métricas")

# Exibir gráfico
plt.show()
