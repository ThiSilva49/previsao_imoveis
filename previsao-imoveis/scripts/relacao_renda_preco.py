import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housing.csv")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["median_income"], y=df["median_house_value"], alpha=0.5, color="green")
plt.xlabel("Média de Renda das Famílias")
plt.ylabel("Preço dos Imóveis")
plt.title("Relação entre Renda e Preço dos Imóveis")
plt.show()
