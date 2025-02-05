import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housing.csv")

plt.figure(figsize=(8, 5))
sns.histplot(df["median_house_value"], bins=30, kde=True, color='blue')
plt.xlabel("Preço dos Imóveis")
plt.ylabel("Frequência")
plt.title("Distribuição dos Preços dos Imóveis")
plt.show() 