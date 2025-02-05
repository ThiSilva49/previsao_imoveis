import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housing.csv")
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor das Correlações")
plt.show()
