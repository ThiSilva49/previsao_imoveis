import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housing.csv")

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["median_house_value"], color='blue')
plt.title("Boxplot dos Preços dos Imóveis")
plt.show()
