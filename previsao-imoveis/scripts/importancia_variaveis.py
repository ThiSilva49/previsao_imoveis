import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("housing.csv")
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X.columns

df_importancia = pd.DataFrame({"Variável": feature_names, "Importância": importances})
df_importancia = df_importancia.sort_values(by="Importância", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importância", y="Variável", data=df_importancia, palette="viridis")
plt.title("Importância das Variáveis no Random Forest")
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.show()
