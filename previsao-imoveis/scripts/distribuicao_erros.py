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
y_pred = model.predict(X_test)
errors = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=30, kde=True, color='red')
plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.title("Distribuição dos Erros (Random Forest)")
plt.show()
