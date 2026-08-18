import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./placement_predict_50k Dataset.csv")
df = df.dropna()

x = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("xX_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
