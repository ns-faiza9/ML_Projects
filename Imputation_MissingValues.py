import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

df = pd.read_csv("./placement_predict_50k Dataset.csv")
x = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

num_cols = x.select_dtypes(include=["number"]).columns
imp = SimpleImputer(strategy="mean")
x[num_cols] = imp.fit_transform(x[num_cols])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
