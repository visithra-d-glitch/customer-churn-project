import pandas as pd

df = pd.read_csv("data/telco.csv")
print(df.shape)
print(df.head())
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})

df["Offer"] = df["Offer"].fillna("No Offer")
df["Internet Type"] = df["Internet Type"].fillna("No Internet")

drop_cols = ["Customer ID", "Count", "Country", "State", "City",
             "Zip Code", "Lat Long", "Latitude", "Longitude",
             "Customer Status", "Churn Label", "Churn Value",
             "Churn Score", "Churn Category", "Churn Reason",
             "Satisfaction Score", "CLTV"]
df = df.drop(columns=drop_cols, errors="ignore")

print(df.shape)
print(df["Churn"].value_counts(normalize=True))
print("Missing:", df.isnull().sum().sum())

sns.countplot(x="Churn", data=df)
plt.show()

sns.countplot(x="Contract", hue="Churn", data=df)
plt.show()