import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/telco.csv")



df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})
df["Offer"] = df["Offer"].fillna("No Offer")
df["Internet Type"] = df["Internet Type"].fillna("No Internet")

drop_cols = ["Customer ID", "Count", "Country", "State", "City",
             "Zip Code", "Lat Long", "Latitude", "Longitude",
             "Customer Status", "Churn Label", "Churn Value",
             "Churn Score", "Churn Category", "Churn Reason",
             "Satisfaction Score", "CLTV"]
df = df.drop(columns=drop_cols, errors="ignore")


X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n====", name, "====")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print(classification_report(y_test, pred))


joblib.dump(models["Random Forest"], "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")
feat = df.drop("Churn", axis=1)
num_cols = feat.select_dtypes(include="number").columns
cat_cols = feat.select_dtypes(exclude="number").columns
meta = {
    "defaults": {**feat[num_cols].median().to_dict(),
                 **{c: feat[c].mode()[0] for c in cat_cols}},
    "options": {c: sorted(feat[c].unique().tolist()) for c in cat_cols},
}
joblib.dump(meta, "meta.pkl")
print("Model saved")