# https://archive.ics.uci.edu/dataset/45/heart+disease


from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

heart_disease = fetch_ucirepo(name = "heart disease")

df = pd.DataFrame(data= heart_disease.data.features)
df["target"] = heart_disease.data.targets


# Veri setindeki nun degerleri cikarmak icin
if df.isna().any().any():
    df.dropna(inplace=True)
    print("nan")
    
    
x = df.drop(["target"],axis = 1).values
y = df.target.values  


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1, random_state= 42)


log_reg = LogisticRegression(penalty="l2", C = 1, solver= "lbfgs", max_iter=100)
log_reg.fit(x_train, y_train)

accuracy = log_reg.score(x_test, y_test)
print("Logistic Regression Acc: ",accuracy)
