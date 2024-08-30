from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)

y_pred = nb_clf.predict(x_test)
print(classification_report(y_test,y_pred))