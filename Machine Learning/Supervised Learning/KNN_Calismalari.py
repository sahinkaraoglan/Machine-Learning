#sklearn: ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


# (1) Veri setinin incelenmesi
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target





# (2) Makine ogrenmesi modelinin secilmesi- KNN siniflandirmasi





# (3) Modelin train edilmesi
x = cancer.data #features
y = cancer.target #target

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 42)

# ölçeklendirme
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#KNN modeli oluştur ve train et
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train) #fit fonksiyonu verilerimizi kullanarak knn algoritmamızı eğitir





# (4) Sonuclarin degerlendirilmesi : TEST
y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:",accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(conf_matrix)





# (5) Hiperparametrelerinin ayarlanmasi
accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)


plt.figure()
plt.plot(k_values, accuracy_values, marker = "o",linestyle ="-")
plt.title("K degerine gore dogruluk")
plt.xlabel("K degeri")
plt.ylabel("Dogruluk")
plt.xticks(k_values)
plt.grid(True)

#yukarda KNN ile sınıflandırma işlemi yapıldı.








# %%
# KNN regresyon işlemi yapıldı

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

x = np.sort(5* np.random.rand(40,1), axis = 0) #features
y = np.sin(x).ravel() #target

#plt.scatter(x,y)

#add noise
T = np.linspace(0, 5, 500)[:,np.newaxis]

for i, weight in enumerate(["uniform","distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(x,y).predict(T)
    
    plt.subplot(2,1,i+1)
    plt.scatter(x,y, color= "green", label = "data")
    plt.plot(T, y_pred, color= "blue",label = "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights =".format(weight))

plt.tight_layout()
plt.show()
