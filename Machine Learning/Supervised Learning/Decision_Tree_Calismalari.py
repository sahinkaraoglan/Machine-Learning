from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

#veri seti inceleme
iris = load_iris()

x = iris.data #features
y = iris.target #target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



#Decision tree oluştur ve train et
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42) #criterion-"entropy
tree_clf.fit(x_train, y_train)


#Decision Tree Evalution Test
y_pred = tree_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("iris veri seti ile egitilen modelin dogrulugu: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("conf matris: ")
print(conf_matrix)

plt.Figure(figsize=(15,10))
plot_tree(tree_clf, filled=True, feature_names= iris.feature_names, class_names=list(iris.target_names))
plt.show()

feature_importances = tree_clf.feature_importances_
feature_names = iris.feature_names

feature_importances_sorted = sorted(zip(feature_importances,feature_names),reverse= True)

for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}:{importance}")
    
    
    

# %%

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt
import numpy as np

#veri seti inceleme
iris = load_iris()

n_classes = len(iris.target_names)
plot_colors = "ryb"
    
for pairidx, pair in enumerate([0,1],[0,2],[0,3],[1,2],[1,3],[2,3]):
    x = iris.data[:,pair]
    y = iris.target
    
    clf = DecisionTreeClassifier().fit(x,y)
    ax = plt.Subplot(2,3,pairidx+1)
    plt.tight_layout(h_pad = 0.5, w_pad = 0.5, pad = 2.5)
    
    DecisionBoundaryDisplay.from_estimator(clf,
                                           x,
                                           cnap = plt.cm.RdY1BuBu,
                                           response_method="predict",
                                           ax = ax,
                                           xlabel = iris.feature_names[pair[0]],
                                           ylabel = iris.feature_names[pair[1]])
    
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y==i)
        plt.scatter(x[idx, 0], x[idx, 1], c = color, label = iris.target_names[i],
                    cnap= plt.cm.RdY1BuBu,
                    edgecolors="black")
        
plt.legend
    