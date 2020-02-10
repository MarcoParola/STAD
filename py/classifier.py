import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

fileFeature = "fileFeatures1.tsv"
fileClasses = "fileClasses1.tsv"

input = []
output = []

with open(fileFeature, 'r', encoding='utf-8') as features, open(fileClasses, 'r', encoding='utf-8') as classes:
    featureReader = csv.reader(features, delimiter = '\t')
    classReader = csv.reader(classes, delimiter = '\t')
    for row in classReader:
        output.append(int(row[0]))
        
        
    for row in featureReader:
        rowInt = []
        for i in range(0, len(row)): 
            if row[i] != '':
                rowInt.append(int(row[i]))
        input.append(rowInt)
    


# CROSS VALIDATION, cv specify the number of iterations

# DECISION TREE
DecTree = DecisionTreeClassifier()
print(cross_val_score(DecTree, input, output, cv = 6))


fig = DecTree.fit(input, output)
tree.plot_tree(fig)
plt.show()

# BAYESS
bayess = GaussianNB()
print(cross_val_score(bayess, input, output, cv = 6))


# SVC
svc = SVC()
print(cross_val_score(svc, input, output, cv = 6))


# K-NN (set k)
knn = KNeighborsClassifier(5)
print(cross_val_score(knn, input, output, cv = 6))


# ADABOOST
ada = AdaBoostClassifier()
print(cross_val_score(ada, input, output, cv = 6))


# RANDOM FOREST
randForest = RandomForestClassifier()
print(cross_val_score(randForest, input, output, cv = 6))



# TRAINING AND PREDICTION 
model = DecTree.fit(input, output)




# CONFUSION MATRIX
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.25, random_state=42)




DecTree.fit(x_train, y_train)
disp = plot_confusion_matrix(DecTree, x_test, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
plt.show()


bayess.fit(x_train, y_train)
disp = plot_confusion_matrix(bayess, x_test, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
plt.show()


svc.fit(x_train, y_train)
disp = plot_confusion_matrix(svc, x_test, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
plt.show()


knn.fit(x_train, y_train)
disp = plot_confusion_matrix(knn, x_test, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
plt.show()


ada.fit(x_train, y_train)
disp = plot_confusion_matrix(ada, x_test, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
plt.show()


randForest.fit(x_train, y_train)
disp = plot_confusion_matrix(randForest, x_test, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
plt.show()



