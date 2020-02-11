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
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score

fileFeature = "fileFeatures1.tsv"
fileClasses = "fileClasses1.tsv"



'''
    This function is used to evaluate a classifier given as parametr.
    
    It computes:
    - Cross-validation
    - Accurancy
    - F-score
'''
def evaluate_classifier(classifier):
    print(cross_val_score(classifier, input, output, cv = cross_validation_iterations))

    classifier.fit(x_train, y_train)
    disp = plot_confusion_matrix(classifier, x_test, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title('Confusion Matrix')
    plt.show()

    y_predicetd = classifier.predict(x_test)
    accurancy = accuracy_score(y_test, y_predicetd)
    f_score = f1_score(y_test, y_predicetd, average='macro',  pos_label=3)
    print('accurancy : ' + str(accurancy))
    print('f_score : ' + str(f_score) + '\n')




# PREPARE DATA
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
    
# SPLIT DATA IN TRAININGSET AND TESTSET 
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.25, random_state=42)


# CROSS VALIDATION PARAMETR
cross_validation_iterations = 6



# --------------- DECISION TREE ---------------
DecTree = DecisionTreeClassifier()
evaluate_classifier(DecTree)



# --------------- BAYESS ---------------
bayess = GaussianNB()
evaluate_classifier(bayess)



# --------------- SVC ---------------
svc = SVC()
evaluate_classifier(svc)



# --------------- K-NN ---------------
k_neighbor = 5
knn = KNeighborsClassifier(k_neighbor)
evaluate_classifier(knn)



# --------------- ADABOOST ---------------
ada = AdaBoostClassifier()
evaluate_classifier(ada)



# --------------- RANDOM FOREST ---------------
randForest = RandomForestClassifier()
evaluate_classifier(randForest)


