import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier

fileFeature = "fileFeatures.tsv"
fileClasses = "fileClasses.tsv"

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
    
    
    
tree = DecisionTreeClassifier()

# TRAINING AND PREDICTION 
model = tree.fit(input, output)
        
print(model.predict_proba([[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] ]))



