import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
from sklearn.neural_network import MLPClassifier
from copy import deepcopy

import random

def isSimilar(vector1, vector2, min_similarity):
    similar = 0
    for i in range(len(vector1)):
        if vector1[i] == vector2[i]:
            similar += 1
    
    if similar >= min_similarity:
        return True
    else:
        return False

#Read and Load the Datasets
d1 = pd.read_csv("d1")
d2 = pd.read_csv("d2")

iterations = 100
change_rate = 0.1


#Separate into Attributes (50 only) and Target Variables
y_d1 = d1.iloc[:,0]
x_d1 = d1.iloc[:,1:50]

y_d2 = d2.iloc[:,0]
x_d2 = d2.iloc[:,1:50]


#Converting target attribute into 10 classes from 100
y_d1 = [y//10 for y in y_d1]
y_d2 = [y//10 for y in y_d2]

dimensions = len(x_d1.columns)
#print(dimensions)

change_count = int(change_rate*dimensions)

#Define the ML Models (Both are Multi Layer Perceptrons)
clf1 = MLPClassifier(solver='sgd',hidden_layer_sizes=(5, 5), random_state=1, alpha=0.01)
clf2 = MLPClassifier(solver='sgd',hidden_layer_sizes=(5, 5), random_state=1, alpha=0.01)


#Train the models
clf1.fit(x_d1, y_d1)
clf2.fit(x_d2, y_d2)


#Generate Random Point
test_vector = np.random.choice([0, 1], size=dimensions)

for iter in range(iterations):  

    #Predict Probability Scores
    clf1_prob = clf1.predict_proba([test_vector]).flatten()
    clf2_prob = clf2.predict_proba([test_vector]).flatten()

    #Get KL Divergence
    #########P.S - KL Divergence is not symmteric ( which would be P and Q)

    divergence = entropy(clf1_prob,clf2_prob)
    print(divergence)  

    div_diff = 0

    temp_vector = deepcopy(test_vector)
    internal_iter = 0

    #Hill climb
    while(div_diff >= 0 and internal_iter < 100):
        change_pos = random.sample(list(range(dimensions)), change_count)

        for pos in change_pos:
            if temp_vector[pos] == 0:
                temp_vector[pos] = 1
            else:
                temp_vector[pos] = 0
        
        clf1_prob = clf1.predict_proba([temp_vector]).flatten()
        clf2_prob = clf2.predict_proba([temp_vector]).flatten()
        divergence_tempvect = entropy(clf1_prob,clf2_prob)
        
        div_diff = divergence_tempvect - divergence
        internal_iter += 1
    
    if div_diff < 0:
        test_vector = deepcopy(temp_vector)
    

clf1_prob = clf1.predict_proba([test_vector]).flatten()
clf2_prob = clf2.predict_proba([test_vector]).flatten()

print(test_vector)


#Check for similarity
similar_inputs_1 = []

for i in range(len(x_d1)):
    if isSimilar(test_vector, x_d1.iloc[i], int(0.7*dimensions)):
        similar_inputs_1.append(x_d1.iloc[i])


similar_inputs_2 = []

for i in range(len(x_d2)):
    if isSimilar(test_vector, x_d2.iloc[i], int(0.7*dimensions)):
        similar_inputs_2.append(x_d2.iloc[i])

#Points in training sets that are similar to the given vector
si_1 = [tuple(x) for x in similar_inputs_1]
si_2 = [tuple(x) for x in similar_inputs_2]
print(set(si_1).intersection(set(si_2)))
