import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
from collections import Counter

import random

max_similarity = 0

def isSimilar(vector1, vector2, min_similarity):
    global max_similarity
    similar = 0
    for i in range(len(vector1)):
        if vector1[i] == vector2[i+1]:
            similar += 1
    if similar >= max_similarity:
            max_similarity = similar
            print(similar)

    if similar >= min_similarity:        
        return True
    else:
        return False

#Read and Load the Datasets
d1 = pd.read_csv("d1")
d2 = pd.read_csv("d2")
#common = pd.read_csv("common")

iterations = 500
internal_iterations_max = 500
change_rate = 0.10

# d1.iloc[:,0] = d1.iloc[:,0]//10
# d2.iloc[:,0] = d2.iloc[:,0]//10

#Separate into Attributes (50 only) and Target Variables
y_d1 = d1.iloc[:,0]
x_d1 = d1.iloc[:,1:]

y_d2 = d2.iloc[:,0]
x_d2 = d2.iloc[:,1:]

# common_y = common.iloc[:,0]
# common_x = common.iloc[:,1:100]


#Am converting target attribute into 10 classes from 100
# y_d1 = [(y-1)//10 for y in y_d1]
# y_d2 = [(y-1)//10 for y in y_d2]

dimensions = len(x_d1.columns)
#print(dimensions)


#Define the ML Models (Both are Multi Layer Perceptrons)
clf1 = MLPClassifier(solver='sgd',hidden_layer_sizes=(30, 15), random_state=1, learning_rate='adaptive', learning_rate_init=0.01, max_iter=1000)
clf2 = MLPClassifier(solver='sgd',hidden_layer_sizes=(40, 20), random_state=1, learning_rate='adaptive', learning_rate_init=0.01, max_iter=1000)


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

    div_diff = 0

    temp_vector = deepcopy(test_vector)
    internal_iter = 0

    if int((iter/iterations)*100)%10 == 0:
        change_rate = change_rate*0.9

    #Hill climb
    while(div_diff >= 0 and internal_iter < internal_iterations_max):
        change_count = int(random.random()*change_rate*dimensions)
        if change_count < 1:
                change_count = 1

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
        print(divergence)


clf1_prob = clf1.predict_proba([test_vector]).flatten()
clf2_prob = clf2.predict_proba([test_vector]).flatten()

f = open("AdvStatsMLPMLP", "w+")

f.write("Test Vector:\n" + str(test_vector) + "\n")

#Check for similarity
min_similarity = 0.55
similar_inputs_1 = []

print("--1--")

for i in range(len(x_d1)):
    if isSimilar(test_vector, d1.iloc[i], int(min_similarity*dimensions)):
        similar_inputs_1.append(d1.iloc[i])

f.write("\nMax similarity 1: " + str(max_similarity))

similar_inputs_2 = []

print("--2--")
max_similarity = 0
for i in range(len(x_d2)):
    if isSimilar(test_vector, d2.iloc[i], int(min_similarity*dimensions)):
        similar_inputs_2.append(d2.iloc[i])

f.write("\nMax similarity 2: " + str(max_similarity))

#Points in training sets that are similar to the given vector
si_1 = [tuple(x) for x in similar_inputs_1]
si_2 = [tuple(x) for x in similar_inputs_2]

commonset = list(set(si_1).intersection(set(si_2)))

common_y = []

for i in range(len(commonset)):
    common_y.append(commonset[i][0])


counter_commonset = Counter(common_y)


f.write("\nNo. common: " + str(len(commonset)))
f.write("\nY values in common set\n" + str(counter_commonset))

si1_y = []

for i in range(len(si_1)):
    si1_y.append(si_1[i][0])


counter_si1y = Counter(si1_y)

f.write("\nSimilar in first dataset:" + str(len(set(si_1))))
f.write("\nY values in similar set 1\n" + str(counter_si1y))

si2_y = []

for i in range(len(si_2)):
    si2_y.append(si_2[i][0])

counter_si2y = Counter(si2_y)

f.write("\nSimilar in second dataset:" + str(len(set(si_2))))
f.write("\nY values in similar set 2\n" + str(counter_si2y))

# f1 = open("similar1MLPMLP", "w+")
# f1.write(str(si_1))
# f1.close()
# f2 = open("similar2MLPMLP", "w+")
# f2.write(str(si_2))
# f2.close()

f.close()
