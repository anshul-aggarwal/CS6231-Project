import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy

#Read and Load the Datasets
d1 = pd.read_csv("d1.csv")
d2 = pd.read_csv("d2.csv")
common = pd.read_csv("common.csv")

#Separate into Attributes and Target Variables
y_d1 = d1.iloc[:,0]
x_d1 = d1.iloc[:,1:]

y_d2 = d2.iloc[:,0]
x_d2 = d2.iloc[:,1:]

y_common = common.iloc[:,0]
x_common = common.iloc[:,1:]

#Define the ML Models
m1 = LogisticRegression(multi_class='multinomial',solver='saga',n_jobs=-1)
m2 = LogisticRegression(multi_class='multinomial',solver='saga',n_jobs=-1)

#Train the models
m1.fit(x_d1,y_d1)
m2.fit(x_d2,y_d2)

#Generate Random Point
start_p = np.random.choice([0, 1], size=(len(x_d1.columns),))

#Predict Probability Scores
m1_prob = m1.predict_proba([start_p]).flatten()
m2_prob = m2.predict_proba([start_p]).flatten()

#Get KL Divergence
#########P.S - KL Divergence is not symmteric ( which would be P and Q)
divergence = entropy(m1_prob,m2_prob)
