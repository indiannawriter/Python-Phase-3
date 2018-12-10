
# coding: utf-8

# In[ ]:


#Question 1:
import numpy as np
import pandas as pd
import math
import random


#Dataset with two skin segments (1, 2 based on Red/Green/Blue value features) 
#from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
breast_cancer = pd.read_csv("Breast-Cancer-Wisconsin.csv")
    
    
    
class sklearn.cluster.KMeans(n_clusters=20, init=’k-means++’, n_init=500, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)


for iteration in range(1, k+1):
    print("Computing fold where test chunk is:", k-iteration)
    
    #Ascribing the test_set and training_set depending on the fold
    test_set = breast_cancer.iloc[(k-iteration)*chunk_size:(k-iteration+1)*chunk_size, ]
    training_set_left = breast_cancer.iloc[:(k-iteration)*chunk_size, ]
    training_set_right = breast_cancer.iloc[(k-iteration+1)*chunk_size:, ]
    training_set = training_set_left.append(training_set_right)
    
    #We train on the training-data here, very simple training classifier/model where we just tally up how many 
    #of a certain class we see and compute the probability of the given class for the whole training data
    print("Training...")
    ones = sum((training_set[["Class"]] == 1).values.T.tolist()[0])
    twos = sum((training_set[["Class"]] == 2).values.T.tolist()[0])

    total = ones + twos
    ones_prob = ones/total
    twos_prob = twos/total
    
    print("Finished Training, now Testing...")
    # We trained out simple model, now we test on our test data and see how well we do
    true_values = test_set[["Class"]]
    predicted_values = []
    for sample in test_set.iterrows():
        #Random number between 0 and 1
        class_generation = random.random()
        #Here we pick the class that has the closest probability (from training) based on the random draw
        choose_one = abs(class_generation - ones_prob)
        choose_two = abs(class_generation - twos_prob)
        if choose_one < choose_two:
            predicted_values.append(1)
        else:
            predicted_values.append(2)

    comparison_list = (true_values == predicted_values).values.T.tolist()
    accuracy = sum(comparison_list[0])/len(predicted_values)
    print("Fold Accuracy:", accuracy*100, "%")
    accuracy_accumulation.append(accuracy)
    
#Printing accuracy for each iteration
print(accuracy_accumulation)
#Printing averaged accuracy
print("Averaged Accuracy:", (sum(accuracy_accumulation)/len(accuracy_accumulation))*100, "%")
#Printing averaged error
print("Averaged Error:", (1-sum(accuracy_accumulation)/len(accuracy_accumulation))*100, "%")


# In[ ]:


#Question 2:
import numpy as np
import pandas as pd
import math
import random


#Dataset with two skin segments (1, 2 based on Red/Green/Blue value features) 
#from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
breast_cancer = pd.read_csv("Breast-Cancer-Wisconsin.csv")
    
    
    
class sklearn.cluster.KMeans(n_clusters=2, init=’k-means++’, n_init=500, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)


for iteration in range(1, k+1):
    print("Computing fold where test chunk is:", k-iteration)
    
    #Ascribing the test_set and training_set depending on the fold
    test_set = breast_cancer.iloc[(k-iteration)*chunk_size:(k-iteration+1)*chunk_size, ]
    training_set_left = breast_cancer.iloc[:(k-iteration)*chunk_size, ]
    training_set_right = breast_cancer.iloc[(k-iteration+1)*chunk_size:, ]
    training_set = training_set_left.append(training_set_right)
    
    #We train on the training-data here, very simple training classifier/model where we just tally up how many 
    #of a certain class we see and compute the probability of the given class for the whole training data
    print("Training...")
    ones = sum((training_set[["Class"]] == 1).values.T.tolist()[0])
    twos = sum((training_set[["Class"]] == 2).values.T.tolist()[0])

    total = ones + twos
    ones_prob = ones/total
    twos_prob = twos/total
    
    print("Finished Training, now Testing...")
    # We trained out simple model, now we test on our test data and see how well we do
    true_values = test_set[["Class"]]
    predicted_values = []
    for sample in test_set.iterrows():
        #Random number between 0 and 1
        class_generation = random.random()
        #Here we pick the class that has the closest probability (from training) based on the random draw
        choose_one = abs(class_generation - ones_prob)
        choose_two = abs(class_generation - twos_prob)
        if choose_one < choose_two:
            predicted_values.append(1)
        else:
            predicted_values.append(2)

    comparison_list = (true_values == predicted_values).values.T.tolist()
    accuracy = sum(comparison_list[0])/len(predicted_values)
    print("Fold Accuracy:", accuracy*100, "%")
    accuracy_accumulation.append(accuracy)
    
#Printing accuracy for each iteration
print(accuracy_accumulation)
#Printing averaged accuracy
print("Averaged Accuracy:", (sum(accuracy_accumulation)/len(accuracy_accumulation))*100, "%")
#Printing averaged error
print("Averaged Error:", (1-sum(accuracy_accumulation)/len(accuracy_accumulation))*100, "%")

#Question 3:
predicted_clusters=variable_labels


# In[ ]:


#Question 4:
#Importing the pandas package
import pandas as pd

#Creating the data and assigning the data to the dataframe variable df
data = [[2,0],[2,1],[4,1],[2,0],[4,0],[4,0],[2,1],[4,1],[4,2]]
df = pd.DataFrame(data,columns=['CLASS','newdata'])

#Creating new columns named mylabels and copying the values from the newdata column
df['mylabels'] = df['newdata']

#Printing the first 3 records of the df data
print(df.head(n=3))

#Replacing the values zeros with the value of two and replacing in the mylabels column
df['mylabels'] = df['mylabels'].replace(0, 2)

#Replacing the the one values with the value of 4 in the mylabels column
df['mylabels'] = df['mylabels'].replace(1, 4)

