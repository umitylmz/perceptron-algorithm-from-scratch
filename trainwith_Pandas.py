import numpy
import pandas
from sklearn.utils import shuffle
import time


#lets load data set
#data format x1 x2 and y
train_data=pandas.read_csv("ML_HW2/train_set_d_1.txt",header=None, delimiter=r"\s+")
test_data=pandas.read_csv("ML_HW2/test_set_d_1.txt",header=None, delimiter=r"\s+")


rows=train_data.shape[0] 
cols=train_data.shape[1]

print(rows)
print(cols)
#Lets check number of 

print(train_data)
X=train_data.iloc[:,0:cols-1]  # input featurs x1 and x2
Y=train_data.iloc[:,cols-1] # label 1 or -1
Xtest=test_data.iloc[:,0:cols-1] 
Ytest=test_data.iloc[:,cols-1]

no_of_weights=X.shape[1]
weight=numpy.zeros(no_of_weights+1)
learning_rate=0.01
start_time = time.time()

for epoch in range(0,100):
      
    sum_error=0
    counter=0

    for i in range(0,X.shape[0]):
        activation=weight[0]+sum(weight[1:]*(X.iloc[i,:]))

        if activation>0:
            prediction=1
        else:
            prediction=-1 

        expected=Y[i]
        error=expected - prediction

        if(error==0):
            counter+=1

        sum_error=sum_error+(error**2)

        weight[0]=weight[0]+(learning_rate*error)
        for j in range(1,3):
            weight[j]=weight[j]+(learning_rate*error*X.iloc[i,j-1])

    print("------->")
    print(counter)      
    
    if(sum_error==0):
        print("------")
        print(epoch)
        print(weight)
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        break
    print(sum_error)

print(weight)

trues=0
for i in range(0,Ytest.shape[0]):
    activation=weight[0]+sum(weight[1:]*(Xtest.iloc[i,:]))

    if activation>0:
        prediction=1
    else:
        prediction=-1 

    expected=Ytest[i]
    if(expected==prediction):
        trues+=1

print(trues)

        
        


