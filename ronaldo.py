import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import scipy 
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

datasets = pd.read_csv('data.csv')

output=pd.DataFrame(datasets)
cols = [17,10]
output = output[output.columns[cols]]

testdata=pd.DataFrame(datasets)
cols = [2,3,5,9,10]
testdata = testdata[testdata.columns[cols]]

df = pd.DataFrame(datasets)
cols = [2,3,5,9,10]
df = df[df.columns[cols]]


###########################################################################################
#Train DATA 
k=0
x=[]
for i in df["is_goal"]:
    if math.isnan(i):
        x.append(k)
        #print(i)
    k+=1
df=(df.drop(x))

k=0
x=[]
for i in df["distance_of_shot"]:
    
    if math.isnan(i):
        x.append(df.index[k])
        #print(i)
    k+=1
df=(df.drop(x))

k=0
x=[]
for i in df["power_of_shot"]:
    
    if math.isnan(i):
        x.append(df.index[k])
        #print(i)
    k+=1
df=(df.drop(x))

k=0
x=[]
for i in df["location_x"]:
    if math.isnan(i):
        x.append(df.index[k])
        #print(i)
    k+=1
df=(df.drop(x))

k=0
x=[]
for i in df["location_y"]:
    if math.isnan(i):
        x.append(df.index[k])
        #print(i)
    k+=1
df=(df.drop(x))
#print(df)

X = df.iloc[:, :-1].values
Y = df.iloc[:, 4].values

#print(X,Y)
###################################################################################################
#Test DATA 
k=0
x=[]
for i in testdata["is_goal"]:
    if math.isnan(i)==False:
        x.append(k)
        #print(i)
    k+=1
testdata=(testdata.drop(x))

values = {'distance_of_shot': 0.0, 'power_of_shot': 0.0, 'location_x': 0.0, 'location_y': 0.0}
testdata.fillna(value=values,inplace=True)

#print(testdata)
X_TEST=testdata.iloc[:,:-1].values
Y_TEST=testdata.iloc[:,4].values
#print(X_TEST,Y_TEST)

k=0
x=[]
for i in output["is_goal"]:
    if math.isnan(i)==False:
        x.append(k)
        #print(i)
    k+=1
output=(output.drop(x))
values ={"shot_id_number":0.0}
output.fillna(value=values,inplace=True)
#print(output)

###################################################################################################

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, random_state=0)

Lr=LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
LRRR=LinearRegression()
SVM = svm.LinearSVC()
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)



Lr.fit(X,Y)
LRRR.fit(X,Y)
SVM.fit(X, Y)
RF.fit(X, Y)
NN.fit(X, Y)



print(Lr.score(X,Y))
print(LRRR.score(X,Y))
print(SVM.score(X,Y))
print(RF.score(X,Y))
print(NN.score(X,Y))

l=[[10,12,3,32]]
print(Lr.predict(X_TEST[0:10]))
print(LRRR.predict(X_TEST[0:10]))
print(SVM.predict(X_TEST[0:10]))
print(RF.predict(X_TEST[0:10]))
print(NN.predict(X_TEST[0:10]))

Y_TEST=Lr.predict(X_TEST)
print(Y_TEST)

output["is_goal"]= Y_TEST
output["shot_id_number"]=output.index+1
print(output)
output.to_csv('file1.csv',index = False) 