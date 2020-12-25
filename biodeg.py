import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
df = pd.read_csv("biodeg.csv",delimiter=";")
orig_columns = df.columns
for i in range(len(df.drop('RB',axis=1).columns)):
    df[str(i)] = scaler.fit_transform(np.reshape(np.array(df[df.columns[i]]),(-1,1)))
df[str(41)] = df['RB'].astype('category').cat.codes

df.drop(orig_columns,axis=1,inplace=True)

x = df.drop('41',axis=1)
y = df['41'].astype('category').cat.codes

sm = SMOTE(random_state=42)
X_res,y_res = sm.fit_resample(x,y)

X_train, X_test, y_train, y_test = train_test_split(X_res.values, y_res.values, test_size = 0.2, random_state = 25, shuffle = True)
X_validation, X_test,y_validation, y_test = train_test_split(X_test,y_test, test_size = 0.5,random_state = 25, shuffle = True)

xgb_model = xgb.XGBClassifier(learning_rate = 0.001, objective = 'binary:logistic')
xgtrain = xgb.DMatrix(x,label=y)
(xgb.cv(params = {'nthread':4,'max_depth':4,'learning_rate':0.1,'reg_alpha':0.003,'reg_lambda':0.003,'objective':'binary:logistic'},dtrain = xgtrain,num_boost_round = 1000,nfold=3,seed=1301,early_stopping_rounds = 20))

clf2 = xgb.XGBClassifier(n_estimators = 81, learning_rate = 0.001, reg_alpha = 0.003, reg_lambda = 0.003, objective = 'binary:logistic')
clf2.fit(X_train,y_train)
pred = clf2.predict(X_test)

import plotly.express as px
importances = clf2.feature_importances_
important_trends = []
for i in range(len(importances)):
    if(importances[i] > 0.1):
        important_trends.append([i,importances[i]])

important_columns = []
important_values = []
for i in range(len(important_trends)):
    important_columns.append(x.columns[important_trends[i][0]])
    important_values.append(important_trends[i][1])
    
    
X_res = np.array(X_res)

preprocessed_df['NssssC: Number of atoms of type ssssC'] = np.array([X_res[i][5] for i in range(len(X_res))])
preprocessed_df['F02[C-N]: Frequency of C - N at topological distance 2'] = np.array([X_res[i][33] for i in range(len(X_res))])
preprocessed_df[' SpMax_B(m): Leading eigenvalue from Burden matrix weighted by mass'] = df['35']

preprocessed_df['ready biodegradable?'] = y_res
preprocessed_df = preprocessed_df.sample(frac = 1) 

examples = len(preprocessed_df)
numTarget = 1
numDependent = len(preprocessed_df.drop('ready biodegradable?',axis=1).columns)

train = (np.append(np.array(preprocessed_df[preprocessed_df['ready biodegradable?']==1][:300]),np.array(preprocessed_df[preprocessed_df['ready biodegradable?']==0][:300]),axis=0))
test = (np.append(np.array(preprocessed_df[preprocessed_df['ready biodegradable?']==1][300:354]),np.array(preprocessed_df[preprocessed_df['ready biodegradable?']==0][300:354]),axis=0))
np.random.shuffle(train)

with open('biodeg.train.txt','w') as file:
    file.write(str(len(train)) + " " + str(numDependent) + " " + str(numTarget)+"\n")
    statements = []
    for i in range(len(train)):
        statement = ""
        for j,val in enumerate(train[i]):
            if(j==len(train[i])-1):
                statement += str(val) + "\n"
            else:
                statement += str(val) + " "
        statements.append(statement)
    for phrase in statements:
        file.write(phrase)
        
        
with open('biodeg.test.txt','w') as file:
    file.write(str(len(test)) + " " + str(numDependent) + " " + str(numTarget)+"\n")
    statements = []
    for i in range(len(test)):
        statement = ""
        for j,val in enumerate(test[i]):
            if(j==len(test[i])-1):
                statement += str(val) + "\n"
            else:
                statement += str(val) + " "
        statements.append(statement)
    for phrase in statements:
        file.write(phrase)
            
            
with open('biodeg.weights.txt','w') as file:
    numInputs = numDependent
    
    numHidden = 20
    numOutputs = numTarget
    file.write(str(numInputs) + " " + str(numHidden) + " " + str(numOutputs) + "\n")
    hiddenLayer = np.random.uniform(size=(numHidden,numInputs+1))
    statements=[]
    for i in range(len(hiddenLayer)):
        statement = ""
        for j,val in enumerate(hiddenLayer[i]):
            if(j==len(hiddenLayer[i])-1):
                statement+='{:.3f}'.format(round(val,3))+"\n"
            else:
                statement+='{:.3f}'.format(round(val,3)) + " "
        statements.append(statement)
   
    outputLayer = np.random.uniform(size = (numOutputs,numHidden+1))
    for i in range(len(outputLayer)):
        statement = ""
        for j,val in enumerate(outputLayer[i]):
            if(j==len(outputLayer[i])-1):
                statement+='{:.3f}'.format(round(val,3))+"\n"
            else:
                statement+='{:.3f}'.format(round(val,3)) + " "
        statements.append(statement)
            
    for phrase in statements:
        file.write(phrase)
