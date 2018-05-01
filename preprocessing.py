import pandas as pd
import numpy as np
from sklearn_pandas import gen_features
import sklearn
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer

class LabelBinarizer2:

    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit X using the LabelBinarizer object
        self.lb.fit(X)
        # Save the classes
        self.classes_ = self.lb.classes_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

    def transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Transform X using the LabelBinarizer object
        Xlb = self.lb.transform(X)
        
        if len(self.classes_) == 2 and len(np.unique(X)) <= 2:
            Xlb = np.hstack((1 - Xlb, Xlb))
        return Xlb

    def inverse_transform(self, Xlb):
        # Convert Xlb to array
        Xlb = np.array(Xlb)
        if len(self.classes_) == 2:
            X = self.lb.inverse_transform(Xlb[:, 0])
        else:
            X = self.lb.inverse_transform(Xlb)
        return X
    
class ChooseFeatureColumns():
    def __init__(self):
        pass
        
    def fit(self,X,y=None):
        self.num_cols=[] 
        self.cat_cols=[]
#         print("sorting features")
        for col in X:
            if X[col].dtype == float or X[col].dtype == int:
                self.num_cols.append(col)
            else:
                self.cat_cols.append(col)
#         print("features sorted")
        return self
    
    def transform(self,X, y=None):
        X = X.copy()
        X = X.reset_index(drop=True)
        newX = pd.DataFrame(index=np.arange(X.shape[0]))
        
        for col in self.num_cols:
            if col in X:
                newX[col] = X[col].astype(float)
                newX[col]=newX[col].fillna(0)
            else:
#                 print("%s not in dataframe - adding empty column" % col)
                newX[col]=np.nan
                newX[col]=newX[col].astype(float)
                newX[col]=newX[col].fillna(0)
                
        for col in self.cat_cols:
            if col in X:
                newX[col] = X[col].astype(str)
            else:
#                 print("%s not in dataframe - adding empty column" % col)
                newX[col]=np.nan
                newX[col]=newX[col].astype(str)
        
        return newX
    
    def fit_transform(self,X, y=None):
        self.fit(X)
        return self.transform(X)

class MyMapper():
    def __init__(self):
        pass
        
    def fit(self,X,y=None):
        self.ncols = []
        self.scols = []
#         print("mapping features")
        for col in X:
            if X[col].dtype == float:
                # print("numerical col: %s" % col)
                self.ncols.append([col])
            else:
                # print("categorical col: %s" % col)
                self.scols.append([col])
        nfeats = gen_features(
              columns=self.ncols,
              classes=[{'class':sklearn.preprocessing.MinMaxScaler,}]  
        )
        sfeats = gen_features(
              columns=self.scols,
              classes=[{'class':LabelBinarizer2}]  
        )
        self.mapper = DataFrameMapper(nfeats+sfeats,df_out=True)
        self.mapper.fit(X)
#         print("features mapped")
        return self
    
    def transform(self,X, y=None):
        X = X.copy()
        X = self.mapper.transform(X)
        return X
    
    def fit_transform(self,X, y=None):
        self.fit(X)
        return self.transform(X)