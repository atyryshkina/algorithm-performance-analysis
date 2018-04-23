import pandas as pd
import numpy as np
from sklearn_pandas import gen_features
import sklearn
from sklearn_pandas import DataFrameMapper

class ChooseFeatureColumns():
    def __init__(self):
        self.num_cols=[] 
        self.cat_cols=[]
        
    def fit(self,X,y=None):
        print("sorting features")
        for col in X:
            if X[col].dtype == float:
                self.num_cols.append(col)
            else:
                self.cat_cols.append(col)
        print("features sorted")
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
                print("%s not in dataframe - adding empty column" % col)
                newX[col]=np.nan
                newX[col]=newX[col].astype(float)
                newX[col]=newX[col].fillna(0)
                
        for col in self.cat_cols:
            if col in X:
                newX[col] = X[col].astype(str)
            else:
                print("%s not in dataframe - adding empty column" % col)
                newX[col]=np.nan
                newX[col]=newX[col].astype(str)
        
        return newX
    
    def fit_transform(self,X, y=None):
        self.fit(X)
        return self.transform(X)

class MyMapper():
    def __init__(self):
        self.ncols = []
        self.scols = []
        
    def fit(self,X,y=None):
        print("mapping features")
        for col in X:
            if X[col].dtype == float:
                print("numerical col: %s" % col)
                self.ncols.append([col])
            else:
                print("categorical col: %s" % col)
                self.scols.append([col])
        nfeats = gen_features(
              columns=self.ncols,
              classes=[{'class':sklearn.preprocessing.MinMaxScaler,}]  
        )
        sfeats = gen_features(
              columns=self.scols,
              classes=[{'class':sklearn.preprocessing.LabelBinarizer}]  
        )
        self.mapper = DataFrameMapper(nfeats+sfeats,df_out=True)
        self.mapper.fit(X)
        print("features mapped")
        return self
    
    def transform(self,X, y=None):
        X = X.copy()
        X = self.mapper.transform(X)
        return X
    
    def fit_transform(self,X, y=None):
        self.fit(X)
        return self.transform(X)