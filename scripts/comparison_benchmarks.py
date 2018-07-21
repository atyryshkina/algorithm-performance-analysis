import pandas as pd
import numpy as np
import sklearn
from sklearn import ensemble, linear_model, gaussian_process, neural_network
from tabulate import tabulate
import ast

from preprocessing import ChooseFeatureColumns
from preprocessing import MyMapper

class TrainModel(object):

    def __init__(self, filename):

        # load the csv file into a dataframe
        self.df=pd.read_csv(filename, low_memory=False)

        if len(self.df)<4:
            print("less than 4 data points, skipping")
            return

        # feature selection
        self.df=self.remove_bad_columns(self.df)
        self.df=self.remove_bad_columns(self.df)

        # split to independent and dependent variables
        df_features, df_labels = self.df, self.df.pop("runtime")

        # collect some statistics
        num_instances=len(df_labels)
        runtime_avg=float(df_labels.mean())
        runtime_std=float(df_labels.std())

        # normalize runtimes
        df_labels = np.log1p(df_labels)

        
        ################################################
        # prepare the category binarizer,  minmaxscaler defined in preprocessing.py
        chooser = ChooseFeatureColumns()
        scaler = MyMapper()

        self.pipe = sklearn.pipeline.Pipeline([
            ('chooser',chooser),
            ('scaler', scaler),
        ])
        ################################################

        # select which regressors to use here
        regressors={
            "RandomForestRegressor":sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=12),
            "ExtraTreesRegressor":sklearn.ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=12),
            "GradientBoostingRegressor":sklearn.ensemble.GradientBoostingRegressor(),
            # "LinearRegression":sklearn.linear_model.LinearRegression(),
            # "Lasso":sklearn.linear_model.Lasso(),
            # "Ridge":sklearn.linear_model.Ridge(),
            # "SVR":sklearn.svm.SVR(),
            # "SGDRegressor":sklearn.linear_model.SGDRegressor(),
            # "KNeighborsRegressor":sklearn.neighbors.KNeighborsRegressor(),
            # "GaussianProcessRegressor":sklearn.gaussian_process.GaussianProcessRegressor(),
            # "MLPRegressor":sklearn.neural_network.MLPRegressor(hidden_layer_sizes=[100,10])
        }

        metrics={}
        train_indeces=[]
        test_indeces=[]
        
        # get split indeced
        kf = sklearn.model_selection.KFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(df_features):
            train_indeces.append(train_index)
            test_indeces.append(test_index)

        # for each regressor, do 3-fold validation an record the r-2 score and time
        for regressor_name in regressors:

            print(regressor_name)
            regr=regressors[regressor_name]
            r2s=[]

            for i in range(len(train_indeces)): 

                tr_features=df_features.iloc[list(train_indeces[i])]
                tr_labels=df_labels[train_indeces[i]]
                ev_features=df_features.iloc[list(test_indeces[i])]
                ev_labels=df_labels[test_indeces[i]]
                
                tr_features = self.pipe.fit_transform(tr_features)
                ev_features = self.pipe.transform(ev_features)

                regr.fit(tr_features, tr_labels)
                ev_pred=regr.predict(ev_features)
                r2=sklearn.metrics.r2_score(ev_labels,ev_pred)
            
                r2s.append(r2)

            metrics[regressor_name]= float(np.mean(r2s))


        metrics["toolid"]=filename
        metrics["num_instances"]=num_instances

        print(tabulate(sorted([(k,v) for k,v in metrics.items()])))
        
        # save metrics in a way you prefer or return tbe values
        self.save_metrics(metrics)
    


    def save_metrics(self, metrics):
        print('define TrainModel.save_metrics() to save the metrics')


        
    def remove_bad_columns(self, df):

        # get a list of all user selected parameters
        parameters = [i for i in list(df) if (i.startswith("parameters."))]

        # get names of files
        filetypes=[i for i in list(df) if ((i.endswith("_filetype")) and not i == 'chromInfo_filetype')]
        files=[i[:-9] for i in filetypes]

        # begin bad parameter list
        bad_starts=["parameters.__workflow_invocation_uuid__","parameters.chromInfo", 'parameters.__job_resource',
                        'parameters.reference_source', 'parameters.reference_genome', 'parameters.rg', 
                        'parameters.readGroup', 'parameters.refGenomeSource', 'parameters.genomeSource']

        for p in bad_starts:
            parameters = [i for i in parameters if not i.startswith(p)]

        bad_ends = ['id', 'identifier', '__identifier__', 'indeces']

        for p in bad_ends:
            parameters = [i for i in parameters if not i.endswith(p)]

        bad_parameters = []

        for parameter in parameters:
            series=df[parameter].dropna()
        
            # trim string of ""   This is necessary to check if the parameter is full of list or dict objects
            if df[parameter].dtype==object and all(type(item)==str and item.startswith('"') and item.endswith('"') for item in series): 
                try:
                    df[parameter]=df[parameter].str[1:-1].astype(float)
                except:
                    df[parameter]=df[parameter].str[1:-1]
            
            # if more than half of the rows have a unique value, remove the categorical feature
            if df[parameter].dtype==object and len(df[parameter].unique())>=0.5*df.shape[0]:
                bad_parameters.append(parameter)

            # if number empty is greater than half, remove
            if df[parameter].isnull().sum()>=0.75*df.shape[0]:
                bad_parameters.append(parameter)

            # if the number of categories is greater than 10 remove
            if df[parameter].dtype == object and len(df[parameter].unique())>=100:
                bad_parameters.append(parameter)

            # if the feature is a list remove
            if all(type(item)==str and item.startswith("[") and item.endswith("]") for item in series):#  and item.startswith("[{'src'")
                if all(type(ast.literal_eval(item))==list for item in series):
                    bad_parameters.append(parameter)

            # if the feature is a dict remove
            if all(type(item)==str and item.startswith("{") and item.endswith("}") for item in series):#  and item.startswith("[{'src'")
                if all(type(ast.literal_eval(item))==list for item in series):
                    bad_parameters.append(parameter)

        for file in files:
            bad_parameters.append("parameters."+file)
            bad_parameters.append("parameters."+file+".values")
            bad_parameters.append("parameters."+file+"|__identifier__")

        for param in set(bad_parameters):
            try:
                parameters.remove(param)
            except:
                pass
        
        hardware=['runtime']
        
        keep=parameters+filetypes+files+hardware
        
        columns=list(df)
        for column in columns:
            if not column in keep:
                del df[column]
                
        return df
        

filenames = ['../examples/bwa_mem_0.7.15.1_example.csv', '../examples/stringtie_1.3.3_example.csv']


for i in range(len(filenames)):
    print("%d/%d %s"%(i,(len(filenames)), filenames[i]))
    TrainModel(filenames[i])

