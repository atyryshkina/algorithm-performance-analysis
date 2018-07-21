import pandas as pd
import numpy as np
import sklearn, skgarden
import scipy
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

        self.df=self.remove_bad_columns(self.df)
        self.df=self.remove_bad_columns(self.df)

        df_features, df_labels = self.df, self.df.pop("runtime")

        num_instances=len(df_labels)
        runtime_avg=float(df_labels.mean())
        runtime_std=float(df_labels.std())

        ################################################
        # prepare the category binarizer,  minmaxscaler defined in preprocessing.py
        chooser = ChooseFeatureColumns()
        scaler = MyMapper()

        self.pipe = sklearn.pipeline.Pipeline([
            ('chooser',chooser),
            ('scaler', scaler),
        ])

        self.regr=skgarden.RandomForestRegressor(n_estimators=100, max_depth=12)
        ################################################

        # this is where we will collect the metrics for each of the three trials
        metrics = {
            'r2score': [],
            'pearson': [],
            'accuracy1': [],
            'accuracy2': [],
            'accuracy3': [],
            'median_interval1': [],
            'median_interval2': [],
            'median_interval3': [],
            'mean_interval1': [],
            'mean_interval2': [],
            'mean_interval3': [],
        }

        # this will help us split everything up
        kf = sklearn.model_selection.KFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(df_features):
            tr_features=df_features.iloc[list(train_index)]
            tr_labels=df_labels[train_index]
            ev_features=df_features.iloc[list(test_index)]
            ev_labels=df_labels[test_index]

            tr_features = self.pipe.fit_transform(tr_features)
            ev_features = self.pipe.transform(ev_features)

            metrics = self.analysis(metrics, tr_features, tr_labels, ev_features, ev_labels)

        for metric in metrics:
            metrics[metric] = float(np.mean(metrics[metric]))
         
        metrics['toolid'] = filename
        metrics['num_instances'] = num_instances   

        print(tabulate(sorted([(k,v) for k,v in metrics.items()])))
        
        # save metrics in a way you prefer or return tbe values
        self.save_metrics(metrics)
    


    def save_metrics(self, metrics):
        print('define TrainModel.save_metrics() to save the metrics')

        
    def analysis(self, metrics, tr_features, tr_labels, ev_features, ev_labels):
        self.regr.fit(tr_features,tr_labels)

        ev_pred = self.regr.predict(ev_features,  return_std=True)

        cq=pd.DataFrame()
        cq["labels"] = (ev_labels)
        cq["pred"] = (ev_pred[0])
        cq["stdd"]=ev_pred[1]
        cq["stdu"]=ev_pred[1]

        metrics['median_interval1'].append((cq["stdu"]*2).quantile(q=0.5))
        metrics['mean_interval1'].append((cq["stdu"]*2).mean())
        metrics['accuracy1'].append(self.get_accuracy(cq, std=1))

        metrics['median_interval2'].append((cq["stdu"]*2*2).quantile(q=0.5))
        metrics['mean_interval2'].append((cq["stdu"]*2*2).mean())
        metrics['accuracy2'].append(self.get_accuracy(cq, std=2))

        metrics['median_interval3'].append((cq["stdu"]*3*2).quantile(q=0.5))
        metrics['mean_interval3'].append((cq["stdu"]*3*2).mean())
        metrics['accuracy3'].append(self.get_accuracy(cq, std=3))

        metrics['r2score'].append(float(sklearn.metrics.r2_score(cq["labels"], cq["pred"])))
        metrics['pearson'].append(float(scipy.stats.pearsonr(cq["labels"], cq["pred"])[0]))

        return metrics
        
    def get_accuracy(self, cq, std=1):
        correct = 0.
        for k, r in cq.iterrows():
            if r["labels"] < (r["pred"]+r["stdu"]*std) and r["labels"] > (r["pred"]-r["stdd"]*std):
                correct+=1.

        return (correct/cq.shape[0])

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


        # remove other commond identifier columns
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


filenames = ['examples/bwa_mem_0.7.15.1_example.csv', 'examples/stringtie_1.3.3_example.csv']


for i in range(len(filenames)):
    print("%d/%d %s"%(i,(len(filenames)), filenames[i]))
    TrainModel(filenames[i])

