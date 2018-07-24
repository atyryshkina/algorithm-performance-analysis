import pandas as pd
import numpy as np
import sklearn
from sklearn import ensemble
from tabulate import tabulate
import ast

from preprocessing import ChooseFeatureColumns
from preprocessing import MyMapper


class TrainModel(object):

    def __init__(self, filename):

        self.df=pd.read_csv(filename, low_memory=False)
        self.df=self.remove_bad_columns(self.df)
        self.df=self.remove_bad_columns(self.df)

        if len(self.df)<4:
            return

        buckets=[0, self.df["runtime"].max()]

        if self.df.shape[0]*0.01 < 100:
            min_bucket_size=100
        else:
            min_bucket_size=self.df.shape[0]*0.01

        buckets=self.make_buckets(self.df, buckets, min_bucket_size)
        self.df=self.mark_buckets(self.df, buckets)

        print('buckets: %r' % buckets)

        del self.df["runtime"]

        df_features, df_labels = self.df, self.df.pop("result")
        num_instances=len(df_labels)
        
        ################################################
        # prepare the category binarizer,  minmaxscaler defined in preprocessing.py
        chooser = ChooseFeatureColumns()
        scaler = MyMapper()

        self.pipe = sklearn.pipeline.Pipeline([
            ('chooser',chooser),
            ('scaler', scaler),
        ])

        self.clf=sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=12)
        ################################################

        # this is where we will collect the metrics for each of the three trials
        metrics = {
            'accuracy': [],
            'mean_interval': [],
            'median_interval': [],
        }

        kf = sklearn.model_selection.KFold(n_splits=3, shuffle=True)
        kf.get_n_splits(df_features)
        for train_index, test_index in kf.split(df_features):
            tr_features=df_features.iloc[list(train_index)]
            tr_labels=df_labels[train_index]
            ev_features=df_features.iloc[list(test_index)]
            ev_labels = df_labels[test_index]

            tr_features = self.pipe.fit_transform(tr_features)
            ev_features = self.pipe.transform(ev_features)

            metrics = self.analysis(metrics, tr_features, tr_labels, ev_features, ev_labels)

        print(metrics)

        for metric in metrics:
            metrics[metric] = float(np.mean(metrics[metric]))
            

        metrics['filename'] = filename
        metrics['num_instances'] = num_instances  

        print(tabulate(sorted([(k,v) for k,v in metrics.items()])))


    def make_buckets(self, df, buckets, min_bucket_size):
        if not df.shape[0]/2 < min_bucket_size:
            new_buck=df["runtime"].quantile(0.5)
            buckets.append(new_buck)
            buckets=self.make_buckets(df[df["runtime"]>new_buck], buckets, min_bucket_size)
        return sorted(buckets)

    def mark_buckets(self, df, buckets):
        for i, r in df.iterrows():
            for buck in buckets:
                if r["runtime"] < buck:
                    break
            df.at[i, 'result'] = '%r-%r' % (buckets[buckets.index(buck)-1],buck)
        return df
        
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
        
    def analysis(self, metrics, tr_features, tr_labels, ev_features, ev_labels):
        print("fitting")
        self.clf.fit(tr_features,tr_labels)
        print("predicting")

        ev_pred = self.clf.predict(ev_features)
        intervals=self.get_intervals(ev_pred)
        
        metrics['accuracy'].append(sklearn.metrics.accuracy_score(ev_labels, ev_pred))
        metrics['median_interval'].append(np.median(intervals))
        metrics['mean_interval'].append(np.mean(intervals))

        return metrics

    def get_intervals(self, preds):
        ints=[]

        for item in preds:
            sp=item.split("-")
            ints.append(float(sp[1])-float(sp[0]))


        return ints 

   



filenames = ['examples/bwa_mem_0.7.15.1_example.csv', 'examples/stringtie_1.3.3_example.csv']


for i in range(len(filenames))[0:]:
    print("%r/%r %s" % (i, len(filenames), filenames[i]))
    TrainModel(filenames[i])
