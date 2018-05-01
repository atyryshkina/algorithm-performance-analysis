import argparse
import scipy
from sklearn.externals import joblib
import math

import pandas as pd

from sklearn import preprocessing
from sklearn_pandas import gen_features
from sklearn_pandas import DataFrameMapper

from sklearn import ensemble
import sklearn
import numpy as np
from matplotlib import pyplot as plt
from preprocessing import ChooseFeatureColumns
from preprocessing import MyMapper
import ast
import os
import skgarden

import mysql.connector
from tabulate import tabulate

cnx = mysql.connector.connect(user='root', password='root',
                              host='localhost',
                              database='datadump')
cursor = cnx.cursor()

class TrainModel(object):

    def __init__(self, filename):

        self.df=pd.read_csv("csv_dump/%s.csv"%filename)
        self.df=self.remove_bad_columns(self.df)
        self.df=self.remove_bad_columns(self.df)

        df_features, df_labels = self.df, self.df.pop("runtime")
        num_instances=len(df_labels)
        runtime_avg=float(df_labels.mean())
        runtime_std=float(df_labels.std())

        df_labels = np.log1p(df_labels)
        original_num_of_cols=df_features.shape[1]
        ################################################
        
        if math.isnan(runtime_std):
            runtime_std=0.
        ################################################
        ################################################
        # prepare the data for the RandomForestRegressor
        print("setting up...")
        chooser = ChooseFeatureColumns()
        scaler = MyMapper()
        # regr = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=12)
        self.regr=skgarden.RandomForestRegressor(n_estimators=100, max_depth=12)

        self.pipe = sklearn.pipeline.Pipeline([
            ('chooser',chooser),
            ('scaler', scaler),
        ])
        ################################################
        
        test_size = 0.2
        test_start=len(df_labels)-int(len(df_labels)*test_size)
        print(test_start, len(df_labels))

 
        split_randomly=False
        time_split=num_instances>10
        if split_randomly and time_split:
            tr_features, ev_features, tr_labels, ev_labels = sklearn.model_selection.train_test_split(df_features, df_labels, test_size=test_size)
            print("splitting randomly")
        elif time_split:
            tr_features, tr_labels, ev_features, ev_labels = df_features[:test_start], df_labels[:test_start], df_features[test_start:], df_labels[test_start:]
            print("splitting non-randomly")
        if time_split and ((len(list(self.pipe.fit_transform(tr_features)))) != (len(list(self.pipe.transform(ev_features))))):
            print("!! pipe transformation broken")
            time_split=False
        ################################################

        df_features=self.pipe.fit_transform(df_features)
        print("%r ----> %r" %(original_num_of_cols,df_features.shape[1]))
        if float(df_features.shape[1])/float(original_num_of_cols)>10:
            print("GREATER THAN 10")

        if time_split:
            tr_features=self.pipe.fit_transform(tr_features)
            ev_features=self.pipe.transform(ev_features)
             


        r2score, pearson, accuracy = self.analysis(filename, df_features, df_labels, df_features, df_labels, timesplit=False)
        if time_split:
            r2score2, pearson2, accuracy2 = self.analysis(filename, tr_features, tr_labels, ev_features, ev_labels, timesplit=True)
            
            metrics={"toolid": filename, "num_instances":num_instances,
                    "r2": r2score, "pearson":pearson, "accuracy": accuracy,
                    "runtime_avg": runtime_avg, "runtime_std":runtime_std,
                    "r2_time":r2score2, "pearson_time": pearson2, "accuracy_time":accuracy2}
            command = '''INSERT INTO tool_pred_metrics (toolid, num_instances, r2, pearson, accuracy, runtime_avg, 
                            runtime_std, r2_time_split, pearson_time_split, accuracy_time_split) 
                            
                            VALUES (%(toolid)s, %(num_instances)s, %(r2)s, %(pearson)s, %(accuracy)s, %(runtime_avg)s, 
                            %(runtime_std)s, %(r2_time)s,%(pearson_time)s,%(accuracy_time)s)
                            
                            ON DUPLICATE KEY UPDATE num_instances=%(num_instances)s, r2=%(r2)s, pearson=%(pearson)s, accuracy=%(accuracy)s, 
                            runtime_avg=%(runtime_avg)s, runtime_std=%(runtime_std)s,
                            r2_time_split=%(r2_time)s,pearson_time_split=%(pearson_time)s,
                            accuracy_time_split=%(accuracy_time)s;
                            '''
        else:
            metrics={"toolid": filename, "num_instances":num_instances,
                    "r2": r2score, "pearson":pearson, "accuracy": accuracy,
                    "runtime_avg": runtime_avg, "runtime_std":runtime_std}
            command = '''INSERT INTO tool_pred_metrics (toolid, num_instances, r2, pearson, accuracy, runtime_avg, 
                            runtime_std) 
                            
                            VALUES (%(toolid)s, %(num_instances)s, %(r2)s, %(pearson)s, %(accuracy)s, %(runtime_avg)s, %(runtime_std)s)
                            
                            ON DUPLICATE KEY UPDATE num_instances=%(num_instances)s, r2=%(r2)s, pearson=%(pearson)s, accuracy=%(accuracy)s, 
                            runtime_avg=%(runtime_avg)s, runtime_std=%(runtime_std)s;
                            '''
        print(tabulate(sorted([(k,v) for k,v in metrics.items()])))
        cursor.execute(command, metrics)
        cnx.commit()
        
        print("done")
        
    def remove_bad_columns(self, df):
        parameters = [i for i in list(df) if (i.startswith("parameters.") and not i.startswith("parameters.__job_r"))]
        filetypes=[i for i in list(df) if (i.endswith("_filetype") and not i.startswith("parameters.__job_r"))]
        files=[i[:-9] for i in filetypes]
        bad_parameters=["parameters.__workflow_invocation_uuid__","parameters.chromInfo"]
        for parameter in parameters:
            series=df[parameter].dropna()
            if all(type(item)==str and item.startswith('"') for item in series): 
                try:
                    df[parameter]=df[parameter].str[1:-1].astype(float)
                except:
                    pass
            if len(df[parameter].unique())>=0.5*df.shape[0]:
                bad_parameters.append(parameter)
            if df[parameter].dtype == object and len(df[parameter].unique())>=10*df.shape[1]:
                bad_parameters.append(parameter)
            if all(type(item)==str and item.startswith("[") and item.endswith("]") for item in series):#  and item.startswith("[{'src'")
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
        hardware=['destination_id',
         'galaxy_slots',
         'handler',
         'job_runner_name',
         'memtotal',
         'processor_count',
         'runtime']
        keep=parameters+filetypes+files+hardware
        columns=list(df)
        for column in columns:
            if not column in keep:
                del df[column]
        return df
        
    def analysis(self, filename, tr_features, tr_labels, ev_features, ev_labels, timesplit=False):
        self.regr.fit(tr_features,tr_labels)
        ev_pred = self.regr.predict(ev_features,  return_std=True)

        # unlog the runtimes (they were previously log transformed in the function clean_data()) 
        cq=pd.DataFrame()
        cq["labels"] = np.expm1(ev_labels)
        cq["pred"] = np.expm1(ev_pred[0])
        cq["std"]=np.expm1(ev_pred[0])-np.expm1(ev_pred[0]-ev_pred[1])
        
        accuracy=self.get_accuracy(cq)
        
        self.plot(cq, filename, timesplit=timesplit)
        r2score=float(sklearn.metrics.r2_score(cq["labels"], cq["pred"]))
        pearson=float(scipy.stats.pearsonr(cq["labels"], cq["pred"])[0])
        if math.isnan(pearson):
            pearson=0.
        return r2score, pearson, accuracy
        
    def get_accuracy(self, cq):
        correct = 0.
        i=0
        for k, r in cq.iterrows():
            if r["labels"] < (r["pred"]+r["std"]) and r["labels"] > (r["pred"]-r["std"]):
                correct+=1.
            i+=1
        

        return (correct/cq.shape[0])

    def plot(self, cq, filename, timesplit=False):
        plot=True
        plot_w_error=True
        if plot:
            plt.figure(figsize=(10,10))
            plt.scatter(cq["labels"],cq["pred"])
            plt.xlabel("Real Runtime")
            plt.ylabel("Predicted Runtime")
            plt.title("Mean predictions")
            if timesplit:
                plt.savefig("plots_dump/timesplit/%s.png"%filename)
                print("saved a plot to plots_dump/timesplit/%s.png"%filename)
            else:
                plt.savefig("plots_dump/%s.png"%filename)
                print("saved a plot to plots_dump/%s.png"%filename)
            plt.close()
        if plot_w_error:
            plt.figure(figsize=(10,10))
            plt.scatter(cq["labels"],cq["pred"])
            plt.errorbar(cq["labels"],cq["pred"], yerr=[cq["std"],cq["std"]], fmt='o')
            plt.xlabel("Real Runtime")
            plt.ylabel("Predicted Runtime")
            plt.title("Mean predictions")
            if timesplit:
                plt.savefig("plots_w_error_dump/timesplit/%s.png"%filename)
                print("saved a plot to plots_w_error_dump/timesplit/%s.png"%filename)
            else:
                plt.savefig("plots_w_error_dump/%s.png"%filename)
                print("saved a plot to plots_w_error_dump/%s.png"%filename)
            
            plt.close()



def getfiles(dirpath):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)), reverse=True)
    return a

finished=getfiles("plots_dump")
filenames=getfiles("csv_dump")
filenames.remove(".DS_Store")

only_new=False

print("length", len(filenames))

if only_new:
    for file in finished:
        # print(file[:-4])
        try:
            filenames.remove(file[:-4]+".csv")
        except:
            pass
print("length", len(filenames))


for i in range(len(filenames)):
    if i % 100 == 0:
        print("%d/%d"%(i,(len(filenames))))
    print(filenames[i][:-4])
    TrainModel(filenames[i][:-4])

cursor.close()
cnx.close()
