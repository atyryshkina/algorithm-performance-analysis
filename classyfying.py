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

        try:
            self.df=pd.read_csv("data/csv_dump/%s.csv"%filename)
        except FileNotFoundError as e:
            print("dont have this file yet")
            return
        df2=pd.read_csv("failed_jobs/walltime_csv/%s.csv"%filename)
        # df3=pd.read_csv("failed_jobs/memory_csv/%s.csv"%filename)
        self.df["result"], df2["result"]="ok", "walltime error"

        self.df.index=self.df["id"]
        df2.index=df2["id"]
        self.df=pd.concat([self.df,df2])
        self.df=self.df.sort_values("id")
        del self.df["id"]
        self.df=self.df.reset_index()
        del self.df["id"]

        self.df=self.remove_bad_columns(self.df)
        self.df=self.remove_bad_columns(self.df)

        num_errors=df2.shape[0]
        df_features, df_labels = self.df, self.df.pop("result")
        num_instances=len(df_labels)

        original_num_of_cols=df_features.shape[1]
        ################################################
        ################################################
        # prepare the data for the RandomForestRegressor
        print("setting up...")
        chooser = ChooseFeatureColumns()
        scaler = MyMapper()
        # regr = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=12)
        self.regr=sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=12)

        self.pipe = sklearn.pipeline.Pipeline([
            ('chooser',chooser),
            ('scaler', scaler),
        ])
        # ################################################
        test_size = 0.2
        test_start=len(df_labels)-int(len(df_labels)*test_size)
 
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
        # ################################################

        df_features=self.pipe.fit_transform(df_features)
        print("%r ----> %r" %(original_num_of_cols,df_features.shape[1]))
        if float(df_features.shape[1])/float(original_num_of_cols)>10:
            print("GREATER THAN 10")

        if time_split:
            tr_features=self.pipe.fit_transform(tr_features)
            ev_features=self.pipe.transform(ev_features)
             
        self.table="walltime_classify"

        accuracy, false_positives, false_negatives=self.analysis(filename, df_features, df_labels, df_features, df_labels, timesplit=False)
        print("whole accuracy: %r" % accuracy)
        if time_split:
            accuracy2, false_positives_time_split, false_negatives_time_split = self.analysis(filename, tr_features, tr_labels, ev_features, ev_labels, timesplit=True)
            print("time accuracy: %r" % accuracy2)
            
            metrics={"toolid": filename, "num_instances":num_instances,
                    "accuracy": accuracy, "accuracy_time_split":accuracy2, "num_walltime_errors":num_errors,
                    "false_positives": false_positives, "false_negatives": false_negatives,
                    "false_positives_time_split":false_positives_time_split, "false_negatives_time_split":false_negatives_time_split}
            print(tabulate(sorted([(k,v) for k,v in metrics.items()])))
            command = '''INSERT INTO walltime_classify (toolid, num_instances, accuracy, accuracy_time_split, num_walltime_errors, 
                        false_positives, false_negatives, false_positives_time_split, false_negatives_time_split)

                        VALUES (%(toolid)s, %(num_instances)s, %(accuracy)s, %(accuracy_time_split)s, %(num_walltime_errors)s, %(false_positives)s, 
                        %(false_negatives)s, %(false_positives_time_split)s, %(false_negatives_time_split)s)

                        ON DUPLICATE KEY UPDATE toolid=%(toolid)s, num_instances=%(num_instances)s, accuracy=%(accuracy)s, accuracy_time_split=%(accuracy_time_split)s, 
                        num_walltime_errors=%(num_walltime_errors)s, false_positives=%(false_positives)s, false_negatives=%(false_negatives)s, 
                        false_positives_time_split=%(false_positives_time_split)s, false_negatives_time_split=%(false_negatives_time_split)s;'''
        else:
            metrics={"toolid": filename, "num_instances":num_instances, "accuracy": accuracy, "num_errors":num_errors}
            command = '''INSERT INTO walltime_classify (toolid, num_instances, accuracy, num_walltime_errors) 
                            
                            VALUES (%(toolid)s, %(num_instances)s, %(accuracy)s, %(num_errors)s)
                            
                            ON DUPLICATE KEY UPDATE num_instances=%(num_instances)s, accuracy=%(accuracy)s, num_walltime_errors=%(num_errors)s;
                            '''
        # print(tabulate(sorted([(k,v) for k,v in metrics.items()])))
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
        hardware=[
        # 'destination_id',
        #  # 'galaxy_slots',
        #  'handler',
        #  'job_runner_name',
        #  # 'memtotal',
         # 'processor_count',
         'result']
        keep=parameters+filetypes+files+hardware
        columns=list(df)
        for column in columns:
            if not column in keep:
                del df[column]
        return df

    def get_false_positives(self, labels, pred):
        labels=labels.reset_index(drop=True)
        total_positive=0
        false_positive=0
        for i in range(len(pred)):
            if pred[i]=="walltime error":
                total_positive+=1
                if labels[i]=="ok":
                    false_positive+=1
        print(total_positive)
        print(false_positive)
        if total_positive == 0:
            return -1
        
        return false_positive/total_positive

    def get_false_negatives(self, labels, pred):
        labels=labels.reset_index(drop=True)
        total_negatives=0
        false_negatives=0
        for i in range(len(pred)):
            if pred[i]=="ok":
                total_negatives+=1
                if labels[i]=="walltime error":
                    false_negatives+=1
        print(total_negatives)
        print(false_negatives)
        
        return false_negatives/total_negatives
        
    def analysis(self, filename, tr_features, tr_labels, ev_features, ev_labels, timesplit=False):
        self.regr.fit(tr_features,tr_labels)
        ev_pred = self.regr.predict(ev_features)
        accuracy=sklearn.metrics.accuracy_score(ev_labels,ev_pred) 
        false_positives=self.get_false_positives(ev_labels, ev_pred)
        false_negatives=self.get_false_negatives(ev_labels, ev_pred)
        return float(accuracy), float(false_positives), float(false_negatives)
        
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


toolids=getfiles("failed_jobs/walltime_csv")
toolids=[i[:-4] for i in toolids]
# toolids=["dummy"]
print(toolids)
# toolids=["iuc_vsearch_vsearch_masking_1.9.7.0"]

for i in range(len(toolids)):
    print(i, toolids[i])
    TrainModel(toolids[i])

cursor.close()
cnx.close()
