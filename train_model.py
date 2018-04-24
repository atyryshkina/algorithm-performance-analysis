import argparse
import scipy
from sklearn.externals import joblib

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

class TrainModel(object):

    def __init__(self, args):

        self.args = args

        # load the input file into a pandas dataframe
        if args.filename.endswith(".csv"):
            self.df=pd.read_csv(args.filename)
        elif args.filename.endswith(".tsv"):
            self.df=pd.read_csv(args.filename, sep="\t")
        else:
            raise ValueError("unrecognized filetype: %s. I only accept tsv or csv files" % self.args.filename)

        df_features, df_labels = self.df, self.df.pop(self.args.runtime_label)
        df_labels = np.log1p(df_labels)

        # prepare the data for the RandomForestRegressor
        print("setting up...")
        chooser = ChooseFeatureColumns()
        scaler = MyMapper()
        regr = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=12)

        self.pipe = sklearn.pipeline.Pipeline([
            ('chooser',chooser),
            ('scaler', scaler),
            ('regr', regr)
        ])
        
        test_size = 0.2
        test_start=len(df_labels)-int(len(df_labels)*test_size)
        print(test_start, len(df_labels))

        # print("self.args.split_randomly ", self.args.split_randomly)

        if ast.literal_eval(self.args.split_train_test) and (ast.literal_eval(self.args.split_randomly)):
            tr_features, ev_features, tr_labels, ev_labels = sklearn.model_selection.train_test_split(df_features, df_labels, test_size=test_size)
            print("splitting randomly")
        elif ast.literal_eval(self.args.split_train_test):
            tr_features, tr_labels, ev_features, ev_labels = df_features[:test_start], df_labels[:test_start], df_features[test_start:], df_labels[test_start:]
            print("splitting non-randomly")
        else:
            tr_features, tr_labels, ev_features, ev_labels = df_features,df_labels,df_features,df_labels
            print("not splitting")
    

        print("fitting the model...")
        self.pipe.fit(tr_features, tr_labels)
        ev_pred = self.pipe.predict(ev_features)

        # unlog the runtimes (they were previously log transformed in the function clean_data()) 
        cq=pd.DataFrame()
        cq["labels"] = np.expm1(ev_labels)
        cq["pred"] = np.expm1(ev_pred)

        # do the final analysis
        self.analysis_of_results(cq, list(df_features))

        print("done")

    def analysis_of_results(self, cq, features):
        # some metrics
        r2_score = sklearn.metrics.r2_score(cq["labels"], cq["pred"])
        pearson = scipy.stats.pearsonr(cq["labels"], cq["pred"])
        mse = sklearn.metrics.mean_squared_error(cq["labels"], cq["pred"])

        print("r2 score: %r" % r2_score)
        print("pearson: %r" % pearson[0])
        print("mse: %r" % mse)

        # save model
        joblib.dump(self.pipe, self.args.model_outfile)
        print("saved model to: %s" % self.args.model_outfile) 


        # save a plot
        if (self.args.plot_outfile != None):
            plt.figure(figsize=(10,10))
            plt.scatter(cq["labels"],cq["pred"])
            # plt.xlim([0,cq["labels"].max()])
            # plt.ylim([0,cq["labels"].max()])
            # plt.plot([0,cq["labels"].max()],[0,cq["labels"].max()], 'r')
            plt.xlabel("Real Runtime")
            plt.ylabel("Predicted Runtime")
            plt.title("Mean predictions")
            plt.savefig(self.args.plot_outfile)
            print("saved a plot to: %s" % self.args.plot_outfile)


def main():
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    parser.add_argument("--runtime_label", dest='runtime_label', action='store', default="runtime")
    parser.add_argument("--split_train_test", dest='split_train_test', action='store', default="False")
    parser.add_argument("--split_randomly", dest='split_randomly', action='store', default="True")
    parser.add_argument('--plot_outfile', dest='plot_outfile', action='store', default="plot.png", help='png output file.')
    parser.add_argument('--model_outfile', dest='model_outfile', action='store', default='model.pkl', help='pkl output file.')
    args = parser.parse_args()
    return TrainModel(args)

if __name__ == '__main__':
    main()
    exit(0)
