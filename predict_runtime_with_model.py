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



class PredictRuntime(object):

    def __init__(self, args):

        self.args = args

        # load the input file into a pandas dataframe
        if args.filename.endswith(".csv"):
            self.df=pd.read_csv(args.filename)
        elif args.filename.endswith(".tsv"):
            self.df=pd.read_csv(args.filename, sep="\t")
        else:
            raise ValueError("unrecognized filetype: %s. I only accept tsv or csv files" % self.args.filename)

        if self.args.runtime_label == None:
            df_features, df_labels = self.df, None
        else:
            df_features, df_labels = self.df, self.df.pop(self.args.runtime_label)
            df_labels = np.log1p(df_labels)

        # prepare the data for the RandomForestRegressor
        if not self.args.single_prediction:
            print("setting up...")
        self.pipe = joblib.load(self.args.model_filename)    

        if not self.args.single_prediction:
            print("predicting")
        df_pred = self.pipe.predict(df_features)

        # unlog the runtimes (they were previously log transformed in the function clean_data()) 
        cq=pd.DataFrame()
        cq["pred"] = np.expm1(df_pred)
        if not df_labels is None:
           cq["labels"] = np.expm1(df_labels)

        # do the final analysis
        self.analysis_of_results(cq, list(df_features))

        if not self.args.single_prediction:
            print("done")

    def analysis_of_results(self, cq, features):
        # some metrics
        if ('labels' in cq):
            r2_score = sklearn.metrics.r2_score(cq["labels"], cq["pred"])
            pearson = scipy.stats.pearsonr(cq["labels"], cq["pred"])
            mse = sklearn.metrics.mean_squared_error(cq["labels"], cq["pred"])

            print("r2 score: %r" % r2_score)
            print("pearson: %r" % pearson[0])
            print("mse: %r" % mse)

        if self.args.single_prediction:
            print(cq["pred"][0])


        # save a plot
        if (self.args.plot_outfile != None) and ('labels' in cq):
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
    parser.add_argument('--plot_outfile', dest='plot_outfile', action='store', default="plot.png", help='png output file.')
    parser.add_argument('--model_filename', dest='model_filename', action='store', default='model.pkl', help='.pkl model file.')
    parser.add_argument('--single_prediction', dest='single_prediction', action='store', default=False)
    args = parser.parse_args()
    return PredictRuntime(args)

if __name__ == '__main__':
    main()
    exit(0)
