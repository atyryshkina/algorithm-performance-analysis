#######
import argparse
import pandas as pd
import numpy as np
import scipy
import operator

from sklearn_pandas import gen_features
from sklearn_pandas import DataFrameMapper
import sklearn
from sklearn.preprocessing import FunctionTransformer
from skgarden import RandomForestRegressor

from matplotlib import pyplot as plt

class GetFeatureImportances(object):

    def __init__(self, args):

        self.args = args

        # load the input file into a pandas dataframe
        if args.filename.endswith(".csv"):
            self.df=pd.read_csv(args.filename)
        elif args.filename.endswith(".tsv"):
            self.df=pd.read_csv(args.filename, sep="\t")
        else:
            raise ValueError("unrecognized filetype: %s. I only accept tsv or csv files" % self.args.filename)

        # prepare the data for the RandomForestRegressor
        print("preprocessing")
        self.df=self.clean_data(self.df)
        self.df_features, self.df_labels = self.transform_data(self.df, self.args.runtime_label)

        # Declare the Random Forest Regressor
        self.regr = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_depth=10)

        # fit the data to the RandomForestRegressor and analyze results
        # Split the data into a training and testing set is split_train_test==true
        if self.args.split_train_test:
            self.train_test_split_analysis(self.df_features,self.df_labels)
        else:
            self.no_split_analysis(self.df_features,self.df_labels)

        print("done")

    def train_test_split_analysis(self, df_features, df_labels):
        # randomly split the data into training and testing set, where the testing set size is 0.2 of the total data
        test_size = 0.2
        tr_features, ev_features, tr_labels, ev_labels = sklearn.model_selection.train_test_split(df_features, df_labels, test_size=test_size)

        print("fitting the model")
        self.regr.fit(tr_features, tr_labels)

        print("making predictions")
        ev_pred = self.regr.predict(ev_features, return_std=True)

        # unlog 
        cq=pd.DataFrame()
        cq["labels"] = np.expm1(ev_labels)
        cq["pred"] = np.expm1(ev_pred[0])

        self.analysis_of_results(cq, list(ev_features))


    def no_split_analysis(self, df_features, df_labels):

        # fit model 
        print("fitting the model")
        self.regr.fit(df_features,df_labels)

        # make predictions
        print("making predictions")
        df_pred = self.regr.predict(df_features, return_std=True)

        # unlog the runtimes (they were previously log transformed in the function clean_data()) 
        cq=pd.DataFrame()
        cq["labels"] = np.expm1(df_labels)
        cq["pred"] = np.expm1(df_pred[0])

        # do the final analysis
        self.analysis_of_results(cq, list(df_features))

    def analysis_of_results(self, cq, features):
        # some metrics
        r2_score = sklearn.metrics.r2_score(cq["labels"], cq["pred"])
        pearson = scipy.stats.pearsonr(cq["labels"], cq["pred"])
        mse = sklearn.metrics.mean_squared_error(cq["labels"], cq["pred"])

        # get feature importances
        feature_importances = dict(zip(features, self.regr.feature_importances_))

        # if treat_categorical_features_as_one == true 
        # reunite the categorical features by adding up all the individual categories importances
        if self.args.treat_categorical_features_as_one:
            categorical_feature_importances=list(self.df.select_dtypes(include=[object]))
            categorical_feature_importances = dict((key,0) for key in categorical_feature_importances)
            remove=[]
            for feat in categorical_feature_importances:
                for key in feature_importances:
                    if key.startswith(feat):
                        categorical_feature_importances[feat] += feature_importances[key]
                        remove.append(key)

            feature_importances = {key:feature_importances[key] for key in feature_importances if key not in remove}
            feature_importances = {**feature_importances, **categorical_feature_importances}

        # save feature importances
        feature_importances=pd.DataFrame(sorted(feature_importances.items(), key=operator.itemgetter(1), reverse=True))
        feature_importances.to_csv(self.args.feature_importances_outfile, sep="\t")
        print("saved feature importances to '%s'" % self.args.feature_importances_outfile)

        # save a plot
        if (self.args.plot_outfile != None):
            plt.figure(figsize=(10,10))
            plt.scatter(cq["labels"],cq["pred"])
            # plt.xlim([0,cq["labels"].max()])
            # plt.ylim([0,cq["labels"].max()])
            plt.plot([0,cq["labels"].max()],[0,cq["labels"].max()], 'r')
            plt.xlabel("Real Runtime")
            plt.ylabel("Predicted Runtime")
            plt.title("Mean predictions")
            plt.annotate("r2 = %.2f" % (r2_score),xy=(0.3, 0.95), xycoords='axes fraction',
                        size=14, ha='right', va='top',
                        bbox=dict(boxstyle='round', fc='w'))
            plt.savefig(self.args.plot_outfile)
            print("saved a plot to '%s'" % self.args.plot_outfile)

    def transform_data(self, df, runtime_label):
        df_features, df_labels = df, df.pop(runtime_label)

        # Define which features are going to be transformed to a range of 0 to 1 (continuous)
        nfeats = gen_features(
            columns=[[i] for i in list(df_features.select_dtypes(include=[float]))],
            classes=[sklearn.preprocessing.MinMaxScaler]  
        )

        # Define which features are going to be binarized (categorical)
        sfeats = gen_features(
            columns=list(df.select_dtypes(include=[object])),
            classes=[sklearn.preprocessing.LabelBinarizer]  
        )

        # Do the transformations defined above
        mapper = DataFrameMapper(nfeats+sfeats,df_out=True)
        df_features = mapper.fit_transform(df_features)

        return df_features, df_labels

    def clean_data(self, df):

        # check to make sure we know which column holds the runtimes
        if not self.args.runtime_label in list(self.df):
            raise ValueError('runtime column label "%s" is not found in the file. you can change the runtime column label name with --runtime_label' % self.args.runtime_label)

        # make an alert if the number of categories in a column exceeds threshold_num_of_categories
        threshold_num_of_categories = 30

        # determine if feature is continuous or categorical 
        #                             (float or str)
        for column in df:
            try:
                df[column] = df[column].astype(float)
                df[column]=df[column].fillna(0)
            except (ValueError,TypeError):
                df[column] = df[column].astype(str)
                if len(df[column].unique())>threshold_num_of_categories:
                        print('"%s" has %d categories' % (column,len(df[column].unique()))) 
            
            if df[column].is_monotonic:
                print('"%s" is monotonic' % column) # you might not want to use features that are monotonic
                if self.args.del_monotonic:
                    print('throwing "%s" away' % column)
                    del df[column]


        # Most tools have an uneven distribution of runtimes. Performing a log transform on the runtimes leads to better results.
        df[self.args.runtime_label] = np.log1p(df[self.args.runtime_label])

        return df

        

def main():
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    parser.add_argument("--runtime_label", dest='runtime_label', action='store', default="runtime")
    parser.add_argument("--delete_monotonic", dest='del_monotonic', action='store', default=False)
    parser.add_argument("--split_train_test", dest='split_train_test', action='store', default=False)
    parser.add_argument("--treat_categorical_features_as_one", dest='treat_categorical_features_as_one', action='store', default=True)
    parser.add_argument('--plot_outfile', dest='plot_outfile', action='store', default=None, help='png output file.')
    parser.add_argument('--outfile', dest='feature_importances_outfile', action='store', default='feature_importances.tsv', help='tsv output file.')
    args = parser.parse_args()
    return GetFeatureImportances(args)

if __name__ == '__main__':
    main()
    exit(0)