import argparse
import pandas as pd
import numpy as np
import scipy
import operator
import math

from sklearn_pandas import gen_features
from sklearn_pandas import DataFrameMapper
import sklearn
from sklearn.preprocessing import FunctionTransformer
from skgarden import RandomForestRegressor

from matplotlib import pyplot as plt

class AnalyzeSingleFeature(object):

    def __init__(self, args):
        self.args = args

        # load the input file into a pandas dataframe
        if args.filename.endswith(".csv"):
            self.df=pd.read_csv(args.filename)
        elif args.filename.endswith(".tsv"):
            self.df=pd.read_csv(args.filename, sep="\t")
        else:
            raise ValueError("unrecognized filetype: %s. I only accept tsv or csv files" % self.args.filename)

        # check the feature of interest if the dataframe
        if not self.args.feature_of_interest in self.df:
            raise KeyError("feature of interest %s is not in dataframe" % (self.args.feature_of_interest))

        # the feature names will be transformed in this program
        # transform the input columns now
        self.args.runtime_label = self.args.runtime_label.replace('.', '_').replace('|', '_')
        self.args.feature_of_interest = self.args.feature_of_interest.replace('.', '_').replace('|', '_')
        # prepare the data for the RandomForestRegressor
        print("preprocessing")
        self.df=self.clean_data(self.df)

        categorized_dfs, buckets=self.categorize_parameters(self.df, ignore=[self.args.runtime_label, self.args.feature_of_interest])

        self.analyze_single_feature(categorized_dfs, buckets, self.args.runtime_label, self.args.feature_of_interest, int(self.args.num_to_plot),
            self.args.outdir)
        print("done")

    def analyze_single_feature(self, categorized, buckets, runtime_label, feature_of_interest, num_to_plot, dirr):

        for i in range(num_to_plot):

            # save plot
            # if self.df[feature_of_interest].dtype == float:
            print(categorized[i].shape)
            plt.figure(figsize=(10,10))
            plt.scatter(categorized[i][feature_of_interest], categorized[i][runtime_label])
            plt.xlabel(feature_of_interest)
            plt.ylabel(runtime_label)
            plt.title("%s vs %s" % (feature_of_interest,runtime_label))
            plt.annotate("number of points = %s" % (categorized[i].shape[0]),xy=(0.4, 0.95), xycoords='axes fraction',
                        size=14, ha='right', va='top',
                        bbox=dict(boxstyle='round', fc='w'))
            plt.savefig(str(dirr)+'/plot_'+str(i))
            print("saved a plot to '%s/plot_%s.png'" % (dirr,str(i)))

            # save parameters
            params=categorized[i].iloc[0]
            params=params.drop([runtime_label,feature_of_interest])
            for binned_param in buckets.index:
                binn=self.find_bucket(params[binned_param],
                                               buckets.loc[binned_param]["bucket_size"],
                                               buckets.loc[binned_param]["num_buckets"])
                if binn == (0,0):
                    params[binned_param] == 0
                else:
                    params[binned_param]="%d-%d"%(self.find_bucket(params[binned_param],
                                               buckets.loc[binned_param]["bucket_size"],
                                               buckets.loc[binned_param]["num_buckets"]))

            params=pd.DataFrame(params.rename("value"))
            params.index.name="parameter"
            params.to_csv(str(dirr)+'/parameters_'+str(i)+".tsv", sep="\t")
            print("saved parameters to '%s/parameters_%s.tsv'" % (dirr,str(i)))

    def categorize_parameters(self,df, ignore=None):
        categorized=[] # list of dataframes with similar features

        # change the names of the columns to suitable names for querying
        cols = df.columns
        cols = cols.map(lambda x: x.replace('.', '_').replace('|', '_') if isinstance(x, (str, 'utf-8')) else x)
        df.columns = cols
        parameters=list(cols)

        # remove features to ignore from the parameters list
        for to_remove in [x for x in ignore]:
            parameters.remove(to_remove)

        ##########################  make bins for continuous features
        buckets = pd.DataFrame(columns=("parameter","bucket_size","num_buckets"))
        high_dim_params=[]
        threshold=20
        num_buckets=10
        for param in parameters:
          if (df[param].dtype == float) and (len(df[param].unique()) > threshold):
            bucket_size= math.ceil((df[param].max()-df[param].min())/ num_buckets)
            high_dim_params.append(param)
            buckets = buckets.append([{
                'parameter':param,
                "bucket_size":bucket_size,
                "num_buckets":num_buckets
            }], ignore_index=True)
        buckets.index=buckets["parameter"]
        del buckets["parameter"]
        print("")
        print("bucket info")
        print(buckets)
        print("")

        ########################## 
        print("sorting...")
        
        i=0
        checked=[]
        categorized=[] # list of dataframes with similar features
        for index in list(df.index):
          i+=1
          if i%1000==0:
            print(str(i)+"/"+str(len(df.index)))
          if index in checked:
            pass
          else:
            
            query=""
            for param in parameters:
              if not pd.isnull(df.loc[index][param]):
                if param in list(buckets.index):
                  boundl,boundu=self.find_bucket(df.loc[index][param], buckets.loc[param]["bucket_size"],
                                           buckets.loc[param]["num_buckets"])
                  query+=("%r <= %s & %s <= %r & " %(boundl, param, param, boundu))
                elif type(df.loc[index][param]) == str:
                  query+= param+ ' == "' +str(df.loc[index][param])+ '" & '
                else:
                  query+= param+ " == " +str(df.loc[index][param])+ " & "
              else:
                query+=param+ " != " +param+" & "
            query=query[:-3]

            sdf=df.query(query)
            categorized.append(sdf)
            checked+=list(sdf.index)

        categorized.sort(key=lambda x: x.shape[0], reverse=True)

        return categorized, buckets

    def find_bucket(self,value, bucket_size, num_buckets):
          if value == 0:
            return (0, 0)
          for i in range(num_buckets):
            if (i*bucket_size < value) and (value <= (bucket_size+i*bucket_size)):
              return  ((i*bucket_size), (bucket_size+i*bucket_size))
          return ((num_buckets*bucket_size), (bucket_size+num_buckets*bucket_size))

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

        return df

        

def main():
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    parser.add_argument("--feature_of_interest", dest='feature_of_interest', action='store', required=True)
    parser.add_argument("--runtime_label", dest='runtime_label', action='store', default="runtime")
    parser.add_argument("--delete_monotonic", dest='del_monotonic', action='store', default=False)
    parser.add_argument('--outdir', dest='outdir', action='store', default='single_feature', help='output folder to place plots.')
    parser.add_argument('--num_to_plot', dest='num_to_plot', action='store', default=1, help='how many plots')
    args = parser.parse_args()
    return AnalyzeSingleFeature(args)

if __name__ == '__main__':
    main()
    exit(0)
