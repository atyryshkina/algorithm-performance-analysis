# Feature Importances with Random Forests

This tool estimates the relative impact of parameters on the runtime of tools. It does so by fitting a Random Forest Regressor to a historical dataset (of parameters and runtimes) and determining the [Mean Decrease Impurity](http://papers.nips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-trees.pdf) of each parameter. The Mean Decrease Impurity is an estimate of how much the Random Forest uses the parameter in it's decisions.



## How to use

The tool accepts a .tsv or .csv file. [Here is a sample csv file].
The file should have one column labeled "runtime", which the Random Forest will treat as the dependent variable it tries to predict.

The tool will warn you if you have a parameter with more than 30 categories, or if you have a parameter that is monotonic (such as an id or a constant number)

The tool should take less than a minute to finish. It will save the important_features in a .tsv file, and optionally save a plot to a .png file.



##### options

|options|default|description|
|--- | --- |---|
|--filename| required | the name of the .csv or .tsv file|
--outfile| default="feature_importances.tsv"| name of a .tsv file where you want the output|
|--plot_outfile|default=None|If you want a plot, use this to name the .png file you want it saved to. If you don't want a plot, leave this as default.
|--runtime_label|default="runtime"| this specifies the label of the variable you want to predict in your dataset
|--treat_categorical_features_as_one|default=True|whether to give the importances of categorical features with one number, or to give the importance of each seperate category (e.g. give importance of "color" vs. give importance of ("color_blue", "color_green", "color_yellow"))
|--delete_monotonic|default=False| whether to use monotnic features
|--split_train_test|default=False| if you are making a graph, do you want to split the dataset into a training and testing set, or you want to use the whole dataset for both training and testing