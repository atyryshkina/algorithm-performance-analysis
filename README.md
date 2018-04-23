# Algorithm Performance Analysis 

This is a collection of tools to analyze the performance of complex algorithms. The tools use machine learning and statistics to look at real data to determine performance bottlenecks and the relationships between parameters and runtime. In the future we want to help determine appropriate hardware constraints (e.g. memory) for efficient use of resources. We also hope to make a tool for reliable runtime prediction. It is made with [Galaxy Project](https://galaxyproject.org/) admins in mind.


### What this can do:
* collect historical data from your galaxy database
* manipulate the data
* determine which parameters affect the runtime of the algorithm
* inspect the impact of a single parameter on runtime

### Future plans:
* determine an appropriate run walltime for tools
* determine appropriate memory allocation
* determine approptiate processor cores


## Collect Data From Galaxy Database

use get_tool_run_info.py to collect data from your [galaxy](https://galaxyproject.org/) database. The output file formats supported are csv and json. 

There is also an example data file available: [bwa_mem_0.7.15.1_example.csv](examples/bwa_mem_0.7.15.1_example.csv)

|options|default|description|
|--- | --- |---|
|--config| default="config/galaxy.yml" | config of your galaxy |
|--toolid| required | tool id (e.g. "toolshed.g2.bx.psu.edu/repos/devteam/tophat2/tophat2/0.9")|
|--outfile|default="job_data.csv"|Output file, extension determines format.|

### Best input format for current and future analysis

The best input format to use is a comma-separated values (CSV).

## Manipulate Data

Use csv_file_manipulation.ipynb for suggestions on how to view and manipulate csv data with python. It has examples to inspect the data, delete data, transform data, and combine data.

## Feature Importances with Random Forests

This tool (feature_importances_with_random_forests.py) estimates the relative impact of parameters on the runtime of tools. It does so by fitting a Random Forest Regressor to a historical dataset (of parameters and runtimes) and determining the [Mean Decrease Impurity](http://papers.nips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-trees.pdf) of each parameter. The Mean Decrease Impurity is an estimate of how much the Random Forest uses the parameter in it's decisions.


The tool accepts a .tsv or .csv file. [Here is a sample csv file](examples/bwa_mem_0.7.15.1_example.csv).
The file should have one column labeled "runtime", which the Random Forest will treat as the dependent variable to predict.

The tool will warn you if you have a parameter with more than 30 categories, or if you have a parameter that is monotonic (such as an id or a constant number)

The tool should take less than a minute to finish. It will save the important features in a .tsv file, and optionally save a plot to a .png file.





|options|default|description|
|--- | --- |---|
|--filename| required | the name of the .csv or .tsv file with the data|
|--outfile| default="feature_importances.tsv"| name of a .tsv file where you want the output|
|--plot_outfile|default=None|If you want a plot, use this to name the .png file you want it saved to. Otherwise, leave as default.
|--runtime_label|default="runtime"| this specifies the label of the variable to predict in your dataset
|--unite_categorical_features|default=True|whether to give the importances of categorical features with one number, or to give the importance of each seperate category (e.g. give importance of "color" vs. give importance of "color_blue", "color_green", "color_yellow")


## Inspect single feature

Inspect the effect of a single parameter on runtime while holding all of the other parameters constant. (single_feature_analysis.py)

The tool accepts a tsv or csv file, and it requires a feature_of_interest and a runtime column to be named. It sorts the dataset into sets of jobs that all have similar parameters. Then it saves the parameters of the largest sets of jobs in files named parameters_i.tsv and saves plots of feature_of_interest vs runtime to plot_i.png. 

For example, say you put in a csv file like this:

|runtime|feature_of_interest|category|number|
|---|---|---|---|
|5|1|True|0|
|10|5|False|0|
|2|3|True|7|
|3|2|True|6|
|4|1|False|25|

First, the continuous columns will be placed into bins like so

|runtime|feature_of_interest|category|number|
|---|---|---|---|
|5|1|True|0|
|10|5|False|0|
|2|3|True|(5, 7.5]|
|3|2|True|(5, 7.5]|
|4|1|False|(23.5, 25]|

The largest set with equivelent parameters is chosen:

|runtime|feature_of_interest|category|number|
|---|---|---|---|
|2|3|True|(5, 7.5]|
|3|2|True|(5, 7.5]|

And the values of the equivelent parameters is saved to parameters_i.tsv and the plot of runtime vs. feature_of_interest is saved to plot_i.png. You can choose to inspect the n largest sets with the option --num_to_plot.


|options|default|description|
|--- | --- |---|
|--filename| required | the name of the .csv or .tsv file with the data|
|--feature_of_interest| requireds | name parameter to inspect|
|--runtime_label|default="runtime"| this specifies the label of the variable to predict in your dataset
|--num_to_plot|default=1| number of parameter sets to examine
|--outdir| default='single_feature' | the name of the directory the output file should go|


# Predicting

Once you are ready to build a random forest predictor, train_model.py will train a random forest and save the model in a pickle file. 

|options|default|description|
|--- | --- |---|
|--filename| required | the name of the .csv or .tsv file with the data|
|--model_outfile| default='model.pkl'| name of a .pkl file where you want the output|
|--plot_outfile|default='plot.png'|If you want a plot, use this to name the .png file you want it saved to. Otherwise, leave as default.
|--runtime_label|default="runtime"| this specifies the label of the variable to predict in your dataset
|--split_train_test|default=False| if you are making a plot, do you want to split the dataset into a training and testing set, or do you want to use the whole dataset for both training and testing
|--split_randomly|default=False| if split_train_test == True, this specifies whether to split the data randomly


Then use the model.pkl to predict the runtime of new instances using predict_runtime_with_model.py

|options|default|description|
|--- | --- |---|
|--filename| required | the name of the .csv or .tsv file with the data|
|--model_filename| required | the name of the .pkl of the model|
|--plot_outfile|default='plot.png'|To get a plot you must also provide a runtime_label
|--runtime_label|default="runtime"| may be set to None. this specifies the label of the variable to predict in your dataset. may be left empty if you don't want prediction metrics
|--single_prediction|default=False| if you want a single prediction, setting this to True will return the prediction and nothing else


