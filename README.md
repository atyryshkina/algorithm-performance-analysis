# The Analysis of Data Collected by the Galaxy Project

## Abstract
The Main public server of the Galaxy Project (https://usegalaxy.org) has been collecting extensive job run data on all analyses since 2013. This large collection of job runs with attributes can be leveraged to determine more efficient ways for allocation of server resources. In addition, these data represent the largest, most comprehensive dataset available to date on the runtime dynamics for some of the most popular biological data analysis software. In this work we were aiming at creating a model for runtime prediction of complex algorithms trained on real data. In this paper we will:

1. Present statistical summaries of the dataset, describe its structure, identify the presence of
undetected errors, and discuss any other insights into the Galaxy server that we believe will be
useful to the community.
2. Confirm that the random forest regressor gives the best performance for predicting the runtime
of complex algorithms as was seen by Hutter et al.
3. Discuss the benefits and drawbacks of using a quantile random forest for creating runtime
prediction confidence intervals.
4. Present an alternative approach for choosing a walltime for complex algorithms with the use of
a random forest classifier.

Studying the Galaxy Project dataset reveals that there may be room to fine tune the resource allocation.
The ability to determine appropriate walltimes will save server resources from jobs that result in errors
undetected by the server — such as jobs that fall into infinite loops. Once freed, these resources can then be used to run jobs in the queue without the need to allocate additional hardware.

## Table of Contents


- [Background](#what-is-the-galaxy-project)
  + [What is the Galaxy Project](#what-is-the-galaxy-project)
  + [scikit-learn and machine learning](#scikit-learn-and-machine-learning)
  + [Previous work on runtime prediction of programs](#previous-work-on-runtime-prediction-of-programs)
- [Overview of Data](#overview-of-data)
  + [Distribution of the Data](#distribution-of-the-data)
  + [Undetected Errors](#undetected-errors)
  + [user selected parameters](#user-selected-parameters)
  + [attribute preprocessing](#attribute-preprocessing)
- [Model Comparison](#model-comparison)
- [Estimating a Range of Runtimes](#estimating-a-range-of-runtimes)
- [Using a random forest classifier](#using-a-random-forest-classifier)
- [Future Work](#future-work)
- [References](#references)

### What is the Galaxy Project

The Galaxy Project is a platform that allows researchers to run popular bioinformatics analyses quickly and easily. In the past, trying to run a popular analysis would require downloading, configuring, and troubleshooting the analysis software on one's own machines. This can be a difficult and time consuming task.

With the Galaxy Project, researchers can run analyses using the graphical user interface in their browser. To do so, the user needs only to connect to a Galaxy server (e.g. https://www.usegalaxy.org), upload their data, choose the analysis and the analysis parameters, and hit run.

The Galaxy Project has been storing the information about the execution of these analyses since 2013, and, to date, has the largest dataset of the performance of popular bioinformatics tools. We begin to study this dataset in this paper.

For more information visit https://www.galaxyproject.org.

### scikit-learn and machine learning

scikit-learn is a library of machine learning tools for Python. It has classes for anything machine learning related - from data preprocessing to regression and classification to model evaluation. The sci-kit learn library is the main library we used in our tests, specifically the regression and classification classes.

In this paper our main tool for regression and classification was the random forest, so we will briefly go over random forests now.

A random forest is a collection of decision trees which themselves are series of questions asked about an object. At the end of the questions, a previously unknown attribute of the object is guessed. An example of a decision tree is shown below. In this case, the decision tree tries predicting how long a job is going to take.

![alt text](images/simple_decision_tree.png)

The decision tree learns what questions to ask by training on a set of previous jobs. At each node, it looks at a subset of attributes and chooses to split the data in the way that minimizes variability in the subsequent two nodes. In this way, it sorts objects by similarity of the dependent and independent variables. In the figure below, a decision tree is training on a set of 100 jobs. All 100 jobs start at the root node, and they are split up as they travel down the tree. Once the tree is trained, it can then be used to predict the runtime of previously unseen jobs.

![alt text](images/decision_tree_vertical.png)

A random forest is a collection of decision trees, each of which are trained with a unique random seed. The random seed determines which sub-sample of the data each decision tree is trained and which sub-sample of attributes each tree uses and which splits each tree attempts. By implementing these constraints, the random forest protects itself from overfitting - a problem to which decision trees are susceptible.

Incidentally, the decision tree also offers a way to see which independent attributes have the greatest effect on the dependent attribute. The more often a decision tree uses an attribute to split a node, the larger its implied effect on the dependent attribute. The scikit-learn Random Forest classes have a way of getting this information with the feature_importances_ class attribute.

### Previous work on runtime prediction of programs

The prediction of runtimes of complex algorithms using machine learning approaches has been done before. [[1]](https://doi.org/10.1007/11508380_24)[[2]](https://doi.org/10.1109/CCGRID.2009.58)[[3]](https://doi.org/10.1145/1551609.1551632)[[4]](https://doi.org/10.1109/CCGRID.2009.77)[[5]](https://doi.org/10.1007/11889205_17)

In a few works, new machine learning methods are designed specifically for the problem. In 2008, [Gupta et al.](http://doi.org/10.1109/ICAC.2008.12) proposed a tool called a PQR (Predicting Query Runtime) Tree to classify the runtime of queries that users place on a server. The PQR tree dynamically choose runtime bins during training that would be appropriate for a set of historical query runtimes. The paper notes that the PQR Tree outperforms the decision tree.

Most previous works tweak and tailor old machine learning methods to the problem. For instance, in 2010, [Matsunaga](http://doi.org/10.1109/CCGRID.2010.98) enhances on the PQR Tree by adding linear regressors at its leaves, naming it PQR2. They test their model against two bioinformatic analyses tools: BLAST (a local alignment algorithm) and RAxML (a phylogenetic tree constructer). The downside of PQR2 is that there are not readily available libraries of the model in popular programming languages like Python or R.

The most comprehensive survey of runtime prediction models was done by [Hutter et al.](https://doi.org/10.1016/j.artint.2013.10.003) In 2014. In the paper, they compared 11 regressors including ridge regression, neural networks, Gaussian process regression, and random forests although they did not include PQR tree in their evaluations. They found that the random forest outperforms the other regressors in nearly all assessments and is able to handle high dimensional data without the need of feature selection.

In our paper, we verify that random forests are the best model for the regression, discuss the merits of quantile regression forests, and present a practical approach for determining an appropriate walltime with the use of a classifier.

## Overview of Data

All of the tool execution on https://usegalaxy.org are tracked. The data is collected using the Galactic Radio Telescope (GRT), which records a comprehensive set of job run attributes.

This includes:

* create time
* runtime
* user selected parameters (tool specific)
* state ("ok", "error", or other)
* input data
  - size of input data
  - filetype of input data (bam, fastq, fastq.gz, etc.)
* hardware information
  - memory allocated
  - swap memory allocated
  - number of processors allotted
  - destination id (which node the job was run on)
* user information
    - user id
    - session id
    - history id


The Main Galaxy dataset contains runtime data for 1372 different tools that were run on the Galaxy Servers over the past five years. A statistical summary of those tools, ordered by most popular, can be found [here](summary_of_tools.csv). The runtimes are in minutes.

The Galaxy server has three different clusters with different hardware specifications. The hardware on which a job was run is recorded in the dataset. [I need to put hardware specs here. or a link to them]

The CPUs are shared with other jobs running on the node, so the performance of jobs is effected by the server load at the time of execution. This attribute is not in the published dataset yet, but we began tracking it not long before writing this, and will publish those datasets when available.


#### Distribution of the Data

Typically, machine learning algorithms, such as, random forests and neural networks prefer to use data with a normal distribution. The distribution of runtimes and file sizes in the Galaxy dataset are highly skewed. The distribution for a tool called BWA (Galaxy version 0.7.15.1) can be seen below.

![alt text](images/runtimes3.png)

![alt text](images/filesize_bwamem.png)

In this project, we address this skewness by doing a log transform on the data.
We use numpy's log transformer numpy.log1p which transforms the data by log(1+x)

![alt text](images/log_runtimes3.png)

![alt text](images/log_filesize3.png)

This transformation works for most of the runtime and input file size attributes to give a more balanced distribution.

#### Undetected Errors

One hurdle the dataset presents is that it contains undetected errors - errors that occurred but were not recorded.

For example, some tools require that an input file is provided. `bwa_mem Galaxy version 0.7.15.1` ([link](https://toolshed.g2.bx.psu.edu/view/devteam/bwa/53646aaaafef))  is one such tool. If an input file is not provided, bwa_mem should not run at all or else the run will result in an error. In spite of this, the number of jobs executing this tool in the dataset that ran successfully and without an input file is 49 or 0.25% of "successful" jobs. These jobs should not have run at all, and yet they are present in the dataset and marked as successfully completed.

Whether the errors were be caused by bugs in the tool code, malfunctions in the server, mistakes in record keeping, or a combination of these, the presence of these of errors is troubling. With undetected input file errors, it is trivial to identify and remove the culprits from the dataset. However, these errors call into question the validity of the rest of the data. If there are many other jobs similarly mislabelled as "successfully completed" that are not as easily identified as input file errors, and these mislabelled jobs are used to train a machine learning model, they could skew the predictions.

Another method of screening the dataset for undetected errors is by looking for jobs that ran faster than possible and jobs that ran slower than probable. A job that finishes in an unreasonably short time (such an human genome alignment job that finishes in 6 seconds), or a job that finishes in an unreasonably long time (such as a ??). However, identifying these errors requires the trained eye of someone who is both familiar with the tools and has ample time to look through the dataset.

Using this heuristic, we can account for undetected errors by getting rid of the jobs that took the longest and the shortest amount of time to complete.

![alt text](images/gaus_dist2.png)

This requires choosing quantiles of contamination for each tool. In the figure above the quantiles used are 2.5%. For bwa_mem - an alignment algorithm - 2.5% of jobs in the collected data took 4 seconds or less to finish. Are all of these jobs undetected errors? If we increase the unreasonable runtime threshold, we see that 5% of jobs took 5 seconds or less to finish. It is difficult, even for a human, to decide if these recordings are reasonable job runtimes.

Since we know that the two variables that have the greatest affect on the runtime of bwa_mem are input file size and reference file size. The larger the file sizes, the longer it would take for the job to run. We should be considering these variables when looking for undetected errors. One method of doing this is by freezing all of the other variables and only looking at the relationship between these input file sizes and runtime.

In the following figures all of the user selected parameters are frozen except for input file size. We were able to freeze the reference file size because many reference genomes, such as the human genome, are popular and commonly used.

![alt text](images/hg19.png)

The reference file, hg19 is the human genome

![alt text](images/hg38patch.png)

The reference file, hg38 is another version of the human genome.

The first graph shows a strong correlation between input file and runtime. This is the correlation we expect. The outliers that we remove are the data points in the bottom right corner. We can do this safely because, while it is possible for a job to run longer than the correlation displayed on the graph, it is impossible for jobs to run faster.

The second graph, displays complete uncorrelation between runtime and input file size. In this case, we would throw all of the data points away.

Using this method to prune out bad jobs requires examining each tool individually or, at the least, it requires writing instructions for each tool individually - instructions that the computer can follow to do the pruning. This type of analysis would lead to the best results, but at the time of this writing, it has not been completed for the Galaxy dataset.

A final method of undetected error detection that we will discuss is with the use of an isolation forest. In a regular random forest, a node in a decision tree chooses to divide data based on the attribute and split that most greatly decreases that variability of the following to datasets. In an isolation forest, the data is split based on a random selection of an attribute and split. The longer it takes to isolate a datapoint, the less likely it is an outlier. As with removing the tails of the runtime distribution, we need to choose the percentage of jobs that are bad before hand.

To remove bad jobs, we used the isolation forest with contamination=0.05. We also removed any obvious undetected errors, such as no-input-file errors, wherever we could.

#### User Selected Parameters

Before we move on to the machine learning models, we also should discuss which variables we used to train the prediction models. The GRT records all parameters passed through the command line. This presents in the dataset as a mixed bag of useful and useless attributes. Useless attributes include:

* labels (such as plot axes names)
* redundant parameters (two attributes that represent the same information)
* identification numbers (such as file ids)

Useful attributes include:

* file sizes
* file types (compressed or uncompressed)
* whether the data was pre-indexed
* analysis type selection
* other analysis parameters

Instead of hand selecting parameters for each tool, we made a filter for the computer to do the pruning. It is a simple filter that attempts to remove any labels or identification numbers that are present.

The parameters are screened in the following way:

1. Remove universally not useful parameters such as:
  + \__workflow_invocation_uuid__
  - chromInfo
  - parameters whose names end with "|\__identifier__"
2. Remove any categorical parameters whose number of unique categories exceed a threshold

With these filters, we are able to remove most of the parameters that are either identifiers or labels. Since identifiers and labels are more likely to negatively affect and add noise to the results of a machine learning model we are more concerned with removing these than removing redundant parameters. In this paper, we used a unique category threshold of 10.

There are also some important attributes, that are not immediately available in the dataset. For instance, the complexity of bwa_mem is O(reference size \* input file size), which is linearly correlated with runtime. However, this product is not a variable of the bwa_mem dataset, but can be calculated and added. Just to note, in the Galaxy dataset, if the reference genome *name* is provided then the reference genome *size* is not provided. This is because of the method in which the attributes were tracked.

Because the random forest is able to find non-linear relationships, we did not combine attributes in the preprocessing step. Combining attributes in preprocessing may make it easier for the random forest to find relationships, which would improve results. However, combining relevant parameters for each tool would require close monitoring and attention, and this is out of the scope of the project.

#### Attribute Preprocessing

There are certain types of data that machine learning models prefer. For the sci-kit learn models, it is advised that the continuous variables be normally distributed and centered about zero, and that the categorical variables be binarized.

For the continuous variables, as previously noted, we log transform with numpy.log1p if the variable is highly skewed. Then, we scale the continuous variable to the range [0,1] with sklearn.preprocessing.MinMaxScaler.

For categorical variables, we binarize them using sklearn.preprocessing.LabelBinarizer. An example of label binarization is shown below. The categorical variable "analysis_type" is expanded into four discrete variables that can take the values 0 or 1.

||x1|x2|x3|x4|
|---|---|---|---|---|
| illumina  | 1  | 0  | 0  |0|
|full   |0   | 1  | 0  |0|
|pacbio   |0   | 0  | 1  |0|
|ont2d   |0   | 0  | 0  |1|

We typically have two or three continuous variables for each tool, and about one hundred expanded categorical variables. Some tools, that accept multiple input files, such as cuffnorm, can have hundreds of continuous variables. Other tools, that do not have many options,  such as fastq groomer, may have only a handful of expanded categorical variables.

## Model Comparison

In this work, we trained popular regression models available on scikit-learn and compared their performance. We used a cross validation of three and tested the models on the dataset of each tool without removing any undedected errors and with removing undetected errors via the isolation forest with contamination=5%. Pruning the datasets with the isolation forest improved the performance of some of the regressors, but it did not improve the performance of the random forest regressor.

For most of the tools, we used the default settings provided by sklearn library: Linear Regressor, LASSO, Ridge Regressor, SVR Regressor, and SGD Regressor. The neural network (sklearn.neural_network.MLPRegressor) had two hidden layers of sizes [100,10] with the rest of the attributes set to default, and the random forest had 100 trees and a max_depth of 12.

The table below shows the r-squared score for a select number of tools, and the total mean and median over each tool with more than 100 recorded runs. The performance is marked as NaN if it's r-squared scores was below -10.0, as was often the case with the linear regressor. We also marked a score as NaN if a model took more than 5 minutes to train, as was sometimes the case with the SVR Regressor, whose complexity scales quickly with the number of datapoints in the training set. The total mean and the total median were taken from the performance of each model over all of the tools with more than 100 recorded runs.

##### model comaprison with full dataset


|                    | Random Forest | Lasso | MLPRegressor | Ridge | SGDRegressor | SVR |LinearRegression|
|--------------------|---------------|------------------|-------|--------------|-------|--------------|-----|
| bwa v 0.7.15.1 | 0.91 | -0.0004 | 0.64 | 0.55 | 0.25 | 0.26 | nan|
| bwa mem v 0.7.15.1 | 0.82 | -0.0001 | 0.71 | 0.40 | 0.17 | nan | nan|
| groomer fastq groomer v 1.1.1 | 0.99 | -0.0001 | 0.97 | 0.51 | 0.28 | 0.45 | nan|
| wrapper megablast wrapper v 1.2.0 | 0.78 | -0.001 | 0.30 | 0.29 | 0.20 | 0.22 | nan|
| total mean | 0.60 | -0.01 | 0.27 | 0.32 | 0.10 | 0.14 | -0.26 |
| total median | 0.71 | -0.002 | 0.27 | 0.33 | 0.06 | 0.11 | 0.18 |


The Random Forest outperforms all of the other regressors followed by the neural network (MLPRegressor). The other regressors are not able to handle the high dimensional, and mixed continuous/categorical, dataset. We have seen in the section [Undetected Errors](#undetected-errors) that a linear relationship was expected when all of the parameters are frozen except for file size. If it was the case that most of the paremeters except for one or two were frozen, the other regressors, like the linear regressor, would be more useful. However, those regressors that are unable to group the jobs into similar parameters, the way the Random Forest is able to do, and is indeed designed to do, do not hold up in the high dimensional space.

Pruning out outliers with contamination=0.05 affected the predictions as follows.

##### model comaprison pruning of 5% of the datasets with isolation forest

|                    | Random Forest | Lasso | MLPRegressor | Ridge | SGDRegressor | SVR |LinearRegression|
|--------------------|---------------|------------------|-------|--------------|-------|--------------|-----|
| bwa v 0.7.15.1 | 0.89 | -0.0001 | 0.78 | 0.67 | 0.53 | nan | nan|
| bwa mem v 0.7.15.1 | 0.78 | -0.00002 | 0.70 | 0.51 | 0.37 | nan | nan|
| groomer fastq groomer v 1.1.1 | 0.99 | -0.0001 | 0.99 | 0.64 | 0.63 | nan | nan|
| wrapper megablast wrapper v 1.2.0 | 0.78 | -0.002 | 0.48 | 0.44 | 0.30 | nan | nan|
| total mean | 0.54 | -0.01 | 0.28 | 0.40 | 0.18 | nan | 0.25 |
| total median | 0.66 | -0.002 | 0.32 | 0.43 | 0.16 | nan | 0.35 |


 SVR Regression was not performed in these tests because of the length of time required to comlete them. Pruning the outliers with the isolation forest improved the performance of the MLPRegressor, the Ridge Regressor, SGD Regressor, and the Linear Regressor. Surpisingly, it did not improve the performance of the Random Forest Regressor or LASSO. Because of this, we did not continue to used the pruned datset for the remainder of the tests.


The full results can be viewed [here](benchmarks/comparison_benchmarks.csv). It includes the time (in seconds) to train the model. And the results for the dataset pruned with the isolation forest can be found [here](benchmarks/comparison_benchmarks_minus_outliers.csv)



## Estimating a Range of Runtimes

The random forest gave us the best results for estimating the runtime of a job. It would be more useful, though, to estimate a range of runtimes that the job would take to complete. This way, when using the model for choosing walltimes, we lower the risk of ending jobs prematurely.

A [quantile regression forest](https://doi.org/10.1.1.170.5707) can be used for such a purpose. A quantile forest works similarly to a normal random forest, except that at the leaves of its trees, the quantile random forest not only stores the means of the variable to predict, but all of the values found in the training set. By doing this, it allows us to determine a confidence interval for the runtime of a job based on similar jobs that have been run in the past.

Storing the entire dataset in the leaves of every tree is computationally costly. An alternative method is to store the means and the standard deviations. Doing so reduces the accuracy of the time ranges, but saves a lot of space. We used the modified version of the quantile regression forest that is described in [Hutter et al.](https://doi.org/10.1016/j.artint.2013.10.003) that uses standard deviation instead of quantiles.

We tested the quantile regression forest against the historical data with three fold validation on the full dataset. A prediction that is within one, two, or three standard deviations of the actual value was counted as an accurate prediction. The results can be viewed [here](benchmarks/quantile_forest_metrics.csv), and a summary is also shown below.

##### Mean accuracy of 3-fold cross-validated tests

|                    | accuracy 1std | accuracy 2std | accuracy 3std | mean interval (1std)| mean_interval (2std) | mean_interval (3std)
|--------------------|---------------|---------------|---------------|---------------|-----------------|
| bwa v 0.7.15.1 | 0.75 | 0.94 | 0.98 | 566.07 | 3054.79 | 12527.35|
| bwa mem v 0.7.15.1 | 0.80 | 0.95 | 0.98 | 286.21 | 2263.54 | 22946.34|
| groomer fastq groomer v 1.1.1 | 0.79 | 0.94 | 0.98 | 54.21 | 223.82 | 534.01|
| megablast v 1.2.0 | 0.69 | 0.91 | 0.97 | 5287.06 | 34613.03 | 182434.66|
| total mean | 0.69 | 0.89 | 0.94 | 333.49 | 4352.68 | |
| total median | 0.69 | 0.90 | 0.95 | 21.74 | 117.49 | |


The largest drawback of the quantile regression forest is that the time ranges that it guesses can be quite large. These large time ranges are not useful for giving a user an idea of how long an analysis will take, but they may be useful for creating walltimes.

For this test, we were log transforming the runtimes, which had given us better results for the regressions in the previous section. But by doing this, it also meant that the standard deviations calculated by the quantile regressor were also on a log scale, and ballooned on the tail end. So, using a confidence interval of two standard deviations about the mean is more than twice as large as using just one standard deviation about the mean.

By not log transforming the the runtimes we improve the interval sizes and the performance of the quantile regressor.

##### Mean accuracy of 3-fold cross-validated tests (runtimes not log-transformed)

|                    | accuracy 1std | accuracy 2std | accuracy 3std | mean interval (1std)| mean_interval (2std) | mean_interval (3std)
|--------------------|---------------|---------------|---------------|---------------|-----------------|
| bwa v 0.7.15.1 | 0.90 | 0.97 | 0.99 | 2196.55 | 4393.11 | 6589.66|
| bwa mem v 0.7.15.1 | 0.94 | 0.98 | 0.99 | 715.16 | 1430.32 | 2145.47|
| groomer fastq groomer v 1.1.1 | 0.84 | 0.96 | 0.98 | 631.85 | 1263.70 | 1895.55|
| wrapper megablast wrapper v 1.2.0 | 0.80 | 0.94 | 0.97 | 14722.29 | 29444.58 | 44166.88|
| total mean  | 0.69 | 0.89 | 0.94 | 326.17 | 74.50|
| total median |0.69 | 0.90 | 0.95 | 22.23 | 3.56 |

The confidence intervals for one standard deviation are larger than the one's found previously, which accounts for the better accuracy


<!-- ![alt text](plots_w_error_dump/timesplit/devteam_bwa_bwa_mem_0.7.15.1.png) -->

## Using a random forest classifier

The main advantage of a random forest classifier over a quantile regression forest, is that you can select the sizes of the runtime bins. Where in the quantile regression forest, you are at the mercy of the bins dynamically selected by the model.

In our tests, we chose buckets in the following fashion.
1. We ordered the jobs by length of runtime.
2. Then, we made (0, median] our first bucket.
3. We recursively took the median of the remaining runtimes until the number of remaining runtimes was less than 100.

|------------50%-----------|-----25%-----|--12.5%--|--12.5%--|      < ---- division of jobs

0-------------------------50s----------612s------4002s-----10530s   < ---- buckets

For example, for bwa_mem the buckets we found (in seconds) were

[0, 155.0, 592.5, 1616.0, 3975.0, 7024.5, 10523.0, 60409.0]

As the buckets become larger. the number of jobs in the bucket decrease by 1/2^i where i is the bucket. For bwa_mem, the last bucket, where i=7, holds only 0.78% of the jobs. Dividing it further would make it so the next bucket created has less than 100 jobs, which we chose as the threshold.

This method of creating buckets puts an arbitrary limit on the longest amount of time that a tool is allowed to run based on the longest a job has been observed to run. It is also susceptible to false positives at the longer runtime buckets if trained by a contaminated dataset.

The aribitrary upper limit would also be present in the original qunatile random forest since it won't create quantiles intervals longer than the longest runtime it has seen. Similarly, with the altered quantile forest with the standard deviations would not have

The results of the classifier can be found [here](benchmarks/classifier_forest_metrics.csv).

A comparison of the accuracy of the classifier vs the quantile regression forest with an interval of one standard deviation can be found below.

|                    | clf accuracy | clf mean interval | clf median interval | qf accuracy | qf mean interval | qf median interval |
|--------------------|--------------|-------------------|---------------------|-------------|------------------|--------------------|
| bwa v 0.7.15.1     | 0.81 | 698.08 | 197.00| 0.90 | 2196.55 | 651.40|
| bwa mem v 0.7.15.1 |  0.76 | 247.83 | 47.00 | 0.94 | 715.16 | 413.40|
| fastq groomer v 1.1.1 |0.93 | 1250.62 | 1010.50 | 0.84 | 631.85 | 130.41|
| megablast v 1.2.0  | 0.72 | 3821.06 | 1352.00 | 0.80 | 14722.29 | 4687.70|
| total mean  | 0.75 | 1343.36 | 1073.15 | 0.69 | 326.17 | 74.50|
| total median |0.75 | 112.32 | 39.50| 0.69 | 22.23 | 3.56|




## Conclusion

In this paper, we introduced the Galaxy dataset, tested popular machine learning models in predicting the runtime of the tools, and compared two methods of predicting runtime intervals: quantile regression forests and random forest classifiers.

Quantile regression forest gives more confidence and accuracy in the runtime intervals predicted. Random forest classifiers give more control over the size of the prediction intervals.

## Future Work

Recently, we have set up the GRT to track additional job attributes: amount of memory used (rather than just memory allocated, which is currently tracked), server load at create time, and CPU time. Once enough data is collected, we want to create models to predict memory usage and CPU time and evaluate their performance.

We also want to find the effect of processor count on runtime. Currently, every job is allotted 32 processor cores, so we do not have the data to investigate the relationship between number of processors and runtime. In the future, we plan to add random variability to the number of processor cores allotted, so that we can see how great of an effect parallelism has on these bioinformatic algorithms.

## References

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
