# The Analysis of Data Collected by the Galaxy Project

## Abstract
The public server of the Galaxy Project ( http://usegalaxy.org ) has been collecting extensive job run data on all analyses since 2013. This large collection of jobs run instances (with their corresponding attributes) can be leveraged to determine more efficient ways for allocation of Galaxy server resources. In addition, these data represent the largest, most comprehensive dataset available to date on the runtime dynamics for some of the most popular biological data analysis software. In this work we were aiming at creating a model for runtime prediction of complex algorithms trained on real data. In this paper we will:

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

The Galaxy Project is a platform that allows researchers to run popular bioinformatics analyses quickly and easily. In the past, trying to run a popular analysis would require downloading, configuring, and troubleshooting the analysys software on one's own machines. This can be a difficult and time consuming task.

With the Galaxy Project, researchers run analyses on the public Galaxy server. To do so, the user needs only to connect to www.usegalaxy.org, upload their data, choose the analysis and the analysis parameters (if any), and hit run. The output can be viewed once it is finished.

For more information visit www.galaxyproject.org.

### scikit-learn and machine learning

scikit-learn is a library of machine learning tools for Python. It has classes for anything machine learning related - from data prepocessing to regression and classification to model evaluation. The sci-kit learn library is the main library we used in our tests, specifically the regression and classification classes.

In this paper our main tool for regression and classification was the random forest, so we will briefly go over random forests.

A random forest is a collection of decision trees, and a desion tree is a series of questions asked about an object. At the end of the questions, a previously unkown attribute of the object is guessed. An example of a decision tree is shown below. In this case, the decision tree tries predict how long a job is going to take.

![alt text](images/simple_decision_tree.png)

The decision tree learns what questions to ask by training on a training set of previous jobs. At each node, it looks at a subset of attributes and chooses to split the data in the way that most minimizes variability in the subsequent two nodes. In this way, it sorts objects by similarity of the dependent and independent variables. In the figure below, a decision tree is training on a set of 100 jobs. All 100 jobs start at the root node, and they are split up as they travel down the tree. Once the tree is trained, it can then be used to predict the runtime of previously unseen jobs.

![alt text](images/decision_tree_vertical.png)

A random forest is a collection of decision trees, each of which are trained with a unique random seed. The random seed determines which sub-sample of the data each decision tree is trained and which sub-sample of attributes each tree uses and which splits each tree attempts. By implementing these constraints, the random forest protects itself from overfitting - a problem to which decision trees are susceptible.

Incidently, the decision tree also offers a way to see which independent attributes have the greatest effect on the dependent attribute. The more often a decision tree uses an attibute to split a node, the larger it's implied effect on the dependent attribute. The scikit-learn Random Forest classes have an easy way of getting this information with the feature_importances_ class attribute.

### Previous work on runtime prediction of programs

The prediction of runtimes of complex algorithms with machine learning approaches has been tackled before. [[1]](https://doi.org/10.1007/11508380_24)[[2]](https://doi.org/10.1109/CCGRID.2009.58)[[3]](https://doi.org/10.1145/1551609.1551632)[[4]](https://doi.org/10.1109/CCGRID.2009.77)[[5]](https://doi.org/10.1007/11889205_17)

In some works, new machine learning methods are designed specifically for the problem. In 2008, [Gupta et al.](http://doi.org/10.1109/ICAC.2008.12) proposed a tool called a PQR (Predicting Query Runtime) Tree to classify the runtime of queries that users place on a server. The PQR tree dynamically choose runtime bins during training that would be appropriate for a set of historical query runtimes. The paper notes that the PQR Tree outperforms the decision tree.

Most previous works tweak and tailor old machine learning methods to the problem. For instance, in 2010, [Matsunaga](http://doi.org/10.1109/CCGRID.2010.98) enhances on the PQR Tree by adding linear regressors at its leaves, naming it PQR2. They test their model against two bioinformatic analyses tools: BLAST (a local alignment algorithm) and RAxML (a phylogenetic tree constructer). The downside of PQR2 is that there are not readily available libraries of the model in Python, R.

The most comprehensive survey of runtime prediction models was done by [Hutter et al.](https://doi.org/10.1016/j.artint.2013.10.003) In 2014. In the paper, they compared 11 regressors including ridge regression, neural networks, Gaussian process regression, and random forests. They did not include PQR tree in their evaluations. They found that the random forest outperforms the other regressors in nearly all assessments and is able to handle high dimensional data without the need of feature selection.

In our paper, we verify that random forests are the best model for the regression, discuss the merits of quantile regression forests, and present a practical approach for determining an appropriate walltime with the use of a classifier.

[

this part in brackets are just my notes

PQR: Predicting Query Execution Times for Autonomous Workload Management (2008) [Gupta et al.](http://doi.org/10.1109/ICAC.2008.12) -> PQR trees are like decision trees, but the categories are chosen dynamically and each node has a different classifier.

On the use of machine learning to predict the time and resources consumed by applications (2010) [Matsunaga et al.](http://doi.org/10.1109/CCGRID.2010.98) -> BLAST (local alignment algorithm) and RAXML PQR2

Algorithm Runtime Prediction: Methods & Evaluation (2014) -> [Hutter et al.](https://doi.org/10.1016/j.artint.2013.10.003)

A Method for Estimating the Execution Time of a Parallel Task on a Grid Node (2005) -> [Phinjaroenphan](https://doi.org/10.1007/11508380_24) uses k-nearest neighbors

A Hybrid Intelligent Method for Performance Modeling and Prediction of Workflow Activities in Grids (2009) -> [Duan](https://doi.org/10.1109/CCGRID.2009.58) a bayesian neural network

Trace-Based Evaluation of Job Runtime and Queue Wait Time Predictions in Grids (2009) -> [Sonmez et al.](https://doi.org/10.1145/1551609.1551632)
classified jobs into where they run, who is running it, and the size of the job then took a running average of the runtime of previous jobs in this category tof make a prediction

A Survey of Online Failure Prediction Methods (2010) -> [SALFNER et al.](https://doi.org/10.1145/1670679.1670680) looks at markers such as memory usage, function called, runtime, to monitor programs being run on the server in real time. --- many different methods are described

]

## Overview of Data

All of the tools found on usegalaxy.org are tracked. The data is collected using the Galactic Radio Telescope (GRT), which records a comprehensive set of job run attributes.

This includes:

* create time
* runtime
* user selected parameters (tool specific)
* state ("ok", "error", or other)
* input data
  - size of input data
  - filetype of input data (.fastq, .fastq.gz, ...)
* hardware info
  - memory allocated
  - swap memory allocated
  - number of processors allotted
  - destination id (which node the job was run on)
* user info
    - user id
    - session id
    - history id


Our dataset contains runtime data for 1372 different tools that were run on the Galaxy Servers over the past five years. A statistcal summary of those tools, ordered by most popular, can be found [here]().

The computer clusters of the Galaxy Server have different hardware specifications. The hardware which tools run on is recorded in the dataset. [I need to put hardware specs here.]

The jobs have amount of memory alotted to them, but they do not have dedicated processors. The processors are shared with other jobs running on the node. The published dataset does not contain server load information or the total amount of memory used for a job. We began tracking those two attributes not long before writing this, and will publish those datasets when available.


#### Distribution of the Data

Typically, machine learning algorithms, such as, random forests and neural networks prefer to use data with a normal distribution. The distribution of runtimes and filesizes in the Galaxy dataset are highly skewed. The distribution for a tool called BWA (version 0.7.15.1) can be seen below.

![alt text](images/runtimes3.png)

![alt text](images/filesize_bwamem.png)

In this project, we address this skewness by doing a log transform on the data.
We use numpy's log transformer numpy.log1p which transforms the data by log(1+x)

![alt text](images/log_runtimes3.png)

![alt text](images/log_filesize3.png)

This transformation works for most of the runtime and intput file size attributes to give a more balanced distribution.

#### Undetected Errors

One hurdle the dataset presents is that it contains undedected errors - errors that occured but were not recorded.

For example, some tools require that an input file be provided. Bwa mem is one such tool. If an input file is not provided, bwa mem should not run at all or else the run should result in an error. In spite of this, the number of bwa mem v. 0.7.15.1 jobs in the dataset that ran succesfuly and without an input file is 49 or 0.25% of "successful" jobs. These jobs should not have run at all, and yet they are present in the dataset and marked as succesfuly completed.

Whether the errors were be caused by bugs in the tool code, malfunctions in the server, mistakes in record keeping, or a combination of these, the presence of these of errors is troubling. With undetected input file errors, it is trivial to identify and remove the culprits from the dataset. However, these errors call into question the validity of the rest of the data. If there are many other jobs similarly mislabelled as "sucessfully completed" that are not as easily identified as input file errors, and these mislabelled jobs are used to train a machine learning model, they could skew the predictions.

Another method of screening the dataset for undetected errors is by looking for jobs that ran faster than possible and jobs that ran slower than probable. A job that finishes in an unreasonably short time (such an alignment job that finishes in 6 seconds), or a job that finishes in an unreasonably long time (such as a ??). However, indentifying these errors requires the trained eye of someone who is both familiar with the tools and has ample time to look through the dataset.

Using this hueristic, we can account for undetected errors by getting rid of the jobs that took the longest and the shortest amount of time to complete.

![alt text](images/gaus_dist2.png)

This requires choosing quantiles of contamination for each tool. In the figure above the quantiles used are 2.5%. For bwa mem (v. 0.7.15.1) - an alignment algorithm - 8.1% of jobs in the collected data took 6 seconds or less to finish. Are all of these jobs undedected errors? If we increase the unreasonable runtime threshhold to 9 seconds, we see that 17.1% of jobs experienced undedected errors. It is difficult, even for a human, to decide if these recordings are reasonable job runtimes.

Since we know that the two variables that have the greatest affect on the runtime of bwa mem are input file size and reference file size. The larger the file sizes, the longer it would take for the job to run. We should be considering these variables when looking for undetected errors. One method of doing this is by freezing all of the other variables and only looking at the relationship between these input file sizes and runtime.

In the following figures all of the user selected parameters are frozen except for input file size. We were able to freeze the reference file size because many reference genomes, such as the human genome, are popular and commonly used.

![alt text](images/hg19.png)

The refernece file, hg19 is the human genome

![alt text](images/hg38patch.png)

The reference file, hg38 is another version of the human genome.

The first graph shows a strong correlation between input file and runtime. This is the correlation we expect. The outliers that we remove are the datapoints in the bottom right corner. We can do this safely because, while it is possible for a job to run longer than the correlation displayed on the graph, it is impossible for jobs to run faster.

The second graph, displays complete uncorrelation between runtime and input file size. In this case, we would throw all of the datapoints away.

Using this method to prune out bad jobs requires examining each tool individually or, at the least, it requires writing instructions for each tool individually - instructions that the computer can follow to do the pruning. This type of analysis would lead to the best results, but at the time of this writing, it has not been completed for the Galaxy dataset.

A final method of undetected error detection that we will discuss is with the use of an isolation forest. In a regular random forest, a node in a decision tree chooses to divide data based on the attribute and split that most greatly decreases that variability of the following to datasets. In an isolation forest, the data is split based on a random selection of an attribute and split. The longer it takes to isolate a datapoint, the less likely it is an outlier. As with removing the tails of the runtime distribution, we need to choose the percentage of jobs that are bad before hand.

To remove bad jobs, we used the isolation forest. We also removed any obvious undetected errors, such as no-input-file errors, wherever we could.

#### user selected parameters

Before we move on to the machine learning models, we also should discuss which variables we used to train the prediction models. The GRT records all parameters passed through the command line to the tool that runs it. This presents in the dataset as a mixed bag of useful and useless attributes. Some of the parameters are very important, such as the reference genome size, and some are not important at all. Unimportant parameters include:

* labels (such as plot axes names)
* redundenct parameters (two attributes that represent the same information)
* identification numbers (such as file ids)

There are also some important attributes, that are not immediately available in the dataset. For instance, the complexity of bwa mem is O(reference size \* input file size), so this is a very important attribute. However, this product is not a variable of the bwa mem dataset, but can be calculated and added. Just to note, in the Galaxy dataset, if the reference genome *name* is provided then the reference genome *size* is not provided. This is because the method in which the attributes were tracked.

The parameters are screened for usefulness in the following way:

1. Remove universally unuseful parameters
  - \__workflow_invocation_uuid__
  - chromInfo
  - parameters whose names end with "|\__identifier__"
2. Remove any categorical parameters whose number of unique categories exceed a threshhold

With these filters, we are able to remove most of the parameters that are either identifiers or labels. Since identifiers and labels are more likely to negatively affect and add noise to the results of a machine learning model we are more concerned with removing these than removing reduntant parameters.

#### attribute preprocessing

There are cetain types of data the machine learning models prefer. For the sci-kit learn models, it is advised that the continuous variables be normally distributed and centered about zero, and that the categorical variables be binarized.

For the continuous variables, as previously noted, we log transform with numpy.log1p if the variable is highly skewed. Then, we scale the continuous variable to the range [0,1] with sklearn.preprocessing.MinMaxScaler.

For categorical variables, we binarize them using sklearn.preprocessing.LabelBinarizer. An example of label binarization is shown below. The categorical variable "analysis_type" is expanded into four discrete variables that can take the values 0 or 1.

||x1|x2|x3|x4|
|---|---|---|---|---|
| illumina  | 1  | 0  | 0  |0|
|full   |0   | 1  | 0  |0|
|pacbio   |0   | 0  | 1  |0|
|ont2d   |0   | 0  | 0  |1|

We typically have two or three continuous variables for every tool, and about one hundred expanded categorical variables. Although some tools, that accept multiple input files, such as cuffnorm, can have hundreds of continuous variables. Other tools, that do not have many options, may have only a handful of expanded categorical variables, such as fastq groomer.

## Model Comparison

In this work, we trained popular regression models available on scikit-learn, and compared their performance. Our results agreed with Hutter et al. that the Random Forest is the best predictor for of the runtime of complex algorithms. We used a cross validation of three and tested the models on the dataset of each tool without removing any undedected errors.

A snapshot of the results can be viewed [here](). We used a correlation metric r-squared to compare the the models. We tested it on historical data with undetected errors present and unpruned and on historical data with 5% of the jobs pruned by an isolation forest. Pruning the datasets improves the predictions of the model.

||unpruned|pruned|
|---|---|---|
|mean r-squared over all tools|-18.5|0.51|

## Estimating a Range of Runtimes

The random forest gave us the best results for estimating the runtime of a job. It would be more useful, though, to estimate a range of runtimes that the job would take to complete. This way, when using the model for choosing walltimes, we lower the risk of ending jobs prematurely.

A [quantile regression forest](https://doi.org/10.1.1.170.5707) can be used for such a purpose. A quantile forest works similarly to a normal random forest, except that at the leaves of its trees, the quantile random forest not only stores the means of the variable to predict, but all of the values found in the training set. By doing this, the quantile random forest allows one to determine a confidence interval for the runtime of a job based on similar jobs that have been run in the past.

Storing the entire dataset in the leaves of every tree is computationally costly. An alternative method is to store the means and the standard deviations. Doing so reduces the accuracy of the time ranges, but saves a lot of space.

We tested the quantile regression forest against the historical data with five fold validation. We tested it on historical data with undetected errors unpruned and on historical data with 5% of the jobs pruned by an isolation forest.

The results can be viewed [here]().

||unpruned|pruned|
|---|---|---|
|mean accuracy of quantile forest for all tools|0.59|0.69|

The largest drawback of the quantile regression forest is that the time ranges that it guesses are very large. These large time ranges are not useful for runtime estimates for users, but they are useful for creating walltimes. Because of the skewed nature of the runtime distribution, the quantile random forest tends to underestimate rather than overestimate, which is problematic. In addition, if there are bad jobs present in the training dataset, it would also mess up the model predictions.

## Using a random forest classifier

## Future Work

Recently, we have set up the GRT to track additional job attributes: amount of memory used (rather than just memory allocated, which is currently tracked), server load at create time, and CPU time. Once enough data is collected, we will create models to predict memory usage and CPU time and evaluate their performance.

We also want to model the relationship between processor count and runtime. Currently, every job is allotted 32 processor cores, so we do not have the data to investigate the relationship between number of processors and runtime. In the future, we plan to add random variability to the number of processor cores allotted, so that we can see how great of an effect parallelability has on these bioinformatic algorithms.

## References

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
