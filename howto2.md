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
* What is the Galaxy Project
* Description of Data Collected
* Limitations (or problems) of the Data
* Possible schemes for outlier detection
* Machine Learning Models
* feature selection
* random forests
* classification results
* future work

### Background: What is the Galaxy Project

The Galaxy Project is a platform that allows researchers to run popular bioinformatics analyses quickly and easily. In the past, trying to run a popular analysis would require downloading, configuring, and troubleshooting the analysys software on one's own machines. This can be a difficult and time consuming task.

With the Galaxy Project, researchers run analyses on the public Galaxy server. To do so, the user needs only to connect to www.usegalaxy.org, upload their data, choose the analysis and the analysis parameters (if any), and hit run. The output of can be viewed once it is finished.

For more information visit www.galaxyproject.org.

### Background: scikit-learn and Machine Learning

scikit-learn is a library of machine learning tools for Python. It has classes for anything machine learning related - from data prepocessing to regression and classification to model evaluation. The sci-kit learn library is the main library we used in our tests - specifically the regression and classification classes.

In this paper our main tool for regression and classification was the random forest, so we will briefly go over what a random forest is.

A random forest is a collection of decision trees, and a desion tree is a series of questions asked about an object. At the end of the questions, a previously unkown attribute of the object is guessed. An example of a decision tree is shown below.

![alt text](images/simple_decision_tree.png)

In this case, the decision tree tries predict how long a job is going to take. The decision tree learns what questions to ask by training on a training set of previous jobs. At each node, it looks at a subset of attributes and chooses to split the data in the way that most minimizes variability in the subsequent two nodes. In this way, it sorts objects by similarity of the dependent and independent variables.

![alt text](images/decision_tree_vertical.png)

A random forest is a collection of decision trees, each of which are trained with a unique random seed. The random seed determines which sub-sample of the data each decision tree is trained and which sub-sample of attributes each tree uses. By implementing these constraints, the random forest protects itself from overfitting, which decision trees are susceptible to.

Incidently, the decision tree also offers a way to see which independent attributes have the greatest effect on the dependent attribute. The more often a decision tree uses an attibute to split a node, the larger it's implied effect on the dependent attribute. The scikit-learn Random Forest classes have an easy way of getting this information with the feature_importances_ class attribute.

### Background: Previous work on runtime prediction of programs

[
this part in brackets are just my notes
PQR: Predicting Query Execution Times for Autonomous Workload Management (2008) [Gupta et al.](10.1109/ICAC.2008.12) -> PQR trees are like decision trees, but the categories are chosen dynamically and each node has a different classifier.

On the use of machine learning to predict the time and resources consumed by applications (2010) [Matsunaga et al.](10.1109/CCGRID.2010.98) -> BLAST (local alignment algorithm) and RAXML PQR2

Algorithm Runtime Prediction: Methods & Evaluation (2014) -> [Hutter et al.](https://doi.org/10.1016/j.artint.2013.10.003)
]

Previous work on runtime prediction of complex algorithms have seen a wide range of approaches.

In 2008, Gupta proposes a tool called a PQR (Predicting Query Runtime) Tree to classify the runtime of queries that users place on a server. In the same paper, Gupta presents a way to dynamically choose runtime bins that would be appropriate for a set of historical query runtimes. [an explanation of how they do this] The paper notes that the PQR Tree outperforms the decision tree, and that the performance of both trees imporoves with the use of dynamically chosen bins over the use of a-priori chosen bins. However, the results were not compared to the performance of a Random Forest.

In 2010, Matsunaga enhances on the PQR Tree by adding linear regressors at its leaves, naming it PQR2. They test their model against two bioinformatic analyses tools: BLAST (a local alignment algorithm) and RAxML (a phylogenetic tree constructer). They used mean performance error (MPE) as a metric for their results. The improvements found are shown in the table below.

#### MPE of PQR vs PQR2

|TOOL|PQR|PQR2|
|---|---|---|
|BLAST|8.81%|8.76%|
|RAxML|40.82%|35.30%|

Previous work done on predicting the runtime of programs mostly focused on predicting the runtimes of short programs such as SQL queries - these are on the order of milliseconds. Recently, there has been some work done on predicting the runtimes of longer, more complex programs, such as (that paper about bio tools) and (the np paper).

This is the first time that such a large dataset has been available to attempt to create a runtime prediction model trained on real data. We verify that Random Forests are the best model for the regression, and present a practical approach for determining an appropriate walltime, which is with the use of a classyfier.

## Overview of Data

### Description of Data Collected

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

Description of bioinformatics algorithms. They are typically run on a large strands of dna... the human genome is 4 giga bytes.. some take long.. some are fast.. bioinfomatics is a growing field... provide a number on how much traffic the galaxy project gets a day and how this has increased over the past

### Undetected Errors

One hurdle the dataset presents is that it contains undedected errors - errors that occured but were not recorded.

A job can be considered to have experienced a undedected error if it finished in an unreasonably short time (such an alignment job that finishes in 6 seconds), of if it finished in an unreasonably long time (such as a ??). Someone familiar with the algorithm can quickly infer that something went wrong in these two cases. Can we teach a computer do the same?

Undedected errors can happen for a variety of reasons. It could be that the job was given bad data that caused it do end quickly without outputing an error. Another possibility is that the algorithm gets stuck in a loop that causes it to go on for ages. Or it could be some error on the server side.

One goal of this project is to create a computer program that predicts if a job has experienced an undedected error and is taking an unreasonably long time to finish. To do that we employ a machine learning model, and train it on the collected data. If the training data contains few undedected errors, we can treat those outliers as noise, and feed the data to the model without much worry. If the training data contains many instances of undedected errors this will effect the performance and reliability of the model, so we would want to filter the bad data out before hand.

For bwa mem (v. 0.7.15.1) - an alignment algorithm - 1.7% of jobs in the collected data took 6 seconds or less to finish. Are all of these jobs undedected errors? If we increase the unreasonable runtime threshhold to 9 seconds, we see that 5.0% of jobs experienced undedected errors. It is difficult, even for a human, to decide if these recordings are reasonable job runtimes.

One way to account for undetected errors is to simply get rid of the jobs that took the longest and the shortest amount of time to complete.

Skewness of runtimes

Limitations of the Data

The GRT was set up to collect all of the important data for a job run. However, as with all data collection, there were some unforseeable problems with the data collected.

Future attributes to track

* memory
* server load
* add noise to processors

messiness of user selected parameters

* labels (such as plot axes names)
* redundency (two attributes that represent the same information)
* lists
* ids


## Future Work

Recently, we have set up the GRT to track additional job attributes: amount of memory used (rather than just memory allocated, which is currently tracked), server load at create time, and CPU time. Once enough data is collected, we will create models to predict memory usage and CPU time and evaluate their performance.

We also want to model the relationship between processor count and runtime. Currently, every job is allotted 32 processor cores, so we do not have the data to investigate the relationship between number of processors and runtime. In the future, we plan to add random variability to the number of processor cores allotted, so that we can see how great of an effect parallelability has on these bioinformatic algorithms.

## References

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
