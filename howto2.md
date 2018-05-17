# The Analysis of Data Collected by the Galaxy Project


Who is my audience?

My past self? The Galaxy Team? Anton? Admins?

## Intro
The public server of the Galaxy Project, www.usegalaxy.org, has been collecting job run data of bioinformatic algorithms since 2013. Since then, the number of jobs run instances (with their corresponding data) collected has exceeded well over 4 million. We hope to use this large collection to determine more efficient ways to allocate the Galaxy server resources.

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

## What is the Galaxy Project

To people who are unfamiliar with the Galaxy Project, it is easier to explain why the Galaxy Project is than to explain what the Galaxy Project is.

The Galaxy Project exists to help researchers run popular bioinformatics algorithms quickly, easily, and reproducibely. In the past, trying a popular algorithm would require downloading, configuring, and troubleshooting the algorithm software on their own machines. This can be a difficult and time consuming task.

With the Galaxy Project, researchers run the algorithms on the public Galaxy server. To do so, the user needs only to connect to www.usegalaxy.org, upload their data, choose the algorithm and the algorithm parameters (if any), and hit run. The output of the algorithm can be viewed once it is finished.

For more information visit www.galaxyproject.org.

## Description of Data Collected

All of the tools found on usegalaxy.org were tracked. The data was collected using the Galactic Radio Telescope (GRT), which records a comprehensive set of job run attributes.

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

## Overview of Data

Statistical Summary

A summary of the tool runtime statistics can be found at summary_statistics.csv

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

One hurdle the dataset presents is that it contains undedected errors - errors that occured but were not recorded.

A job can be considered to have experienced a undedected error if it finished in an unreasonably short time (such an alignment job that finishes in 6 seconds), of if it finished in an unreasonably long time (such as a ??). Someone familiar with the algorithm can quickly infer that something went wrong in these two cases. Can we teach computer do the same?

Undedected errors can happen for a variety of reasons. It could be that the job was given bad data that caused it do end quickly without outputing an error. Another possibility is that the algorithm gets stuck in a loop that causes it to go on for ages. Or it could be some error on the server side.

One goal of this project is to create a computer program that predicts if a job has experienced an undedected error and is taking an unreasonably long time to finish. To do that we employ a machine learning model, and train it on the collected data. If the training data contains many instances of jobs that experienced undedected errors this will effect the performance and reliability of the model.

For bwa mem (v. 0.7.15.1) - a alignment algorithm - 1.7% of jobs in the collected data took 6 seconds or less to finish. Are all of these jobs undedected errors? If we increase the unreasonablness threshhold to jobs that took less than 9 seconds to finish, we see that 5.0% of jobs experienced undedected errors. Again, we do not know if these are

Skewness of runtimes

Future Work
We have set up the GRT to track additional job attributes: amount of memory used (rather than just memory allocated, which is currently tracked), server load at create time, and CPU time. Once enough data is collected, we will create models to predict memory usage and CPU time and evaluate their performance.

We also want to model the relationship between processor count and runtime. Currently, every
