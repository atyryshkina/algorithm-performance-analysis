# The Analysis of Data Collected by the Galaxy Project

<br>

Who is my audience?

My past self?
The Galaxy Team?
Anton?
Admins?

## Intro

The public server of the Galaxy Project, www.usegalaxy.org, has been collecting job run data of bioinformatic algorithms since 2013. Since then, the number of jobs run instances (with their corresponding data) collected has exceeded well over 4 million. We hope to use this large collection to determine more efficient ways to allocate the server resources.

## Table of Contents

1. What is the Galaxy Project
2. Description of Data Collected
3. Limitations (or problems) of the Data
4. Possible schemes for outlier detection
5. Machine Learning Models
5. feature selection
6. random forests
7. classification results
8. future work


## What is the Galaxy Project

To people who are unfamiliar with the Galaxy Project, it is easier to explain why the Galaxy Project is than to explain what the Galaxy Project is.

The Galaxy Project exists to help researchers run popular bioinformatics algorithms quickly, easily, and reproducibely. In the past, if researchers wanted to try a popular algorithm, they would have to download, configure, and troubleshoot the algorithm software on their own machines. This can be a difficult and time consuming task.

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

One hurdle the dataset presents is that it contains what I call silent errors. By that, I mean, an error that occured was not recorded.

A job can be considered to have experienced a silent error if it finishes in an unreasonably short time (such an alignment job that finishes in 6 seconds), of if it finishes in an unreasonably long time (such as a )
