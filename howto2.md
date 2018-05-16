# The Analysis of Data Collected by the Galaxy Project

<br>

Who is my audience?

My past self?
The Galaxy Team?
Anton?
Admins?

<br>

The public server of the Galaxy Project, www.usegalaxy.org, has been collecting data on job runs since 2013 (November 9). Since then, the number of jobs runs collected has exceeded well over 4 million job runs. This large collection of

<br>

Outline

1. What is the Galaxy Project
2. The Type of Data Collected
3. Problems with the Data
4. Possible schemes for outlier detection
5. feature selection
6. random forests
7. classification results
8. future work


<br>


It is easier to explain why the Galaxy Project is than to explain what the Galaxy Project is.

The Galaxy Project exists to help researchers run popular bioinformatics algorithms quickly, easily, and reproducibely. In the past, researchers had to either
  1. write their own algorithms or
  2. download and configure algorithm software on their own machines.

This can be a difficult and time consuming task.

With the Galaxy Project, researchers run the algorithms on the public Galaxy server. To do so, the user needs to connect to www.usegalaxy.org, upload any data, choose the algorithm and the algorithm parameters (if any), and hit run. The output of the algorithm can be viewed once it is finished.

The first public instance of the Galaxy Project www.usegalaxy.org launched in 2006. They have been collecting data about job run info since 2013 (November 9). Since then, they have collected a dataset of over 4 million job runs.
