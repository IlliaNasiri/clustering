# Implementation and Comparison of KMeans, KMneas++, DBSCAN

## Contents:
1. [Motivation](#motivation) 
2. [Installation](#how-to-install) <br>
3. [How to use?](#how-to-use) <br>
- 3.1. [DBSCAN](#DBSCAN)
- 3.2. [K-Means](#K-Means)
- 3.3. [K-Means++](#K-MeansPlus)

## Motivation

### <a name="motivation"> </a>

My motivation behind this project was to deepen my knowledge of clustering algorithms, which I first learnt in a university data mining course. We were assigned to implement the K-means and K-means++ algorithm. I really enjoyed this assignment, so I wanted to extend it further by also implementing the DBSCAN algorithm. 
Having done this project, my understanding of clusterings has definitely improved. I really started to see the strengths and weaknesses of each of the algorithms. In addition, I got more hands-on expereince with Numpy, as I avoided using Python for/while loops as much as possible to improve the effciency and runtime of the code. 

## How to install 
Assuming you have **pip**:
```console
pip install clustering-project
```
### <a name="how-to-install"> </a>

## How to use?

### <a name="how-to-use"> </a>

### Importing the library:

```python
from clustering import DBSCAN, KMeans
```

### DBSCAN:

### <a name="DBSCAN"> </a>

DBSCAN has two hyper-parameters: 
1. epsilon: the radius around every data-point used for constructing core-points
2. min_sample: the minimum number of points that have to fall within epsilon for a point to be considered a core point

Here is an example of how to use DBSCAN:
```python
from clustering import DBSCAN

epsilon = 0.5
min_samples = 5

X = # YOUR DATA HERE

dbscan = DBSCAN(epsilon, min_sample)
clustering = dbscan.fit(X)
```

### K-means:

### <a name="K-Means"> </a>

K-means has 3 parameters:
1. n_clusters: number of clusters created by k-means
2. init: (default: "random")
  <br>"random": corresponding to random initialization of cluster representatives. <br>
   "kmeans++": corresponding to a more sophisticated initialization of cluster representatives.
4. max_iter: (default: 300) the maximum number of updates of centroids that K-means will perform (unless it converges and stops earlier)

Here is an example of how to use KMeans:
```python
from clustering import KMeans

n_clusters = 2

X = # YOUR DATA HERE

kmeans = KMeans(n_clusters)
clustering = kmeans.fit(X)
```

### <a name="K-MeansPlus"> </a>

Here is an example of how to use KMeans++:
```python
from clustering import KMeans

n_clusters = 2
init = "kmeans++"

X = # YOUR DATA HERE

kmeans = KMeans(n_clusters, init)
clustering = kmeans.fit(X)
```

## Demonstration of strengths and weaknesses of the implemented algorithms
