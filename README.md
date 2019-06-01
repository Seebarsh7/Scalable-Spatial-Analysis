# Scalable-Spatial-Analysis
This is the project files for class at UC Berkeley. It consists the applications of machine learning algorithms in the realm of Transportation, including Principle Component Analysis (PCA), Clustering algorithms, Ramdom Forrest and Boosting. The projects deal with several pretigious datasets like NYC Yellow Taxi datasets, Didi Trajectory Datasets, MNIST Hand-written Digits, Twitter Dataset and so on. Shapely, GeoPandas and PostgresSQL are mentioned in the projects.  
Projects are designed by fantastic faculty members at UC Berkeley Intelligent Transportation Study Center.
## Project 0: Overview of Python
* Few useful libraries.
* Interfaces of natural language processing tool class, including word frequency calculation, character frequency calculation, and other useful tools.   
  * dictionary (Hash table in Python):    
  ``` dict_alphabet = {s: 0 for s in list(string.ascii_lowercase)} ```    
  will create a dictionary    
  ```dict_alphabet = {'a': 0, 'b': 0, ..., 'z': 0}```    
* Geo-spatial tools shapely, including creating polygons, relationship like intersection and within.
  * Guide: https://shapely.readthedocs.io/en/stable/manual.html
## Project 1: Vectorized Coordinates Transformation Algorithm
* The project is transfering WGS84 to the GCJ02 (coordinates used by Chinese Government). Data source is not available in this repo.
* The main task in this homework is to transform the GCJ-2 coordinates in the DiDi dataset to WGS-84 coordinates and select the GPS points that are only in the area of interest.
* Apply function to dataframe
  * Directly call the function gcj2wgs_exact()from [evil_transform.py](https://github.com/googollee/eviltransform) and use the Pandas ```apply``` and ```lambda``` functions.
  * Useful links    
  •https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html    
  •https://stackoverflow.com/questions/47372274/apply-custom-function-in-pandas-passing-values-multiple-columns     
  •U2EF1’s  answer  at https://stackoverflow.com/questions/23690284/pandas-apply-function-that-returns-multiple-values-to-rows-in-pandas-dataframe    
  •http://pandas-docs.github.io/pandas-docs-travis/user_guide/groupby.html     
* How to improve the efficiency, indicating by the CPU time?
  * multiprocessing, that is use all the cpu for training. Sample code:
  ``` python
  #!/usr/bin/env python
  # coding: utf-8

  from multiprocessing import Pool
  import pandas as pd
  import numpy as np

  test_arr = np.random.randn(5000,2)
  test_df = pd.DataFrame(test_arr, columns = ['c1', 'c2'])

  def test_func(x, y):
      for _ in range(30):
          _a = y
          _b = (x + _a)**2
          if x >= _b:
              _a = x
          else:
              return np.sinh(x)
      return np.tanh(x)

  def manipulate_pd(df):
      df['c3'] = df.apply(lambda x: test_func(x.c1, x.c2), axis = 1)
      return df

  with Pool(4) as p:
      test_rslt = p.map(manipulate_pd, [test_df.loc[:999], 
                                        test_df.loc[1000:1999], 
                                        test_df.loc[2000:2999],
                                        test_df.loc[3000:3999],
                                        test_df.loc[4000:]])
  print(type(test_rslt))

  merged_test_rslt = pd.concat(test_rslt)
  print(merged_test_rslt.shape)
  merged_test_rslt.head(3)

 * Vectorized everything instead of using a for loop.
 * Corp the region that interested in.
## Project 2: Dataframe calculation and Eigen-decomposition
* Speed calculation:azimuth    
The azimuth is 0 to 180 if heading north, and -180 to 0 if heading south. Speed calculation is complicated in its flow control rather than the algorithm. Therefore, details would not be explained in the README file.   
To calculate the azimuth, there is a nice package in python using the following sample code:
```
global time_tuple
global time_tuple_plt
global g
g = pyproj.Geod(ellps='WGS84')
azN,azS,distance = g.inv(lng1, lat1, lng2, lat2)
```
Few useful links that motivated the calculationL
Pandas groupby module:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.htmlhttps://pandas.pydata.org/pandas-docs/version/0.22/groupby.html   
Pyproj Geod module:https://jswhit.github.io/pyproj/pyproj.Geod-class.html    
* Eigen Image Decomposition   
Eigen Image Decomposition is implemented based on famous image dataset, MNIST hand-written digit dataset.   
The  first k eigenvectors associated  with  the  fisrt k eigenvalues arethe first k principal  components that maximize the variance of the input dataset (dimension m * n) directions.In computer vision, those components are of great use for tasks such as face recognition, and computer scientists defined those principal components as eigen pictures.
The first step is to calculate the covariance matrxi with mean and transpose. Refer to the [wikipedia](https://en.wikipedia.org/wiki/Covariance_matrix). The major problem is, how to avoid the imaginary eigenvalue when dealing with such large dataset (MNIST is 28 * 28 which flattened to 784 dimensions). There are two methods that are used to stablized the calculation, that is, ensuring the matrix *P.S.D*:   
  - in python, there is a nice function in linear algebra library [eigh](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html) that could only return real value.   
  - A very simple and intuitive approach is simply to enforce A = (A.T + A)/2.
* PostgresSQL (I'll skip this part.Please look at the code3 if interested.)
## Project 3: Clustering
* [Dynamic Time Wrapping (DTW)](https://en.wikipedia.org/wiki/Dynamic_time_warping)    
In time series analysis, dynamic time warping (DTW) is one of the algorithms for measuring similarity between two temporal sequences, which may vary in speed. For instance, similarities in walking could be detected using DTW, even if one person was walking faster than the other, or if there were accelerations and decelerations during the course of an observation.    
It is named as 'dynamic' because it is using dynamic programming. Given two trajectory objects,  one has N * d dimension, and one has M * d dimension. The dynamic time wrapping could be expressed by the following formula (how to type formula in github anyways :)    
*gamma(i,j) = D(i,j) + min(gamma(i,j-1), gamma(i-1,j), gamma(i-1,j-1))*     
- D(i,j) is the pairwise distance given a sequence of trajectory, which could be calculated using 2 for loops (simple idea :)))    
```python
for i in range(x.length):
   for j in range(y.length):
      D[i][j] = numpy.linalg.norm(x[i] - y[j]) #this is the 2-norm distance
```
Therefore in the algorithm, it is considering the boundary conditions and then do the looping. It is adding the minimum value from gamma matrix to the original distance matrix.    
Being dynamic could be reflected from the *finding the minumum* part. Boundary condition needs to be considered in the looping. Please refer to Part 3 code 1 for details.    
A useful 2-d DTW example to facilitate the understanding: https://nipunbatra.github.io/blog/2014/dtw.html     
## Project 4: Travel Time Prediction - Kaggle  
### 1.	Data Cleaning   
The data contains 2012 Sept’s SF data and 2015 whole year’s NYC data. 
  #### (1)	NaN Value
  Some values are NaN in the dataset. Those are dropped.
  #### (2)	Noise Data
  -	A bunch of data all have duration 40000, which is needed to be dropped from the dataset.
  -	Some of the data shows a weird speed pattern. They are dropped from the dataset as well.
  - This is a draft of distance versus duration. Outliers could be observed. 
  -	Speed is calculated for training set to do the data cleaning. All the speed that larger than 37 are dropped from the data set.
### 2.	Exploratory data analysis
#### 1)	Distance and headings calculation
The distances are calculated as the distance between two points with the function in Assignment 2. Headings are calculated as azimuth and classified into 6 categories: N, S, NE, NW, SE, SW. They are made to dummy variables in the final model.
#### 2)	Normalization and Performance comparison
The duration is not normal distributed for both SF and NYC. This is the illustration from NYC data. I tried log the duration to make it normal distributed while the prediction results need to be logged back. The error could be created during the ‘np.exp()’ back part.
#### 3)	Clustering and Neighbors
Neighbors are found as the length of a cluster. Using kmeans, each pick up location and drop off location is assigned to the number of neighbors they have.   
#### 4)	PCA (2d to 2d) 
The PCA in this competition is not for the dimensional reduction. It is a 2D to 2D transformation. I find this way in NYC taxi data competition. It is good for tree growth and split according to the experiment.    
#### 5)	Visualization
For the visualization, we could see that the dataset size for SF is relatively small. I tried training two separate models for the two cities, but the performance is not ideal. I think it is because of the dataset size. Therefore, I add a new feature which indicates which city the point belongs to.
**Please refer to the write-up in Project 4 for more details.**
### 3.	Modelling
Because of the page limitation, I only include the model. I started from neural networks which used to perform well in my past projects. However, this is not the case for this competition as the dataset and number of features is not big enough to build a deep neural network. Therefore, I turned my way to ensemble modelling and find XGBoost has the best performance.    
#### 1)	Neural Network based on MXNet: 2-layer with one non-linear layer, bad result, around 380
#### 2)	Ensemble Method
-	Random Forrest: Not so good, around 320
-	Gradient Boosting: Not enough
-	XGBoost (chosen): 250 number of estimators and 10 max depth. Proved to be the best.
#### 3） Concat Models (this is the amazing part :))))
My best entries were 297 and 301, therefore I concat them together based on the final score. That is, the 297 model takes up for 301/598 and 301 model takes the other part. I make a new prediction based on the model. After the concatenation, the result is 294.    
   
