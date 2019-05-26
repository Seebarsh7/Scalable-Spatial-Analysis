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
* Dynamic Time Wrapping (DTW)   
It is named as 'dynamic' because it is using dynamic programming. 
