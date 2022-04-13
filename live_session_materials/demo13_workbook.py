# Databricks notebook source
# MAGIC %md
# MAGIC # wk11 Demo - Recommender Systems; ALS
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2018`__
# MAGIC 
# MAGIC Last week we did pagerank. This week we're doing Alternating Least Squares (ALS) Regression. 
# MAGIC 
# MAGIC In class today we'll start with a general discussion of recommender systems, then we'll look at some basic theory of ALS and how it can be prallelized in a map/reduce framework like Spark. 
# MAGIC 
# MAGIC We provide the code for a closed-form ridge regression (l2) starting with a single node implementation, distributed implementation, and a mllib implementation, for your reference.
# MAGIC 
# MAGIC By the end of today's demo you should be able to:  
# MAGIC * ... __identify__ pros and cons in various RS approaches
# MAGIC * ... __describe__ ALS regression 
# MAGIC * ... __implement__ ALS regression in a distributed fashion.
# MAGIC 
# MAGIC __`Additional Resources:`__ 
# MAGIC - The code in this notebook was based on several notebooks by Jimi Shanahan   
# MAGIC - Recommendation Systems: Techniques, Challenges, Application, and Evaluation https://www.researchgate.net/publication/328640457_Recommendation_Systems_Techniques_Challenges_Application_and_Evaluation_SocProS_2017_Volume_2
# MAGIC - Matrix Completion via Alternating Least Square(ALS) https://web.stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf    
# MAGIC - Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares https://arxiv.org/pdf/1410.2596.pdf     
# MAGIC - Explicit Matrix Factorization: ALS, SGD, and All That Jazz https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea    
# MAGIC - Collaborative Filtering for Implicit Feedback Datasets http://yifanhu.net/PUB/cf.pdf    
# MAGIC - Joeran Beel https://www.tcd.ie/research/researchmatters/joeran-beel.php   
# MAGIC - Collaborative Filtering - RDD-based API https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html

# COMMAND ----------

# MAGIC %md
# MAGIC # Background discussion
# MAGIC 
# MAGIC >From a research perspective, recommender-systems are one of the most diverse areas imaginable. The areas of interest range from hard mathematical/algorithmic problems over user-centric problems (user interfaces, evaluations, privacy) to ethical and political questions (bias, information bubbles). Given this broad range, many disciplines contribute to recommender-systems research including computer science (e.g. information retrieval, natural language processing, graphic and user interface design, machine learning, distributed computing, high performance computing) the social sciences, and many more. Recommender-systems research can also be conducted in almost every domain including e-commerce, movies, music, art, health, food, legal, or finance. This opens the door for interdisciplinary cooperation with exciting challenges and high potential for impactful work. ~Joeran Beel    
# MAGIC *Dr Joeran Beel is an Ussher Assistant Professor in the Artificial Intelligence Discipline at the School of Computer Science & Statistics at Trinity College Dublin. https://www.tcd.ie/research/researchmatters/joeran-beel.php*

# COMMAND ----------

# MAGIC %md
# MAGIC # Recommender-System Evaluation
# MAGIC >‘What constitutes a good recommender system and how to measure it’ might seem like a simple question to answer, but it is actually quite difficult.  For many years, the recommender-systems community focused on accuracy. 
# MAGIC >
# MAGIC >Accuracy, in the broader sense, is easy to quantify: numbers like error rates such as the difference between a user’s actual rating of a movie and the previously predicted rating by the recommender system (the lower the error rate, the better the recommender system); or precision, i.e. the fraction of items in a list of recommendations that was actually bought, viewed, clicked, etc. (the higher the precision, the better the recommender system). 
# MAGIC >
# MAGIC >Recently, the community’s attention has shifted to other measures that are more meaningful but also more difficult to measure including __serendipity__, __novelty__, and __diversity__. I contributed to this development by critically analyzing the state of the art [15] ; comparing evaluation metrics (click-through rate, user ratings, precision, recall, …) and methods (online evaluations, offline evaluations, user studies) [13] as well as introducing novel evaluation methods [3].
# MAGIC >
# MAGIC >Regardless of the metrics used to measure how “good” a recommender system is (accuracy, precision, user satisfaction…), studies report surprisingly inconsistent results on how effective different recommendation algorithms are. For instance, as shown in Figure 2, one of my experiments shows that five news recommendation-algorithms perform vastly different on six news websites [5]. Almost every algorithm performed best on at least one news website. Consequently, the operator of a new news website would hardly know which of the five algorithms is the best to use, because any one could potentially be it.  ~Joeran Beel   
# MAGIC *Dr Joeran Beel is an Ussher Assistant Professor in the Artificial Intelligence Discipline at the School of Computer Science & Statistics at Trinity College Dublin.* https://www.tcd.ie/research/researchmatters/joeran-beel.php

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion questions:
# MAGIC 
# MAGIC Most of us are inundated with examples of recommendation systems, from Facebook, to Amazon, to Netflix. So instead of  starting with ‘what they are’,   maybe it’s good to start with a quick discussion about what unspoken assumptions underlie their ubiquity. 
# MAGIC    
# MAGIC    
# MAGIC * What are some of the political and ethical questions related to RS? What could go wrong? Examples?
# MAGIC * What are some assumptions to keep in mind and/or try to avoid when designing RS?
# MAGIC * What is the value proposition? who are the stakeholders?
# MAGIC * What are some of the areas of expertise involved in designing RS?

# COMMAND ----------

# MAGIC %md
# MAGIC ### <--- SOLUTION --->
# MAGIC 
# MAGIC Issues   
# MAGIC * privacy (Target example)
# MAGIC * biases
# MAGIC * information bubbles
# MAGIC 
# MAGIC Value Proposition Assumption
# MAGIC * “Help people” navigate information complexity?
# MAGIC * Drive usage?
# MAGIC * Harness cognitive biases?
# MAGIC 
# MAGIC Feasibility Assumption
# MAGIC * Something in the data we have allows us to know something about your future choices.
# MAGIC 
# MAGIC Believer: recommendations save the user time    
# MAGIC Skeptic: why does the system know better than the user what they want?
# MAGIC 
# MAGIC While we're not going to delve into how to address some of these ethical and philosophical questions, it's important to keep these things in mind when thinking about designing your system.

# COMMAND ----------

# MAGIC %md
# MAGIC # Types of Recommender Systems

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/types-diagram.png" width=70%>

# COMMAND ----------

# MAGIC %md
# MAGIC https://www.researchgate.net/publication/328640457_Recommendation_Systems_Techniques_Challenges_Application_and_Evaluation_SocProS_2017_Volume_2

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/RS-comparison-table.png">

# COMMAND ----------

# MAGIC %md
# MAGIC # Representation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommender System (RS) as bipartite graph
# MAGIC 
# MAGIC We can think of the recommender problem as a weighted bipartite graph, where one set of nodes represents users, and the other set represents items. 
# MAGIC 
# MAGIC * __NODES__ - Each user can be represented by a vector of features, thinkgs like preferences, demographics, traits, etc.. and likewise, our items can also be represented by feature vectors. For example, if our items are movies, then we might have features like genre, director, lead actor, etc... (We'll talk about how these features are derived later).  
# MAGIC 
# MAGIC * __EDGES__ - The edges in our graph could be ratings, a positive or negative indicator, or another continuous measure of preference.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/bipartite-graph.png">

# COMMAND ----------

# MAGIC %md
# MAGIC # Content-based
# MAGIC Content-Based systems focus on properties of items. Similarity of items is determined by measuring the similarity in their properties. If a user bought a particular item, then we can recommend similar items. Early recommender systems employed this approach. 
# MAGIC 
# MAGIC This approach depends heavily on the similarity metrics beteen items and feature engineering. For movies that might be genre, lead actors, etc. When comparing news articles we might want to perform some topic modeling, TFIDF, cosine similarities, etc..
# MAGIC 
# MAGIC We represent each item as a vector of features, and each user as a vector of these item features, and we compute the cosine similarity between a user and an item to determine if the user will like this item.
# MAGIC 
# MAGIC While it was intuitive and easily interpretable, more effective methods have been developed since. 
# MAGIC 
# MAGIC Pros: interpretable, no cold start problem, makes use of implicit data collection .  
# MAGIC Cons: creates filter bubble

# COMMAND ----------

# MAGIC %md
# MAGIC # Collaborative Filtering

# COMMAND ----------

# MAGIC %md
# MAGIC ## Neighborhood Based
# MAGIC As discussed in DDS, we could now take a Nearest Neighbor approach, which is intuitive and simple to reason about. The intuition being that similar people like similar things.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion questions:
# MAGIC 1. How can we measure similarity between people? (*HINT:* Think about the person node representation)
# MAGIC 2. What are the challenges with this approach from a theoretical standpoint, as well as a computational one?

# COMMAND ----------

# MAGIC %md
# MAGIC ### <--- SOLUTION --->
# MAGIC 1. Jaccard, cosine, euclidean
# MAGIC > 
# MAGIC     One example when the opinions are binary:
# MAGIC     Jaccard distance, i.e., 1–(the number of things they both like divided
# MAGIC     by the number of things either of them likes)
# MAGIC     DDS pg.202
# MAGIC 
# MAGIC 2. 
# MAGIC > 
# MAGIC     * Curse of dimensionality
# MAGIC     * Overfitting
# MAGIC     * Correlated features
# MAGIC     * Relative importance of features
# MAGIC     * Sparseness
# MAGIC     * Measurement errors
# MAGIC     * Computational complexity
# MAGIC     * Sensitivity of distance metrics
# MAGIC     * Preferences change over time
# MAGIC     * Cost to update

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Based

# COMMAND ----------

# MAGIC %md
# MAGIC __2.2.2 Model-Based Filtering__
# MAGIC Model-based techniques make use of data mining and machine learning approaches
# MAGIC to predict the preference of a user to an item. These techniques include 
# MAGIC * association rule mining, (Apriori)
# MAGIC * clustering, 
# MAGIC * decision tree, 
# MAGIC * artificial neural network, 
# MAGIC * Bayesian classifier, 
# MAGIC * regression, 
# MAGIC * link analysis, and 
# MAGIC * __latent factor models.__   
# MAGIC 
# MAGIC Among these, __latent factor models__ are the most studied and used model-based techniques.
# MAGIC These techniques perform dimensionality reduction over user–item preference matrix
# MAGIC and learn latent variables to predict preference of the user to an item in the recommendation
# MAGIC process. These methods include:
# MAGIC * __matrix factorization__, 
# MAGIC * singular value decomposition, 
# MAGIC * probabilistic matrix factorization, 
# MAGIC * Bayesian probabilistic matrix factorization, 
# MAGIC * low-rank factorization, 
# MAGIC * nonnegative matrix factorization, and 
# MAGIC * latent Dirichlet allocation.   
# MAGIC 
# MAGIC Source: https://www.researchgate.net/publication/328640457_Recommendation_Systems_Techniques_Challenges_Application_and_Evaluation_SocProS_2017_Volume_2

# COMMAND ----------

# MAGIC %md
# MAGIC # Data collection - implicit vs explicit feedback
# MAGIC Before we dive in to the methodology, let's talk about how the system is popuated in the first place?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explicit feeback - users provide ratings.
# MAGIC 
# MAGIC ### Discussion questions: 
# MAGIC * What are some of the limitations of star ratings?

# COMMAND ----------

# MAGIC %md
# MAGIC ### <--- SOLUTION --->
# MAGIC 
# MAGIC * This approach is limited in its effectiveness, since generally
# MAGIC users are unwilling to provide responses, and the information from those
# MAGIC who do may be biased by the very fact that it comes from people willing
# MAGIC to provide ratings. MMDS pg.311
# MAGIC * People lie.
# MAGIC * People don't have the ability to accurately measure on a scale
# MAGIC * People's tastes change over time
# MAGIC * Some people rate higher overall, some people generally rate lower. ie, rating are not equivalent across users.
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/star_ratings.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Digression: Yahoo experiment -   
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/yahoo-experiment.png">   
# MAGIC * On the left - When users are asked to rate movies from a random list, there are many very low ratings.
# MAGIC * On the right - When the user has the freedom to choose what items to rate, instead of giving a low rating, they don’t give a rating at all.    
# MAGIC 
# MAGIC These two distributions are very different. The challenge is that we have this data for training and testing where the "true" distribution is like one on the left, but the model we build (user experience) depends on the distribution on the right.
# MAGIC We can reframe this challenge in terms of missing data. And what this means, is that we cannot make the assumption that the data is missing at random. The consequence is that we cannot ignore missing values, instead, we need a mechanism for imputing those values.   
# MAGIC For more information and how this issue can be addressed, see https://www.youtube.com/watch?v=aKQfUbxU96c marker 6:00

# COMMAND ----------

# MAGIC %md
# MAGIC ## Implicit feedback - We can make inferences from users’ behavior. 
# MAGIC If a user buys a product at Amazon, watches a movie on YouTube, or reads a news article, then the user can be said to “like” this item.
# MAGIC 
# MAGIC ### Discussion questions:
# MAGIC * What are some of the challenges of this approach?
# MAGIC * Ex: If I click on a movie but don't watch it, is that a positive or negative indicator?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Matrix Factorization

# COMMAND ----------

# MAGIC %md
# MAGIC We'll limit our implementation to explicit feedback given by users in the form of ratings. We might want to do some preprocessing, like normalization. For example, we might want to subtract the mean of the ratings to account for user bias - some users tend to rate higher than others, and vs. And we may or may not want to impute missing values, as discussed above.
# MAGIC 
# MAGIC We can represent our bipartite graph by a \\(n\times m\\) "utility" matrix \\(R\\) with entries \\(r\_{u,i}\\) representing the \\(i^{th}\\) item rating by user \\(u\\) with \\(n\\) users and \\(m\\) items.
# MAGIC 
# MAGIC Our goal is to fill in the missing (or previously imputed) values of our matrix with good estimates of future ratings. 
# MAGIC 
# MAGIC A common approach for this problem is matrix factorization where we make estimates for the complete ratings matrix \\(R\\) in terms of two matrix "factors" \\(U\\) and \\(V\\) which multiply together to form \\(R\\). Where \\(U\\) is the user matrix and \\(V\\) is the item matrix.
# MAGIC 
# MAGIC $$
# MAGIC R \approx UV
# MAGIC $$
# MAGIC 
# MAGIC We can estimate \\(R\\) by creating factor matricies with reduced complexity \\(U\in\mathbb{R}^{k,n}\\) and \\(V"\in\mathbb{R}^{k,m}\\) with \\(n\\) users, \\(m\\) items, and \\(k\\) factors.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction 
# MAGIC If we multiply each feature of the user by the corresponding feature of the movie and add everything together, this will be a good approximation for the rating the user would give that movie.
# MAGIC 
# MAGIC $$
# MAGIC r'\_{u,i} = \boldsymbol{u}^{T}\_{u}\boldsymbol{v}\_{i} = \sum\_{k} u\_{u,k}v\_{k,i}
# MAGIC $$
# MAGIC 
# MAGIC #### Assumptions
# MAGIC - Each user can be described by \\(k\\) attributes or features. For example, feature 1 might be a number that says how much each user likes sci-fi movies; however, they are ambiguous since the model derives them similar to a neural network. So we do not get the interpretability.
# MAGIC - Each item (movie) can be described by an analagous set of \\(k\\) attributes or features. To correspond to the above example, feature 1 for the movie might be a number that says how close the movie is to pure sci-fi.
# MAGIC 
# MAGIC These user and item vectors are often called latent vectors or low-dimensional embeddings.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion questions:
# MAGIC * What is a latent factor? Intuitively? Mathematically?
# MAGIC * How many latent factors should we choose? What would it mean if we had 1 latent factor? What about if we had too many? HINT: how does it relate to underfitting and overfitting

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/MF-01.png" width=70%>

# COMMAND ----------

# MAGIC %md
# MAGIC ### <--- SOLUTION --->
# MAGIC * What is a latent factor? Intuitively? Mathematically?
# MAGIC >Some "aspect" of a user or item that captures several characteristics of a user or item. Mathematically, we can think of it as a projection of the feature vector onto a lower dimension.
# MAGIC 
# MAGIC * How many latent factors should we choose? What would it mean if we had 1 latent factor? What about if we had too many? 
# MAGIC >It is possible to tune the expressive power of the model by changing the number of latent factors. It has been demonstrated that a matrix factorization with one latent factor is equivalent to a most popular or top popular recommender (e.g. recommends the items with the most interactions without any personalization). Increasing the number of latent factors will improve personalization, therefore recommendation quality, until the number of factors becomes too high, at which point the model starts to overfit and the recommendation quality will decrease. A common strategy to avoid overfitting is to add regularization terms to the objective function. https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training - Alternating Least Sqaures (ALS)
# MAGIC ### How can we find U and V to approximate R?
# MAGIC * What are we trying to optimize?       
# MAGIC * What do we start with?
# MAGIC * Explain the 3 components of this loss function.

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC L(U,V) = \Vert R- U\cdot V\Vert ^2 + \lambda \Vert U \Vert ^2 + \lambda \Vert V \Vert ^2 
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC * There is something left out of the notation shown, what is it? (hint: are we really computing the loss for every cell?)
# MAGIC * Why wouldn't we use GD on this loss function?

# COMMAND ----------

# MAGIC %md
# MAGIC Turns out that minimizing the joint optimaztion is hard. For one, this function is not convex, so there are local minima we could "get stuck in". 
# MAGIC 
# MAGIC This is where ALS comes in. It turns out if we constrain \\(U\\) or \\(V\\) to be constant, that this is a convex problem since the multiplicative factor is a constant and it has the same matrix notation as standard least square regression.
# MAGIC 
# MAGIC __STEPS__
# MAGIC 
# MAGIC - Initialize \\(U\_0\\) and \\(V\_0\\)   
# MAGIC - Holding \\(U\\) constant, solve for \\(V\_1\\) to minimize:
# MAGIC 
# MAGIC $$
# MAGIC L(V) = \Vert R- U\_0\cdot V\Vert ^2 + \lambda \Vert V \Vert ^2
# MAGIC $$
# MAGIC 
# MAGIC - Holding \\(V\\) constant, solve for \\(U\_1\\) to minimize:   
# MAGIC 
# MAGIC $$
# MAGIC L(U) = \Vert R- V\_1\cdot U\Vert ^2 + \lambda \Vert U \Vert ^2
# MAGIC $$
# MAGIC 
# MAGIC - Repeat until convergence

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion questions:
# MAGIC * When we say “solve” for \\(U\_i\\) and \\(V\_i\\), how is that actually done?
# MAGIC * Where are there opportunities to parallelize this training process?
# MAGIC * How will we partition the data to avoid shuffling?

# COMMAND ----------

# MAGIC %md
# MAGIC # Parallel ALS

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/parallel-ALS.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion questions
# MAGIC * What data needs to be cached/ broadcast at each phase?
# MAGIC * What happens in ‘mappers’ (narrow transformations) & what will happen in aggregation (wide transformations)?
# MAGIC * How many shuffles per iteration?
# MAGIC * Are there any limitations to this approach?

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notebook Set-Up

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType
schema_rtgs = StructType([StructField('user_id', IntegerType(), True), 
                      StructField('movie_id', IntegerType(), True),
                      StructField('rating', FloatType(), True),
                      StructField('timestamp', LongType(), True)
                    ])
schema_movies = StructType([StructField('movie_id', IntegerType(), True),
                      StructField('title', StringType(), True),
                      StructField('genres', StringType(), True)
                    ])

# COMMAND ----------

# imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path


# COMMAND ----------

# MAGIC %md
# MAGIC ## About the Data
# MAGIC https://grouplens.org/datasets/movielens/
# MAGIC 
# MAGIC MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.
# MAGIC 
# MAGIC This data set consists of: 
# MAGIC * 100,000 ratings (1-5) from 943 users on 1682 movies. 
# MAGIC * Each user has rated at least 20 movies. 
# MAGIC * Simple demographic info for the users (age, gender, occupation, zip)

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/ratings.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","


# The applied options are for CSV files. For other file types, these will be ignored.
ratings = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(ratings)

# COMMAND ----------

file_location = "/FileStore/tables/movies.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
movies = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(movies)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Python - Single Node Implementation

# COMMAND ----------

# The MovieLens dataset contains 10000054 ratings and 95580 tags applied to 10681 movies by 71567 users.

ratings_pd = ratings.toPandas()

movies_pd = movies.toPandas()
movie_titles = movies_pd.title.tolist()

# COMMAND ----------

df = pd.merge(movies_pd, ratings_pd, on=['movieId'], how='inner')
df['rating'] = df['rating'].astype('float')
df['movieId'] = df['movieId'].astype('int')
df['userId'] = df['userId'].astype('int')
display(df.head())

# COMMAND ----------

movies_pd.shape

# COMMAND ----------

# Getting Q Matrix
rp = df.pivot_table(columns=['movieId'],index=['userId'],values='rating').fillna(0)
rp.head()
Q = rp.values
Q.shape

# COMMAND ----------

# build a binary weight matrix (so the algo focuses on say the movies a user rated during each subproblem 
# (each user can be view as an atomic problem to be solved) that is solved)
W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)
lambda_ = 0.1 # learning rate
n_factors = 100
m, n = Q.shape

#setup user and movie factor matrices of order n_factors between [0, 5] stars
X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)
X.shape

#compute the error (Frobenus norm) where
# Q target ratings matrix
# X and Y are the factorized matrices
# W weight matrix
def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)

print(W)

# COMMAND ----------

# non-weighted version of ALS (does not work well!)
# uses all user item values (as opposed to the subset of actual ratings)

n_iterations = 20 # orig had 20

errors = []
for i in range(n_iterations):
    X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), 
                        np.dot(Y, Q.T)).T
    Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors),
                        np.dot(X.T, Q))
    print(f'{i}th iteration is completed')
    errors.append(get_error(Q, X, Y, W))
Q_hat = np.dot(X, Y)

print('')
print(f'Error of rated movies: {get_error(Q, X, Y, W)}')

# COMMAND ----------

# spark.databricks.workspace.matplotlibInline.enabled = True
%matplotlib inline

# COMMAND ----------


# display plots inline (otherwise it will fire up a separate window)
plt.plot(errors);
#plt.ylim([0, 60000000]);
#plt.ylim([0, 20000]);
plt.xlabel("ALS Iteration")
plt.ylabel("Total Squared Error")

# COMMAND ----------

n_iterations = 10
weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        #AX=B =>  X=A^-1B ; in python use solve(A, B) 
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print(f'{ii}th iteration is completed')
weighted_Q_hat = np.dot(X,Y)
print(f'Error of rated movies: {get_error(Q, X, Y, W)}')
plt.plot(weighted_errors);
plt.xlabel('Iteration Number');
plt.ylabel('Mean Squared Error');
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark - Distributed Implementation

# COMMAND ----------

import numpy as np
from numpy.random import rand
from numpy import matrix

# COMMAND ----------

def rmse(R, U, V): # Metric
    return np.sqrt(np.sum(np.power(R-U*V, 2))/(U.shape[0]*V.shape[1]))

# COMMAND ----------

def solver(mat, R, LAMBDA):  # solver to get R*mat(matT*mat + lambda*I)^-1
    d1 = mat.shape[0]
    d2 = mat.shape[1]

    X2 = mat.T * mat
    XY = mat.T * R.T

    for j in range(d2):
        X2[j, j] += LAMBDA * d1

    return np.linalg.solve(X2, XY)

# COMMAND ----------

# Not only caculation is paralleized but also the data is wisely partitioned and shared to improve locality.
def closedFormALS(R,InitialU,InitialVt,rank,iterations,numPartitions,LAMBDA=0.01):
    R_Userslice = sc.parallelize(R,numPartitions).cache() # R will automaticly be partitioned by row index
    R_Itemslice = sc.parallelize(R.T,numPartitions).cache() # R_T will automaticly be partitioned by row index
    U = InitialU
    Vt = InitialVt
    
    for i in range(iterations):
        
        print(f"Iteration: {i}")
        print(f"RMSE: {rmse(R, U, Vt.T)}")
        
        Vtb = sc.broadcast(Vt)
        U3d = R_Userslice.map(lambda x:solver(Vtb.value,x,LAMBDA)).collect() # a list of two 2-D matrix
        U = matrix(np.array(U3d)[:, :, 0]) # transfered to 2-D matrix
        
        Ub = sc.broadcast(U)
        Vt3d = R_Itemslice.map(lambda x:solver(Ub.value,x,LAMBDA)).collect() # a list of two 2-D matrix
        Vt = matrix(np.array(Vt3d)[:, :, 0])  # transfered to 2-D matrix
    
    return U, Vt 

# COMMAND ----------

# Only parallelize the calculation. It does not consider the data transmission cost
def simpleParalleling(R,InitialU,InitialVt,rank,iterations,numPartitions,LAMBDA=0.01):
    Rb = sc.broadcast(R)
    U = InitialU
    Vt = InitialVt
    Ub = sc.broadcast(U)
    Vtb = sc.broadcast(Vt)
    numUsers = InitialU.shape[0]
    numItems = InitialVt.shape[0]
    
    for i in range(iterations):
        print(f"Iteration: {i}")
        print(f"RMSE: {rmse(R, U, Vt.T)}")
        U3d = sc.parallelize(range(numUsers), numPartitions) \
           .map(lambda x: solver( Vtb.value, Rb.value[x, :],LAMBDA)) \
           .collect() # a list of two 2-D matrix
        U = matrix(np.array(U3d)[:, :, 0]) # transfered to 2-D matrix
        Ub = sc.broadcast(U)

        Vt3d = sc.parallelize(range(numItems), numPartitions) \
           .map(lambda x: solver(Ub.value, Rb.value.T[x,:],LAMBDA)) \
           .collect() # a list of two 2-D matrix
        Vt = matrix(np.array(Vt3d)[:, :, 0]) # transfered to 2-D matrix
        Vtb = sc.broadcast(Vt)
    return U, Vt

# COMMAND ----------

def main():
    LAMBDA = 0.01   # regularization parameter
    np.random.seed(100)
    numUsers = 5000
    numItems = 100
    rank = 10
    iterations = 5
    numPartitions = 2

    trueU = matrix(rand(numUsers, rank)) #True matrix U to generate R
    trueV = matrix(rand(rank, numItems)) #True matrix V to generate R
    R = matrix(trueU*trueV)   #generate Rating matrix
    
    InitialU = matrix(rand(numUsers, rank)) #Initialization of U
    InitialVt = matrix(rand(numItems,rank)) #Initialization of V
    
    print(f"Running ALS with numUser={numUsers}, numItem={numItems}, rank={rank}, iterations={n_iterations}, numPartitions={numPartitions}")
    
    print("Distributed Version---Two copies of R, one is partitioned by rowIdx, the other is partitioned by colIndx")
    closedFormALS(R,InitialU,InitialVt,rank,n_iterations,numPartitions,LAMBDA)
    
    print("Simple paralleling ---Suppose User Matrix R is small enough to be broadcast")
    simpleParalleling(R,InitialU,InitialVt,rank,n_iterations,numPartitions,LAMBDA)

# COMMAND ----------

main()

# COMMAND ----------

# MAGIC %md
# MAGIC # Spark ML implementation of ALS

# COMMAND ----------

# MAGIC %md
# MAGIC ### MlLib Tutorial on Personalized Movie Recommendation
# MAGIC Link to Docs: https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
# MAGIC 
# MAGIC * What are the parameters for the MLLib implementation of collaborative filtering?
# MAGIC * How do you determine the ‘rank’ of your latent space vectors (i.e. number of latent factors)?
# MAGIC * After training your CF model a new user joins your platform. What will you need to do to generate predictions for that user?

# COMMAND ----------

# MAGIC %md
# MAGIC ### <--- SOLUTION --->
# MAGIC * What are the parameters for the MLLib implementation of collaborative filtering?
# MAGIC >See docs
# MAGIC * How do you determine the ‘rank’ of your latent space vectors (i.e. number of latent factors)?
# MAGIC >See discussion above - think about underfitting/overfitting. For the Netflix Prize, a good heuristic is 20-100 latent factors
# MAGIC * How to make recommendations for new users? RETRAIN   
# MAGIC >When using collaborative filtering, getting recommendations is not as simple as predicting for the new entries using a previously generated model. Instead, we need to train again the model but including the new user preferences in order to compare them with other users in the dataset.
# MAGIC That is, the recommender needs to be trained every time we have new user ratings (although a single model can be used by multiple users of course!). This makes the process expensive, and it is one of the reasons why scalability is a problem (and Spark a solution!). Once we have our model trained, we can reuse it to obtain top recomendations for a given user or an individual rating for a particular movie. These are less costly operations than training the model itself.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Some implementation details
# MAGIC 
# MAGIC Solve for U and V for maxIterations
# MAGIC https://github.com/apache/spark/blob/e1ea806b3075d279b5f08a29fe4c1ad6d3c4191a/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala#L1001
# MAGIC ```
# MAGIC for (iter <- 0 until maxIter)
# MAGIC     itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
# MAGIC           userLocalIndexEncoder, solver = solver)
# MAGIC     userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
# MAGIC           itemLocalIndexEncoder, solver = solver)      
# MAGIC ```
# MAGIC 
# MAGIC where the default non-negative solver is the `ML` NNLSSolver (non-negative least squares solver)     
# MAGIC https://github.com/apache/spark/blob/e1ea806b3075d279b5f08a29fe4c1ad6d3c4191a/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala#L767
# MAGIC 
# MAGIC which calls `MLLIB` NNLS Solver...
# MAGIC which implements the [conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
# MAGIC https://github.com/apache/spark/blob/e1ea806b3075d279b5f08a29fe4c1ad6d3c4191a/mllib/src/main/scala/org/apache/spark/mllib/optimization/NNLS.scala

# COMMAND ----------

# MAGIC %md
# MAGIC # Going further

# COMMAND ----------

# MAGIC %md
# MAGIC TODO: Explain why the function is not convex. https://www.quora.com/Why-is-the-matrix-factorization-optimization-function-in-recommender-systems-not-convex
# MAGIC 
# MAGIC >A function f(x) is said to be convex if it satisfies the following property:
# MAGIC 
# MAGIC >𝑓(𝛼𝑥+𝛽𝑦)≤𝛼𝑓(𝑥)+𝛽𝑓(𝑦) where, 𝛼+𝛽=1,𝛼,𝛽≥=0and the domain of 𝑓 is a convex set.
# MAGIC 
# MAGIC >A simple matrix factorization based model would predict, say ratings, using the product of item and user latent factors
# MAGIC 
# MAGIC >𝑅𝑢𝑖=<𝑝𝑢,𝑞𝑖> where 𝑝𝑢 is the user latent factor representation and 𝑞𝑖is the item latent factor representation. The objective function includes this term and therefore is equivalent to minimizing 𝑓(𝑥,𝑦)=𝑥𝑦
# MAGIC 
# MAGIC >𝑓(𝛼𝑥1+𝛽𝑥2,𝛼𝑦1+𝛽𝑦2)=(𝛼𝑥1+𝛽𝑥2)(𝛼𝑦1+𝛽𝑦2)You can easily find a counter example to prove that this function does not satisfy the above property of convexity.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Acronym Disambiguation
# MAGIC Factoring matrices comes up a lot in the context of ML.
# MAGIC 
# MAGIC - __SVD__  - “Singular Value Decomposition”
# MAGIC - __PCA__ - “Principal Component Analysis” 
# MAGIC - __FM__ - “Factorization Machine” (one latent vector per user or item)
# MAGIC - __FFM__ - “Field Aware Factorization Machine” (multiple latent vectors depending on the latent space)

# COMMAND ----------

# MAGIC %md
# MAGIC ALS is tolerant of missing values.  
# MAGIC SVD requires all values to be present.   
# MAGIC To find the principal components of R, one might perfom SVD.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk11-demo/SVD.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Netflix Recommender system
# MAGIC https://www.youtube.com/watch?v=aKQfUbxU96c
# MAGIC 
# MAGIC SVD++ (uses both explicit and implict feedback, takes into account user and item bias)     
# MAGIC Restricted Bolzman Machine   
# MAGIC Nuclear Norm -> \\(||A||\_{nuclear} = \sigma\_1 + \sigma_2 + ... + \sigma\_r\\)

# COMMAND ----------

# MAGIC %md
# MAGIC Class for training explicit matrix factorization model using either ALS or SGD
# MAGIC 
# MAGIC https://gist.github.com/EthanRosenthal/a293bfe8bbe40d5d0995

# COMMAND ----------

# MAGIC %md
# MAGIC ## RS as a bi-partite graph - how do we solve this in graphX/graphframes?
# MAGIC Would Personalized PageRank be a reasonable approach?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deep-Learning MF
# MAGIC In recent years a number of neural and deep-learning techniques have been proposed, some of which generalize traditional Matrix factorization algorithms via a non-linear neural architecture [15]. While deep learning has been applied to many different scenarios: context-aware, sequence-aware, social tagging etc. its real effectiveness when used in a simple Collaborative filtering scenario has been put into question. A systematic analysis of publications applying deep learning or neural methods to the top-k recommendation problem, published in top conferences (SIGIR, KDD, WWW, RecSys), has shown that on average less than 40% of articles are reproducible, with as little as 14% in some conferences. Overall the study identifies 18 articles, only 7 of them could be reproduced and 6 of them could be outperformed by much older and simpler properly tuned baselines. The article also highlights a number of potential problems in today's research scholarship and calls for improved scientific practices in that area.[16] Similar issues have been spotted also in sequence-aware recommender systems.[17] https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)

# COMMAND ----------


