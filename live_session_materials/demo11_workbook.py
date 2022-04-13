# Databricks notebook source
# MAGIC %md
# MAGIC # wk11 Demo - Decision Trees
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Spring 2021`__
# MAGIC 
# MAGIC This week we'll be looking at Decision Trees
# MAGIC 
# MAGIC In class today we'll start out by reviewing the Decision Tree algorithm. We'll look at Regression and Classification trees, learning, pruning, and evluation. We'll extend our discussion to Ensemble methods, including Random Forests, Bagging, and Boosting.  
# MAGIC 
# MAGIC By the end of today's demo you should be able to:  
# MAGIC * ... __describe__ Decision Tree CART algorithm  
# MAGIC * ... __identify__ Assumptions/constraints for learning DTs
# MAGIC * ... __explain__ The difference between regression trees and classification trees
# MAGIC * ... __explain__ The difference between bagging, RF, and boosting
# MAGIC * ... __describe__ The PLANET method for distributing DT learning
# MAGIC 
# MAGIC 
# MAGIC __`Additional Resources:`__    
# MAGIC Chapter 9.2 ESL (or ISL Chapter 8) - Tree-Based Methods    
# MAGIC https://explained.ai/decision-tree-viz/index.html - How to visualize decision trees      
# MAGIC https://www.youtube.com/watch?v=iOucwX7Z1HU "Wisdom of the crowd" (jelly beans)      
# MAGIC https://explained.ai/gradient-boosting/index.html - Gradient Booted Models (GBMs) explained   
# MAGIC https://statweb.stanford.edu/~jhf/ftp/trebst.pdf - Greedy Function Approximation - a Gradient Boosting Machine  
# MAGIC https://statweb.stanford.edu/~jhf/ftp/stobst.pdf - Stochastic Gradient Boosting   
# MAGIC https://xgboost.readthedocs.io/en/latest/tutorials/model.html XGBoost Docs

# COMMAND ----------

# MAGIC %md
# MAGIC # I. Decision Tree Review 
# MAGIC *Based on ESL Chapter 9.2 - Tree Based Methods*
# MAGIC 
# MAGIC ## Benefits
# MAGIC * One of the most popular approaches to ML in  practice
# MAGIC * Can handle numeric, categorical, and ordinal  features
# MAGIC * No preprocessing required, no standardization/scaling
# MAGIC * Handles Missing values naturally
# MAGIC * NAs do not affect performance metrics
# MAGIC * Interaction features
# MAGIC * Highly Scalable
# MAGIC * Variable selection
# MAGIC * Excellent performance on a variety of problems
# MAGIC * Off the shelf with very few hyperparameters
# MAGIC 
# MAGIC 
# MAGIC ## Approach
# MAGIC * A decision tree represents a hierarchical  segmentation of the data
# MAGIC * The original segment is called the root node and is the entire data set
# MAGIC * The root node is partitioned into two or more segments by applying a series of simple rules  over input variables
# MAGIC * For example, `risk == low`, vs `risk == not low`
# MAGIC * Each rule assigns the observations to a segment based on its  input value
# MAGIC * Each resulting segment can be further  partitioned into sub-segments, and so  on
# MAGIC * For example `risk == low` can be partitioned into  `income == low` and `income == not low`
# MAGIC * The segments are also called nodes,  and the final segments are called leaf  nodes or leaves

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk12_demo/fig9.2-ESL-tree-diagrams.png?raw=true" width=100%>
# MAGIC 
# MAGIC 
# MAGIC *Partitions and CART. Left panel shows a partition of a
# MAGIC two-dimensional feature space by recursive binary splitting, as used in CART,
# MAGIC applied to some fake data. Middle panel shows the tree corresponding
# MAGIC to the partition in the left panel, and a perspective plot of the
# MAGIC prediction surface appears in the right panel.* __Based on FIGURE 9.2 Elements of Statistical Learning.__

# COMMAND ----------

# MAGIC %md
# MAGIC ## CART - Classification And Regression Trees

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression Trees
# MAGIC ### How do we grow a regression tree?
# MAGIC 
# MAGIC Our data consists of \\(N\\) observations with \\(p\\) features. Suppose we partition the data into \\(M\\) regions \\(R_1, R_2,...,R_M\\) , and model the response as a constant \\(c_m\\) in each region.
# MAGIC 
# MAGIC $$f(x) = \sum_{m=1}^{M}c_mI(x \in R_m)$$

# COMMAND ----------

# MAGIC %md
# MAGIC Using the sum of squares criterion: $$
# MAGIC \sum(y_i - f(x_i))^2 
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC The prediction \\(\hat c_m\\) is just the average of \\(y_i\\) in region \\(R_m\\): $$
# MAGIC \hat c_m = avg(y_i|x_i \in R_m)
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC __DISCUSSION__ 
# MAGIC 
# MAGIC * Can we find the best binary partition in terms of minimum sum of squares? 
# MAGIC * What is the big \\(O\\) complexity of this problem?
# MAGIC * What is another criterion often used for regression tree partitioining?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Since finding the best binary partition in terms of minimum sum of squares is computationaly infeasable, we proceed with a greedy algorithm
# MAGIC Starting with all of the data, consider a splitting variable \\(j\\) and a split point \\(s\\), we define the pair of half planes:
# MAGIC 
# MAGIC $$
# MAGIC R_1(j,s) = \{X|X_j \leq s\} \text{ and } R_2(j,s) = \{X|X_j \gt s\}
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC We seek the splitting variable \\(j\\) and split point \\(s\\) that solve  
# MAGIC 
# MAGIC $$
# MAGIC \min\_{j,s} [\min\_{c\_1} \sum\_{x\_i \in R\_1(j,s)} (y\_i - c\_1)^2 + \min\_{c\_2} \sum\_{x\_i \in R\_2(j,s)} (y\_i - c\_2)^2]
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC For any choice \\(j\\) and \\(s\\), the inner minimization is solved by
# MAGIC 
# MAGIC $$
# MAGIC \hat c_1 = avg(y_i|x_i \in R_1(j,s)) \text{ and } \hat c_2 = avg(y_i|x_i \in R_2(j,s))
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC For each splitting variable, the determination of the split point \\(s\\) can be done very quickly and hence by scanning through all of the inputs, determination of the best pair \\((j,s)\\) is feasible.
# MAGIC 
# MAGIC Having found the best split, we partition the data into the two resulting regions and repeat the splitting process on each of the two regions. Then this process is repeated on all of the resulting regions.

# COMMAND ----------

# MAGIC %md
# MAGIC __DISCUSSION__ 
# MAGIC 
# MAGIC Consider three types of variables: continuous, ordered (ex. ratings), and categorical
# MAGIC * How many split points will we have? 
# MAGIC * Notice that in equation 9.12, we split our data based on whether it is smaller or larger than our split point \\(s\\). How can we find split points for categorical variables (ie, variables which are not ordered)?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Brieman's theorem
# MAGIC 
# MAGIC For unordered domains, there are \\(p \choose 2\\) possible splits, where \\(p\\) is the number of categories.   
# MAGIC 
# MAGIC Breiman presents an algorithm for finding the best split predicate for a categorical attribute without evaluating
# MAGIC all possible subsets of \\(p\\), based on the observation that the optimal split predicate is a subsequence in the list of values for \\(p_i\\) sorted by the average \\(y\\) value.
# MAGIC 
# MAGIC <!-- <img src="brieman.png"> -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXERCISE 1
# MAGIC Run the code cells below, and answer the following questions.

# COMMAND ----------

# GENERATE DATASET: Run this cell as is
import pandas as pd

x = ["c","b","b","c","a","b","a"]
y = [0.8,0.9,1.4,0.6,3.2,2.5,3.0]
    

df = pd.DataFrame([x,y]).transpose()
df.columns = ['x', 'y']
df

# COMMAND ----------

# GET MEAN VALUES OF Y : Run this cell as is
df["y"] = pd.to_numeric(df["y"])
mean_y = df.groupby('x').mean().reset_index().sort_values(by=['y'])
mean_y

# COMMAND ----------

# MAGIC %md
# MAGIC __DISCUSSION__
# MAGIC * How many possible split points are there to start with?
# MAGIC * How many possible split points are there using Brieman's method? List the splits.
# MAGIC * How large should we grow the tree? What are the tradeoffs?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cost-Complexity Pruning 
# MAGIC ESL, pg. 308

# COMMAND ----------

# MAGIC %md
# MAGIC The preferred strategy is to grow a large tree \\(T_0\\) stopping the splitting
# MAGIC process only when some minimum node size (say 5) is reached. Then this
# MAGIC large tree is pruned using cost-complexity pruning, which we now describe.
# MAGIC We define a subtree \\(T \subset T_0 \\) to be any tree that can be obtained by
# MAGIC pruning \\(T_0\\), that is, collapsing any number of its internal (non-terminal)
# MAGIC nodes. We index terminal nodes by \\(m\\), with node \\(m\\) representing region
# MAGIC \\(R_m\\). Let \\(\lvert{T}\rvert \\) denote the number of terminal nodes in \\(T\\). Letting

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC N\_m = \\# \\{x\_i \in R\_m \\},
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC \hat{c\_m} = \frac{1}{N\_m} \sum\_{x_i \in R_m} y_i,
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC Q\_m(T) = \frac{1}{N\_m} \sum\_{x_i \in R_m} (y_i - \hat{c\_m})^2,
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC we define the *cost complexity criterion*

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC C\_{\alpha}(T) = \sum_{m=1}^{|T|} N\_m Q\_m(T) + \alpha|T|
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC The idea is to find, for each \\(\alpha\\), the subtree \\(T\_{\alpha} \subseteq T\_0\\) to minimize \\(C\_{\alpha}(T)\\).
# MAGIC The tuning parameter \\(\alpha \geq 0\\) governs the tradeoff between tree size and its
# MAGIC goodness of fit to the data. Large values of \\(\alpha\\) result in smaller trees \\(T\_{\alpha}\\), and
# MAGIC conversely for smaller values of \\(\alpha\\). As the notation suggests, with \\(\alpha = 0\\) the
# MAGIC solution is the full tree \\(T\_0\\).

# COMMAND ----------

# MAGIC %md
# MAGIC For each \\(\alpha\\) one can show that there is a unique smallest subtree \\(T\_{\alpha}\\) that
# MAGIC minimizes \\(C\_{\alpha}(T)\\). To find \\(T\_{\alpha}\\) we use *weakest link pruning*: we successively
# MAGIC collapse the internal node that produces the smallest per-node increase in
# MAGIC \\(mN\_mQ\_m(T)\\), and continue until we produce the single-node (root) tree.
# MAGIC This gives a (finite) sequence of subtrees, and one can show this sequence
# MAGIC must contain \\(T\_{\alpha}\\). See Breiman et al. (1984) or Ripley (1996) for details.

# COMMAND ----------

# MAGIC %md
# MAGIC Estimation of \\(\alpha\\) is achieved by five- or tenfold cross-validation: we choose
# MAGIC the value \\(\hat{\alpha}\\) to minimize the cross-validated sum of squares. Our final tree
# MAGIC is \\(T\_{\hat{\alpha}}\\). See Breiman et al. (1984) or Ripley (1996) for details.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification Trees
# MAGIC 
# MAGIC __DISCUSSION__
# MAGIC * What modifications do we need to make for classification trees?

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk12_demo/purity-functions.png?raw=true" width=100%>

# COMMAND ----------

# MAGIC %md
# MAGIC __FIGURE 9.3.__ *Node impurity measures for two-class classification, as a function
# MAGIC of the proportion p in class 2. Cross-entropy has been scaled to pass through
# MAGIC (0.5, 0.5).*

# COMMAND ----------

# MAGIC %md
# MAGIC In a node \\(m\\), representing a region \\(R\_m\\) with \\(N\_m\\) observations, let

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC \hat{P}\_{mk} = \frac{1}{N\_m}\sum\_{x_i \in R\_m} I(y\_i = k)
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC the proportion of class \\(k\\) observations in node \\(m\\). We classify the observations
# MAGIC in node \\(m\\) to class \\(k(m)\\) = \\(argmax\_k\\) \\(\hat{p}\_{mk}\\), the majority class in
# MAGIC node \\(m\\). Different measures \\(Q\_m(T)\\) of node impurity include the following:

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk12_demo/purity-equations.png?raw=true" width=50%>

# COMMAND ----------

# MAGIC %md
# MAGIC All three are similar, but crossentropy and the Gini index are differentiable, and hence more amenable to numerical optimization. 
# MAGIC 
# MAGIC In addition, cross-entropy and the Gini index are more sensitive to changes
# MAGIC in the node probabilities than the misclassification rate. For example, in
# MAGIC a two-class problem with 400 observations in each class (denote this by
# MAGIC (400, 400)), suppose one split created nodes (300, 100) and (100, 300), while
# MAGIC the other created nodes (200, 400) and (200, 0). Both splits produce a misclassification
# MAGIC rate of 0.25, but the second split produces a pure node and is
# MAGIC probably preferable. Both the Gini index and cross-entropy are lower for the
# MAGIC second split. For this reason, either the Gini index or cross-entropy should
# MAGIC be used when growing the tree. 
# MAGIC 
# MAGIC To guide cost-complexity pruning, any of the three measures can be used, but typically it is the misclassification rate.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Other Issues

# COMMAND ----------

# MAGIC %md
# MAGIC * __Why binary splits?__
# MAGIC Rather than splitting each node into just two groups at each stage (as
# MAGIC above), we might consider multiway splits into more than two groups. While
# MAGIC this can sometimes be useful, it is not a good general strategy. The problem
# MAGIC is that multiway splits fragment the data too quickly, leaving insufficient
# MAGIC data at the next level down. Hence we would want to use such splits only
# MAGIC when needed. Since multiway splits can be achieved by a series of binary
# MAGIC splits, the latter are preferred.
# MAGIC 
# MAGIC * __Missing Predictor Values__
# MAGIC See ESL p.311 - Surrogate predictors and split points.

# COMMAND ----------

# MAGIC %md
# MAGIC # II. Distributed Tree Algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC ## PLANET: Massively Parallel Learning of Tree Ensembles with MapReduce
# MAGIC https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36296.pdf   
# MAGIC 
# MAGIC 
# MAGIC The greedy tree induction algorithm we have described
# MAGIC is simple and works well in practice. However, it does not
# MAGIC scale well to large training datasets. FindBestSplit requires
# MAGIC a full scan of the node’s input data, which can be large at
# MAGIC higher levels of the tree. Large inputs that do not fit in main
# MAGIC memory become a bottleneck because of the cost of scanning
# MAGIC data from secondary storage. Even at lower levels of the tree
# MAGIC where a node’s input dataset D is typically much smaller
# MAGIC than D, loading D into memory still requires reading and
# MAGIC writing partitions of D to secondary storage multiple times.
# MAGIC 
# MAGIC PLANET uses MapReduce to distribute and scale tree induction to very large datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC __TODO__: Step by Step PLANET 
# MAGIC [See slides for now](https://docs.google.com/presentation/d/1Womuq5YmCNfvRZceguNjettzK0_hh3XojIanrqZ_auQ/edit#slide=id.g2b2b939f7f_1_244)

# COMMAND ----------

# MAGIC %md
# MAGIC # III. Ensembles and gradient boosting
# MAGIC 
# MAGIC The basic idea with ensembles is that several independent models can be combined to make a better model.  One example of this comes from the Netflix prize where several of the top competitors for the million dollar prize joined forced and averaged their models to produce a superior model.  There were a handful of competitors and that was enough improvement to put them into a tie for the best results.  
# MAGIC 
# MAGIC Generally, ensembles include many more than a handful of fully trained models.  The package guidance for the R package gbm (gradient boosting machine) suggests using 3000 models, for example.  How can that many different models be generated?  All of the models need to be solving more or less the same problem.  You can't do them by hand.  You need a systematic method for generating these models.  We'll look quickly at two different methods.  
# MAGIC 
# MAGIC ## Independent methods
# MAGIC Build individual trees independently
# MAGIC 
# MAGIC ### Bagging & Random Forests
# MAGIC First we'll look at Bootstrap Aggregation (called bagging).  That was invented by late Professor Leo Breiman, the famous Berkeley statistician.  Professor Breiman invented bagging to deal with the well known high-variance of binary decision trees.  Here's how it works.  
# MAGIC 
# MAGIC Generate a multitude of different training sets for the same problem.  Train a binary decision tree for each training set and average the results.  To generate multiple training sets take a random sample of the data.  The nominal formula for generating random training sets is to take a sample whose size is 50% of the original data set and extract the data from the original by sample with replacement.  Here's are some simple example to illustrate.

# COMMAND ----------

## Data generator.  
import numpy as np 
from sklearn import tree
import matplotlib.pyplot as plt


def EnsembleDataGen(npts, stdDev):
    #Define data set:
    #Starter is X sampled regularly in [-10, 10], Y = X + noise
    #Try swapping Y = X + noise for Y = np.sin(X) + noise
    X = np.linspace(-10.0, 10.0, npts)
    Y = X + np.random.normal(0.0, stdDev, npts)
    #Y = np.sin(X) + np.random.normal(0.0, stdDev, npts)
    return X.reshape([-1,1]), Y.reshape([-1,1])

X,Y = EnsembleDataGen(100, 1.0)
plt.scatter(X,Y)
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building trees for bagging
# MAGIC The core idea for bagging is to build high variance (complex) trees and then overcome the high-variance by average.  With binary decision trees the depth controls complexity (and variance).  So the trees for bagging are deeper than you might train if you were only building one tree and were trying to do the best trade off between bias and variance for a single tree.  One of the benefits of ensemble methods is that they don't require quite as much fussing with regularization parameters as single trees.

# COMMAND ----------

#Take random samples with replacement, build trees for each one and average.  
from sklearn.tree import DecisionTreeRegressor

#define weighted sum function to accumulate - (function currying)
def wt_sum_fcn(f1, f2, wt):
    def wsf(x):
        return (f1(x) + wt * f2(x))
    return wsf

def Bagging(nTrees, nDepth, bagFrac, X, Y):
    """
    nTrees - number of trees in ensemble
    nDepts - max depth of trees
    bagFrac - fractional size of bags relative to full data set
    X, Y - features, labels
    
    Return: Prediction function that is average of prediction functions of trees in ensemble
    """
    nDataPts = len(X)
    wt = float(1.0 / nTrees)
    nSamp = int(bagFrac * nDataPts)
        
    #Define function T to accumulate average prediction functions from trained trees.  
    #initialize T to fcn mapping all x to zero to start 
    T = lambda x: 0.0
    
    #loop to generate individual trees in ensemble
    for i in range(nTrees):
        
        #take a random sample from the data
        sampIdx = np.random.choice(nDataPts, nSamp)
        xTrain = X[sampIdx]
        yTrain = Y[sampIdx]
        
        #build a tree on the sampled data
        tree = DecisionTreeRegressor(max_depth=nDepth)
        tree.fit(xTrain, yTrain)
        
        #Add the new tree with a weight
        T = wt_sum_fcn(T, tree.predict, wt)
    return T
    

# COMMAND ----------

nTrees = 2  #try changing the number of trees being built
nDepth = 3   #fairly deep for 100 data points
bagFrac = 0.7   #Bag fraction - how many points in each of the random subsamples.  

bag = Bagging(nTrees, nDepth, bagFrac, X, Y)

result = bag(X)

plt.plot(X, result, 'r')
plt.scatter(X,Y)
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Some things to try
# MAGIC 1.  Change the number of trees in the ensemble through a range of values 1, 3, 5, 10.  Notice how the prediction smooths out.  
# MAGIC 2.  Change the trees to depth 1 trees.  What happens as you put more and more trees into the ensemble?  This is a good example showing that no amount of averaging will overcome a bias error.  This is why it's important to grow deep trees for bagging.  
# MAGIC 2.  In the code generator there's a suggestion to change the dependence of Y on X into a sinusoid.  Make that change and try some values for tree depth, number of trees in the ensemble to see what effect it has.  Also change the number of points in the data set and see what's required to get a relatively smooth fit.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC Random forest is similar to bagging in that we take sub-samples of the data to train individual trees and combine the results to form the final prediction. At a high level, the difference is the sampling method. RF builds trees on subsets of the features (columns), while bagging builds trees on subsets of the data (rows).    
# MAGIC 
# MAGIC In reality we often implement a combination of bagging and RF where we take a sample of data and features.
# MAGIC 
# MAGIC 
# MAGIC __Some considerations for building ensembles__:
# MAGIC * __Tree diversity__ - Creating an ensemble in which each classifier is as different as possible
# MAGIC while still being consistent with the training set is theoretically known to be
# MAGIC an important feature for obtaining improved ensemble performance.
# MAGIC 
# MAGIC * __Sub-sample distribution__ - Preprocess using K-means to insure that each tree contains some data points from each cluster, thus the individual trees distributions are similar to the original full dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combination Methods for ensembles
# MAGIC 
# MAGIC * Majority Voting
# MAGIC * Performance Weighting
# MAGIC * etc..
# MAGIC 
# MAGIC See also *Data Mining with Decision Trees: Theory and Applications; Lior Rokach and Oded Maimon* Chapter 7.3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dependent methods
# MAGIC Build trees sequentially which optimize for the error in predictions from the previous iteration.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gradient Boosting
# MAGIC Gradient boosting operates on a different principle from bagging.  The principle is easiest to explain for a regression problem like the one you just saw for bagging.  The idea with gradient boosting is that you fit a tree to the problem, then generate predicitons with that tree and subtract a small amount of the tree's prediction from the original regression labels.  Then the next tree gets trained on the leftovers.  
# MAGIC 
# MAGIC See also: https://explained.ai/gradient-boosting/index.html

# COMMAND ----------

def GradientBoosting(nTrees, nDepth, gamma, bagFrac, X, Y):
    nDataPts = len(X)
    nSamp = int(bagFrac * nDataPts)
    
    # Define function T to accumulate average prediction functions from trained trees.  
    # initialize T to fcn mapping all x to zero to start 
    T = lambda x: 0.0
    
    # loop to generate individual trees in ensemble
    for i in range(nTrees):
        
        # take a random sample from the data
        sampIdx = np.random.choice(nDataPts, nSamp)
        
        xTrain = X[sampIdx]
        
        # estimate the regression values with the current trees.  
        yEst = T(xTrain)
        
        # subtract the estimate based on current ensemble from the labels
        yTrain = Y[sampIdx] - np.array(yEst).reshape([-1,1])
        
        # build a tree on the sampled data using residuals for labels
        tree = DecisionTreeRegressor(max_depth=nDepth)
        tree.fit(xTrain, yTrain)
                
        # add the new tree with a learning rate parameter (gamma)
        T = wt_sum_fcn(T, tree.predict, gamma)
    return T

# COMMAND ----------



# COMMAND ----------

nTrees = 50  # try changing the number of trees being built
nDepth = 3   # fairly deep for 100 data points
gamma = 0.1
bagFrac = 0.5   # Bag fraction - how many points in each of the random subsamples.  

gbst = GradientBoosting(nTrees, nDepth, gamma, bagFrac, X, Y)

result = gbst(X)

plt.plot(X, result, 'r')
plt.scatter(X,Y)
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comments and some things to try
# MAGIC You may have noticed that the sampling machinery from bagging was left in the code for gradient boosting.  Friedman's first paper "Greedy Function Approximation" did not include sampling the input data.  But sampling and the basic mechanics of functional gradient descent are separate matters and Friedman's second paper "Stochastic Gradient Boosting" added that element.  Links to both these papers can be found below.  
# MAGIC 
# MAGIC #### Some things to try
# MAGIC - Try running some of the same experiments as you did with bagging. Change the tree depth, number of trees etc.  Also try the sine function for Y(X) and see how gradient boosting does with it.  
# MAGIC 
# MAGIC Here are some things that will highlight an important difference between gradient boosting and methods that are primarily variance reduction techniques.  
# MAGIC 
# MAGIC - Experiment with different tree depths and see how it affects the accuracy of the final model.  With bagging, you saw that using depth 1 trees resulted in a bias error that could not be overcome by adding more trees.  Does that happen with gradient boosting?  
# MAGIC 
# MAGIC Since gradient boosting is constantly changing the labels to emphasize the portions of the X-space where it's making the most mistakes, it will eventually pay so much attention to the edges of the data that it will start putting split points for depth 1 trees at the extreme edges of the data.  That raises the question: "Why would you ever use trees deeper than 1 with gradient boosting?"  
# MAGIC 
# MAGIC The reason for adding tree depth with gradient boosting is to cover problems where there is joint dependence on two or more variables and that dependence plays an important role in predicting the labels.  Modeling two-way or dependence requires that pairs of variables both affect some of the splits in a single tree.  That requires more than a single split in the trees.  Start with relatively shallow trees for gradient boosting.  After you've got that dialed in, then try more depth to see if you get an improvement.  
# MAGIC 
# MAGIC I hope you like gradient boosted trees.  It has won more Kaggle competitions than any other algo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### References:
# MAGIC https://explained.ai/gradient-boosting/index.html - GBM explained (__READ THIS FIRST__)    
# MAGIC https://statweb.stanford.edu/~jhf/ftp/trebst.pdf - Greedy Function Approximation - a Gradient Boosting Machine  
# MAGIC https://statweb.stanford.edu/~jhf/ftp/stobst.pdf - Stochastic Gradient Boosting

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC For __regression__ problems, the predicted response for an observation is the weighted average of the predictions using selected trees only. That is,
# MAGIC 
# MAGIC 
# MAGIC $$
# MAGIC \hat{y}\_{bag} = \frac{1}{\sum_{t=1}^{T}\alpha_t I(t\in S)} \\sum_t^T \\alpha_t \\hat{y}\_t I(t\in S)
# MAGIC $$  
# MAGIC 
# MAGIC * \\( \hat{y_i} \\) is the prediction from tree t in the ensemble.
# MAGIC 
# MAGIC * \\(S\\) is the set of indices of selected trees that comprise the prediction. \\(I(t\in S)\\) is 1 if \\(t\\) is in the set \\(S\\), and 0 otherwise.
# MAGIC 
# MAGIC * \\(\alpha_t\\) is the weight of tree \\(t\\).
# MAGIC 
# MAGIC For __classification__ problems, the predicted class for an observation is the class that yields the largest weighted average of the class posterior probabilities (i.e., classification scores) computed using selected trees only. That is,
# MAGIC 
# MAGIC * For each class \\(c\in C\\) and each tree \\(t=1...T\\), predict computes \\( \hat{P_t}(c|x) \\) which is the estimated posterior probability of class \\(c\\) given observation \\(x\\) using tree \\(t\\). \\(C\\) is the set of all distinct classes in the training data. 
# MAGIC 
# MAGIC * To make a prediction, compute the weighted average of the class posterior probabilities over the selected trees. (Note: t goes from t=1 to t=T in the summation)
# MAGIC 
# MAGIC $$
# MAGIC \hat{P}\_{bag}(c|x) = \frac{1}{\sum_{t=1}^{T}\alpha_t I(t\in S)} \sum_t^T \alpha_t \hat{P}\_t(c|x) I(t\in S)
# MAGIC $$  
# MAGIC 
# MAGIC * The predicted class is the class that yields the largest weighted average. (Let g stand for "bag" in this next eq, due to latex rendering issues)
# MAGIC 
# MAGIC $$
# MAGIC \hat{y}\_g = argmax\_{ c \in C} \hat{P}\_g(c|x)
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handling Missing Values when Applying Classification Models
# MAGIC http://jmlr.csail.mit.edu/papers/volume8/saar-tsechansky07a/saar-tsechansky07a.pdf   
# MAGIC It is important to distinguish two contexts: features may be missing at induction time, in the
# MAGIC historical “training”data, or at prediction time, in to-be-predicted “test”cases. This paper compares
# MAGIC techniques for handling missing values at prediction time.
# MAGIC 
# MAGIC 1. __Discard instances.__
# MAGIC 2. __Acquire missing values.__
# MAGIC 3. __Imputation.__
# MAGIC 4. __Reduced-feature Models:__ This can be accomplished by delaying
# MAGIC model induction until a prediction is required, a strategy presented as “lazy” classificationtree induction by Friedman et al. (1996). Alternatively, for reduced-feature modeling one may
# MAGIC store many models corresponding to various patterns of known and unknown test features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importances
# MAGIC 
# MAGIC Explaining Feature Importance by example of a Random Forest
# MAGIC - https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
# MAGIC 
# MAGIC References:
# MAGIC - https://explained.ai/rf-importance/index.html    
# MAGIC - https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-307    
# MAGIC - http://blog.datadive.net/interpreting-random-forests/
# MAGIC - http://blog.datadive.net/random-forest-interpretation-conditional-feature-contributions/

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Trees with Spark MLLib

# COMMAND ----------

# MAGIC %scala
# MAGIC val training = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
# MAGIC val test = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")
# MAGIC 
# MAGIC // Cache data for multiple uses.
# MAGIC training.cache()
# MAGIC test.cache()
# MAGIC 
# MAGIC println(s"We have ${training.count} training images and ${test.count} test images.")

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
# MAGIC import org.apache.spark.ml.feature.StringIndexer
# MAGIC import org.apache.spark.ml.Pipeline
# MAGIC 
# MAGIC // StringIndexer: Read input column "label" (digits) and annotate them as categorical values.
# MAGIC val indexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
# MAGIC // DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column.
# MAGIC val dtc = new DecisionTreeClassifier().setLabelCol("indexedLabel")
# MAGIC // Chain indexer + dtc together into a single ML Pipeline.
# MAGIC val pipeline = new Pipeline().setStages(Array(indexer, dtc))
# MAGIC 
# MAGIC val model = pipeline.fit(training)
# MAGIC 
# MAGIC // The tree is the last stage of the Pipeline.  Display it!
# MAGIC val tree = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]
# MAGIC display(tree)

# COMMAND ----------

# MAGIC %scala
# MAGIC val variedMaxDepthModels = (0 until 15).map { maxDepth =>
# MAGIC   // For this setting of maxDepth, learn a decision tree.
# MAGIC   dtc.setMaxDepth(maxDepth)
# MAGIC   // Create a Pipeline with our feature processing stage (indexer) plus the tree algorithm
# MAGIC   val pipeline = new Pipeline().setStages(Array(indexer, dtc))
# MAGIC   // Run the ML Pipeline to learn a tree.cl
# MAGIC   pipeline.fit(training)
# MAGIC }
# MAGIC 
# MAGIC // Define an evaluation metric.  In this case, we will use "weightedPrecision", which is equivalent to 0-1 accuracy.
# MAGIC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
# MAGIC val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setMetricName("weightedPrecision")
# MAGIC 
# MAGIC // For each maxDepth setting, make predictions on the test data, and compute the accuracy metric.
# MAGIC val accuracies = (0 until 15).map { maxDepth =>
# MAGIC   val model = variedMaxDepthModels(maxDepth)
# MAGIC   // Calling transform() on the test set runs the fitted pipeline.
# MAGIC   // The learned model makes predictions on each test example.
# MAGIC   val predictions = model.transform(test)
# MAGIC   // Calling evaluate() on the predictions DataFrame computes our accuracy metric.
# MAGIC   (maxDepth, evaluator.evaluate(predictions))
# MAGIC }.toDF("maxDepth", "accuracy")
# MAGIC 
# MAGIC display(accuracies)

# COMMAND ----------


