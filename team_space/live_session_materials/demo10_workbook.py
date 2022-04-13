# Databricks notebook source
# MAGIC %md
# MAGIC # Unit 10 - Graphs Part 2 - Page Rank
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Summer 2019`__

# COMMAND ----------

# MAGIC %md
# MAGIC Last week we introduced graphs and their applications. We looked at the specific challenges with parallelizing graph traversal and in particular the Single Source Shortest Path algorithm. In class today we'll look at another famous algorithm called PageRank which is an implementation of the Markov Chain, and which established Google as one of the world's most popular search engines.
# MAGIC 
# MAGIC By the end of this session, you will be able to:  
# MAGIC 
# MAGIC * ... __interpret__ the meaning of graph centrality metrics in the context of a specific problem or datset. (e.g. _the connection between eigenvector centrality and PageRank; bipartite graphs & recommender systems_.)
# MAGIC * ... __describe__ a transition matrix and stationary vector in the context of markov chains.
# MAGIC * ... __implement__ the power method to find the stationary vector.
# MAGIC * ... __identify__ the properties of a "well behaved" graph.
# MAGIC * ... __explain__ how the "random jump" adjustment makes the web graph both irreducible and aperiodic.
# MAGIC 
# MAGIC 
# MAGIC Time permitting...   
# MAGIC * ... __identify__ what chages need to be made to PR to implement Topic Specific PageRank
# MAGIC * ... __formulate__ the steps needed to implement TextRank (e.g. _how to convert unstructured text into a graph datastructure which can be fed to  the PageRank algorithm_. )
# MAGIC * ... __describe__ the relationship of *PageRank* to *PCA (Principal Components Analysis)* and *SVD (Singular Value Decomposition)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook setup

# COMMAND ----------

pip install networkx

# COMMAND ----------

# imports
from IPython.display import Image
import re
import heapq
import itertools
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph Centrality

# COMMAND ----------

# MAGIC %md
# MAGIC __Identifying Special Nodes__ (Centrality): There are many ways to define what 'special' means, including metrics based on node in-degree, average distance to other nodes, and relationship to cluster structure. These special nodes are important to investigators attempting to break up terrorist cells, epidemiologists modeling the spread of diseases, advertisers trying to promote products, and many others. In particular, eigenvector centrality is used to find popular webpages using the PageRank algorithm, which in turn has been extended to Text summarization, Keyword extraction/Concept extraction using TextRank.
# MAGIC 
# MAGIC ### A few examples of graph centrality measures
# MAGIC 
# MAGIC __Notation:__
# MAGIC The neighborhood of a vertex \\(v\\) in graph \\(G\\) is defined as a set of neighbors of a vertex \\(v\\) and is denoted by \\(N(v)\\). The neighborhood size is the number of immediate neighbors to a vertex. 
# MAGIC 
# MAGIC The number of edges between all neighbors of a vertex is denoted by \\(E(v)\\). In the directed graph, the set of \\(N\_{in}(v)\\) is the set of vertices that point to a vertex \\(v\\) (predecessors) and set of \\(N\_{out}(v)\\) is the set of vertices that vertex \\(v\\) points to (successors).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Clustering coefficient
# MAGIC The clustering coefficient of a vertex measures the density of edges among the immediate neighbors of a vertex. It determines the probability of the presence of an edge between any two neighbors of a vertex. It is calculated as a ratio between the number of edges \\(E\_i\\) that actually exist among these and the total possible number of edges among neighbors: 
# MAGIC 
# MAGIC $$
# MAGIC c(v) = \frac{2E(v)}{|N(v)|(|N(v)|-1)}
# MAGIC $$
# MAGIC 
# MAGIC 
# MAGIC #### Degree Centrality
# MAGIC The degree \\(d(v)\\) of a vertex 𝑣 is the number of edges at \\(v\\); it is equal to the number of neighbors of \\(v\\). The degree centrality \\(C\_d(v)\\) of a vertex \\(v\\) is defined as the degree of the vertex. It can be normalized by dividing it by the maximum possible degree \\(N - 1\\):
# MAGIC 
# MAGIC $$
# MAGIC C\_d(v) = \frac{d(v)}{N-1}
# MAGIC $$
# MAGIC 
# MAGIC #### Betweenness centrality 
# MAGIC The betweenness centrality \\(C\_b(v)\\) of a vertex \\(v\\) quantifies the number of times a vertex acts as a bridge along the shortest path between two other vertices. Let \\(\sigma\_{out}\\) be the number of the shortest paths from vertex \\(u\\) to vertex \\(t\\) and let \\(\sigma\_{out}(v)\\) be the number of those paths that pass through the vertex \\(v\\). The normalized betweenness centrality of a vertex \\(v\\) should be divided by the number of all possible edges in the graph and is given by:
# MAGIC 
# MAGIC $$
# MAGIC C\_b(v) = \frac{2\sum\_{v\ne u\ne t}\frac{\sigma\_{ut}(v)}{\sigma\_{ut}}}{(N-1)(N-2)}
# MAGIC $$
# MAGIC 
# MAGIC #### Closeness centrality 
# MAGIC In a connected graph, closeness centrality (or closeness) of a node is a measure of centrality in a network, calculated as the reciprocal of the sum of the length of the shortest paths between the node and all other nodes in the graph. Thus, the more central a node is, the closer it is to all other nodes. When speaking of closeness centrality, people usually refer to its normalized form which represents the average length of the shortest paths instead of their sum. It is generally given by:
# MAGIC 
# MAGIC $$
# MAGIC C\_{closeness}(v) = \frac{|V|}{\sum\_u d(u,v)}
# MAGIC $$
# MAGIC 
# MAGIC where where \\(|V|\\) is the number of vertices in the graph, and \\(d(u,v)\\) is the distance between vertices \\(v\\) and \\(u\\). 
# MAGIC 
# MAGIC 
# MAGIC #### Eigenvector centrality   
# MAGIC The eigenvector centrality \\(C\_{EV}(v)\\) measures the centrality of a vertex \\(v\\) as a function of the centralities of its neighbors. For the vertex \\(v\\) and constant \\(\lambda\\) it is defined:
# MAGIC 
# MAGIC $$
# MAGIC C\_{EV}(v) = \frac{1}{\lambda} \sum\_{u\in N(v)} C\_{EV}(u)
# MAGIC $$
# MAGIC 
# MAGIC In the case of weighted networks, the equation can be generalized. Let \\(w\_{uv}\\) be the weight of edge between vertices \\(u\\) and \\(v\\) and \\(\lambda\\) a constant. The eigenvector centrality of a vertex \\(v\\) is given by:
# MAGIC 
# MAGIC $$
# MAGIC C\_{EV}(v) = \frac{1}{\lambda} \sum\_{u\in N(v)} w\_{uv} \times C\_E(u)
# MAGIC $$
# MAGIC 
# MAGIC The PageRank centrality is based on the eigenvector centrality measure and implements the concept of "voting". The PageRank score of a vertex \\(v\\) is initialized to a default value and computed iteratively until convergence using the following equation:
# MAGIC 
# MAGIC $$
# MAGIC C\_{pagerank}(v) = (1-d) + d \sum\_{u \in N\_{in}(v)} \frac {C\_{PageRank}(u)}{|N\_{out}(u)|}
# MAGIC $$
# MAGIC where \\(d\\) is the damping factor set between 0 and 1 (usually set to 0.85).
# MAGIC 
# MAGIC  
# MAGIC __Sources__:   
# MAGIC https://www.researchgate.net/publication/280092953_An_Overview_of_Graph-Based_Keyword_Extraction_Methods_and_Approaches   
# MAGIC https://en.wikipedia.org/wiki/Centrality

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Background - From Random Walks to Markov Chains to PageRank
# MAGIC 
# MAGIC ## The World Wide Web

# COMMAND ----------

# MAGIC %md
# MAGIC There are over 4 billion internet users in 2019, and nearly 2 billion websites. Each of those websites can have multiple pages, and Google only crawls a fraction of them.
# MAGIC 
# MAGIC Wikipedia alone consists of almost 6 million articles, and nearly 50 million wiki pages.
# MAGIC 
# MAGIC Source of above stats: [https://www.websitehostingrating.com/internet-statistics-facts/](https://www.websitehostingrating.com/internet-statistics-facts/)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/google-index.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Source of above chart: [https://www.worldwidewebsize.com/](https://www.worldwidewebsize.com/)

# COMMAND ----------

# MAGIC %md
# MAGIC ### So how does Google decide which pages are "relevant"?
# MAGIC https://stanford.edu/~rezab/classes/cme323/S15/notes/lec7.pdf
# MAGIC 
# MAGIC The goal is to find a good metric for measuring the importance of each node (page) in the graph (corresponding to a ranking over the websites). A website’s importance will be measured by the number of sites that link to it. Ideally, we would like the amount of importance conferred on a website by receiving a link to be proportional to the importance of the website giving the link. (*the centrality of a vertex \\(v\\) as a function of the centralities of its neighbors - see eignvector centrality definition above*)

# COMMAND ----------

# MAGIC %md
# MAGIC We can formalize this intuition via the process of a random walk. We would like to model a __random surfer__ who is traversing the web with uniform probability of following any link outgoing from the page the surfer is currently on. We are intersted in the behavior of this random surfer in the limit as she takes an infinite number of jumps.

# COMMAND ----------

# MAGIC %md
# MAGIC To express this in linear algebra, we will make use of the adjacency matrix, \\(A\\), and a out-degree matrix \\(D\\), where:

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/rezab-01.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Let \\(Q = D ^{-1} A\\). This forms the transition matrix of the random-walker. Given the current state
# MAGIC of the walker each row of the matrix gives the probability the walker will transition to each new
# MAGIC state.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/rezab-02.png">

# COMMAND ----------

# MAGIC %md
# MAGIC It’s interesting to note that \\(Q^k_{ij}\\) is equal to the probability of going from node \\(i\\) to node \\(j\\) in exactly \\(k\\) steps, in a random walk over graph \\(G\\). (The \\(k\\) steps are the time-steps, not to be confused with hops (*paths*))
# MAGIC 
# MAGIC The __stationary distribution (or steady state)__ of the __Markov Chain__ with the transition probabilities defined by \\(Q\\) is the solution to our PageRank problem as defined above. The stationary distribution can be interpreted as the proportion of time on average is spent at a specific state (page) during an infinitely long random walk.

# COMMAND ----------

# MAGIC %md
# MAGIC # WAIT, WAT?
# MAGIC ## A historical digression:

# COMMAND ----------

# MAGIC %md
# MAGIC <table>
# MAGIC     <tr>
# MAGIC         <td><img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/220px-Jakob_Bernoulli.jpg" style="text-align:left;"></td>
# MAGIC         <td><img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/NekrasovPA.jpg" style="text-align:left;"></td>
# MAGIC         <td><img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/markov-coffee.png" style="text-align:left;"></td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td style="text-align:left;width:33%;"><h3>Jacob Bernoulli</h3></td>
# MAGIC         <td style="text-align:left;width:33%;"><h3>Pavel Alekseevich Nekrasov</h3></td>
# MAGIC         <td style="text-align:left;width:33%;"><h3>Andrey Andreyevich Markov</h3></td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC         <td style="text-align:left;vertical-align:top;"><p>Jacob Bernoulli (also known as James or Jacques; 6 January 1655 – 16 August 1705) was one of the many prominent mathematicians in the Bernoulli family. Following his father's wish, he studied theology and entered the ministry. But contrary to the desires of his parents, he also studied mathematics and astronomy. </p>
# MAGIC             <ul>
# MAGIC                 <li>Weak law of large numbers </li>
# MAGIC                 <li>Central Limit theorem</li>
# MAGIC             </ul>
# MAGIC </td>
# MAGIC         <td style="text-align:left;vertical-align:top;"><p>Pavel Alekseevich Nekrasov (1853–1924) was a Russian mathematician and a Rector of the Imperial University of Moscow.</p></td>
# MAGIC         <td style="text-align:left;vertical-align:top;"><p>Andrey Andreyevich Markov (1856 – 1922) was a Russian mathematician best known for his work on stochastic processes. A primary subject of his research later became known as Markov chains and Markov processes. He was seen as a rebellious student by a select few teachers. In his academics he performed poorly in most subjects other than mathematics. Markov was an atheist. In 1912 he protested Leo Tolstoy's excommunication from the Russian Orthodox Church by requesting his own excommunication. The Church complied with his request.</p></td>
# MAGIC     </tr>
# MAGIC     <tr><td colspan="3">Source: <a href="https://en.wikipedia.org/wiki/Main_Page">Wikipedia</a></td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC ## The origin of the Markov Chain by Britt Cruise
# MAGIC https://www.khanacademy.org/computing/computer-science/informationtheory/moderninfotheory/v/markov_chains

# COMMAND ----------

# MAGIC %md
# MAGIC Plato speculated that after an uncountable number of years, the universe will reach an ideal state, returning to its perfect form. But it wasn't until the 16th century that Bernoulli came along and refined the idea of expectation. He was focused on a method of accurately estimating the unknown probability of some event based on the number of times the event occurs in independent trials.
# MAGIC (pebbles in cups example). He went on to conclude "If observations of all events be continued for the entire infinity, it will be noticed that everything in the world is governed by precise ratios and a constant law of change." This became known as the weak __law of large numbers__.  This idea was quickly extended as it was noticed that not only did things converge on an expected average, but the probability of variation away from averages also follow a familiar, underlying shape, or distribution. This overall curvature, known as the binomial distribution, appears to be an ideal form as it kept appearing everywhere any time you looked at the variation of a large number of random trials (coin toss example). It seems the average fate of these events is somehow predetermined, known today as the __central limit theorem__.
# MAGIC 
# MAGIC This was a dangerous philosophical idea to some!
# MAGIC 
# MAGIC Nekrasov didn't like the idea of a predetermined statistical fate, and claimed that __Independence__ is a neccessary condition for the law of large numbers. Since independence just describes these toy examples using beans or dice, where the outcome of previous events doesn't change the probability of the current or future events. However, as we all can relate, most things in the physical world are clearly dependent on prior outcomes, such as the chance of fire or sun or even our life expectancy. When the probability of some event depends, or is conditional, on previous events, we say they are dependent events, or dependent variables.
# MAGIC 
# MAGIC This claim angered another Russian mathematician, Andrey Markov, who maintained a very public animosity towards Nekrasov. He goes on to say in a letter that "this circumstance prompts me to explain in a series of articles that the law of large numbers can apply to dependent variables", using a construction which he brags Nekrasov cannot even dream about! 
# MAGIC 
# MAGIC Markov extends Bernoulli's results to dependent variables using an ingenious construction. (*putting pebbles back into buckets depending on state*), and thus coining the __Markov property__. Let's explore what that looks like:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discussion Question:
# MAGIC * What becomes steady? Do the states become steady? Does it mean that you will always be guaranteed to go from page i to page j?
# MAGIC * No - the state of the chain keeps jumping forever! What besomes steady, are the probailities.   
# MAGIC https://www.youtube.com/watch?v=IkbkEtOOC1Y marker 38:30

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Markov Chain via Power Iteration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap:
# MAGIC 
# MAGIC * A __Markov chain__, named after Andrey Markov, is a  mathematical system that undergoes transitions  from one state to another, between a finite or  countable number of possible states.
# MAGIC * It is a __random process__ usually characterized as  __memoryless__: the next state depends only on the  current state and not on the sequence of events that preceded it.
# MAGIC * This specific kind of "memorylessness" is called  the __Markov property__. Markov chains have many  applications as statistical models of real-world  processes.
# MAGIC * A Markov chain consists of __state vector__ of dimension \\(n\\) and an \\(n \times n\\) transition __probability matrix__ \\(P\\).
# MAGIC * At each time step \\(k\\), we are in exactly one of the states.
# MAGIC * For \\(1 ≤ i,j ≤ n\\), the matrix entry \\(P\_{ij}\\) tells us the  probability of \\(j\\) being the next state, given we are  currently in state \\(i\\).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's look at an example "web graph" with nodes representing pages, and edges representing links:
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/web-graph01.png" width="30%">
# MAGIC We can represent this graph as a transition matrix \\(A\\), and an out-degree matrix \\(D\\), from which we can construct our transition matrix \\(Q\\).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/graph02.png" width="80%">

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/graph03.png" width=90%>

# COMMAND ----------

# MAGIC %md
# MAGIC # Will this always work?

# COMMAND ----------

# MAGIC %md
# MAGIC # Exercise 1
# MAGIC In the following examples, we're going to make some modifications to our graph structure, and attempt to perform Power Iteration using the provided function. Notice if/how each converges and comment on the behavior you obeserve. Let's start with a baseline and use the networkx library to establish some expected results. Run the next 3 cells as is.

# COMMAND ----------

# RUN THIS CELL AS IS
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

# Let's make a graph - RUN THIS CELL AS IS
G=nx.DiGraph()

G.add_edge("1","2",weight=1)
G.add_edge("1","4",weight=1)
G.add_edge("2","1",weight=1)
G.add_edge("2","3",weight=1)
G.add_edge("2","4",weight=1)
G.add_edge("3","1",weight=1)
G.add_edge("4","1",weight=1)
G.add_edge("4","3",weight=1)

esmall=[(u,v) for (u,v,d) in G.edges(data=True)]

pos=nx.spring_layout(G)

# nodes
nx.draw_networkx_nodes(G,pos,node_size=1200,node_color="#cccccc")
nx.draw_networkx_edges(G,pos,edgelist=esmall,width=1,alpha=0.5,edge_color='b',arrows=True)

# labels
nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')

plt.axis('off')
plt.show() # display

# COMMAND ----------

# Sanity check using networkx - RUN THIS CELL AS IS
nx.pagerank(G, alpha=0.85)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Power Iteration method produces the steady state vector, which should match the PageRank result above. Or does it?

# COMMAND ----------

# Power Iteration helper function - RUN THIS CELL AS IS
import numpy as np
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    state_vector = xInit
    for ix in range(nIter):    
        
        new_state_vector = state_vector@tMatrix
        state_vector = new_state_vector
        
        if verbose:
            print(f'Step {ix}: {state_vector}')
            
    return state_vector

xInit = np.array([1,0,0,0]) 

# COMMAND ----------

# MAGIC %md
# MAGIC > __DISCUSSION QUESTIONS:__     
# MAGIC For each of the graphs below, answer the following questions:
# MAGIC * Does this matrix converge? On What?
# MAGIC * What could explain this behavior?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 1
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/web-graph01.png" width="30%">

# COMMAND ----------

# Example 1
A = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [1, 0, 1, 0]])
D_1 = np.array([[1/2, 0, 0, 0], [0, 1/3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1/2]])
Q = D_1@A

# Change the number of iterations. 
# Set VERBOSE to true to see the output at each iteration
n_iterations = 30
VERBOSE = True
xInit = np.array([1,0,0,0])
power_iteration(xInit, Q, n_iterations, verbose=VERBOSE)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 2
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/web-graph02.png" width="30%">

# COMMAND ----------

# Example 2
A = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]])
D_1 = np.array([[1/3, 0, 0, 0], [0, 1/3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1/2]])
Q = D_1@A

# Change the number of iterations. 
# Set VERBOSE to true to see the output at each iteration
n_iterations = 2000
VERBOSE = True

power_iteration(xInit, Q, n_iterations, verbose=VERBOSE)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Example 3:
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/web-graph03.png" width="30%">

# COMMAND ----------

# Example 3
A = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
D_1 = np.array([[1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1/2, 0], [0, 0, 0, 1/2]])
Q = D_1@A

# Change the number of iterations. 
# Set VERBOSE to true to see the output at each iteration
n_iterations = 20
VERBOSE = True

power_iteration(xInit, Q, n_iterations, verbose=VERBOSE)


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Well behaved graphs
# MAGIC #### MIT 6.041 Probabilistic Systems Analysis and Applied Probability   
# MAGIC These lectures go into more detail about Markov Chains, and are a great resource if you want to dig deeper. The examples below were taken from these lectures.
# MAGIC 
# MAGIC Lecture 16 - Markov Chains Part 1  
# MAGIC https://www.youtube.com/watch?v=IkbkEtOOC1Y
# MAGIC 
# MAGIC Lecture 17 - Markov Chains Part 2  
# MAGIC https://www.youtube.com/watch?v=ZulMqrvP-Pk

# COMMAND ----------

# MAGIC %md
# MAGIC ## Primitivity - a graph is primitive if it is irreducible and aperiodic
# MAGIC * [Ergodic Markov Chain](https://www.youtube.com/watch?v=ZjrJpkD3o1w)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Irreducibility
# MAGIC * A Markov chain is said to be irreducible if its state space is a __single communicating class__; in other words, if it is possible to get to any state from any state.
# MAGIC 
# MAGIC Why is this important? Let's look at an example from the MIT videos.    
# MAGIC 
# MAGIC ### Recurrent and transient states
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/irreducibility-02.png">

# COMMAND ----------

# Recurrent State - musical interlude! (RUN CELL AS IS)
from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/EqPtz5qN7HM?start=234" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Does the limit depend on initial state?
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/irreducibility-01.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Periodicity
# MAGIC * Well behaved graphs are aperiodic: The GCD (greatest common divisor) of all  cycle lengths is 1. The GCD is also called period.
# MAGIC 
# MAGIC Why is this important? Again, let's look at an example from the MIT videos.
# MAGIC Remember that the Markov Property of independence says that no matter where we start, the probability of transitioning to a given state does not change. In other words, we have a steady state distribution.
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/periodicity-01.png">
# MAGIC 
# MAGIC If you start at state 2, you have a 50/50 chance of going left or right. There is some randomness, but this randomness is limited. No matter whether you go left or right, you always come back to 2 in the next step. You go out, you go in, you go out, you go in - there is a periodic pattern that gets repeated.
# MAGIC 
# MAGIC It means that if you start at state 2, after an even number of steps, you are certain to be back at state 2, so the probability is always 1 (for an even number of steps). If the number of transitions is odd, there is no way you can be at state 2. At odd number of steps, you will be at either the left or right, so the probability of being at state 2 is always 0.
# MAGIC 
# MAGIC As n (number of transitions) goes to infinity, this state 2 probability does not converge to anything. It keeps alternating between 0 and 1.

# COMMAND ----------

# MAGIC %md
# MAGIC # Exercise 2
# MAGIC ## Is this graph aperiodic?    
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/peridicity-fun-example-01.png">

# COMMAND ----------

# MAGIC %md
# MAGIC __NO!__
# MAGIC If you start in a purple state, you can only go to a white state, and vs. Again, as N (number of transitions) goes to infinity, the probability of being in the purple or white state does not converge to anything. It keeps alternating between 0 and 1.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/peridicity-fun-example-02.png">

# COMMAND ----------

# MAGIC %md
# MAGIC If you are at a purple state, then the probability of going into a white state is 1. But if you are in a white state, then the probability of going to a white state is 0. No matter how many times you walk this graph (ie, ad infinitum), these probabilities will continue to oscillate, and __will never settle on a steady state distribution. Here the initial condition has an influence on the probabilities of being in each state.__

# COMMAND ----------

# MAGIC %md
# MAGIC ## How can we fix this?

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/peridicity-fun-example-03.png">

# COMMAND ----------

# MAGIC %md
# MAGIC A single self loop fixes this. You can now got to any color state in both even and odd number of steps. As the number of iterations approaches infinity, the probabilities converge (they no longer alternate).

# COMMAND ----------

# MAGIC %md
# MAGIC ## The steady state convergence theorem - Intuition
# MAGIC 
# MAGIC Think of two copies of the chain that start at different initial states. 
# MAGIC 
# MAGIC The state moves randomly, and as the state moves around randomly starting from the two initial states, on a random trajectory… as long as you have a single recurrent class, and you don’t have periodicity, at some point those two states (those trajectories) are going to collide. Just because there’s enough randomness there. After the state becomes the same, the future of those trajectories, probabilistically, is the same, because they both started in the same state.
# MAGIC 
# MAGIC So this means that the initial conditions stopped having any influence.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stochasticity - a graph is stochastic if all rows sum to 1
# MAGIC 
# MAGIC ### Perron-Frobenius Theorem:
# MAGIC Any __irreducible, aperiodic, stochastic__ matrix \\(P\\) has an eigenvalue \\(\lambda\_0 = 1\\) with unique associated left eigenvector \\(e\_0 > 0\\). Moreover, all other eigenvalues \\(\lambda\_i\\) of \\(P\\) satisfy \\(|\lambda\_i| < 1\\).

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Connecting the dots: 
# MAGIC ## Random Surfer -> Markov  Process -> PageRank
# MAGIC 
# MAGIC Adapting the machinery of the Markov processes gives us a principled approach to calculate the pagerank of each webpage. 
# MAGIC It is nothing more than  the steady state probability distribution of the  markov process underlying the random surfer model of web navigation.
# MAGIC 
# MAGIC 
# MAGIC __PageRank__ is a link analysis algorithm that  "measures” relative importance of each page within  the webgraph.
# MAGIC \\(v\_0P = v\_0\\)
# MAGIC 
# MAGIC ## But... is the web-graph well behaved?
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/web-graph-01.png" width="40%" align="left">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adjustments for PageRank
# MAGIC 
# MAGIC * Random Jump factor adds a small amount of probability to each node to teleport to any other node.
# MAGIC * Both issues of periodicity and irreducibility are solved at once.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/web-graph.png">

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. PageRank at Scale

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/wk10-demo/PR-illustration.png">

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC P^T = (1-\alpha)\big(\frac{m}{|G|} + P'\big) + \alpha \frac{1}{|G|} 
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC Just like Single Source Shortest Path, this is an iterative algorithm that requires that we maintain the entire graph structure at each iteration.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimization
# MAGIC Custom partitioning - how can we leverage custom partitioning to speed up PageRank? http://stanford.edu/~rezab/dao/notes/Partitioning_PageRank.pdf   
# MAGIC Use preservesPartitioning=True. See HW5 solution notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC # Extensions
# MAGIC 
# MAGIC ### Personalized PageRank
# MAGIC Markov models have also been used to analyze web navigation behavior of users. A user's web link transition on a particular website can be modeled using first- or second-order Markov models and can be used to make predictions regarding future navigation and to personalize the web page for an individual user.
# MAGIC 
# MAGIC ### Topic Specific PageRank
# MAGIC ### TextRank
# MAGIC ### PCA/SVD

# COMMAND ----------

# MAGIC %md
# MAGIC # Going Further

# COMMAND ----------

# MAGIC %md
# MAGIC ### Google Knowledge Graph
# MAGIC https://en.wikipedia.org/wiki/Knowledge_Graph

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scalable Graph Processing Frameworks: A Taxonomy and Open Challenges
# MAGIC http://www.buyya.com/papers/GraphProcessing-ACMCS.pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ### Learning to Rank
# MAGIC https://en.wikipedia.org/wiki/Learning_to_rank

# COMMAND ----------

# MAGIC %md
# MAGIC > At web search, there are many signals which can represent relevance, for example, the anchor
# MAGIC texts and PageRank score of a web page. Incorporating
# MAGIC such information into the ranking model and automatically constructing the ranking model using machine
# MAGIC learning techniques becomes a natural choice. In web
# MAGIC search engines, a large amount of search log data, such
# MAGIC as click through data, is accumulated. This makes it
# MAGIC possible to derive training data from search log data
# MAGIC and automatically create the ranking model. In fact,
# MAGIC learning to rank has become one of the key technologies for modern web search.
# MAGIC 
# MAGIC http://times.cs.uiuc.edu/course/598f14/l2r.pdf

# COMMAND ----------

from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *

# COMMAND ----------

vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])

# COMMAND ----------

edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
  ("f", "c", "follow"),
  ("e", "f", "follow"),
  ("e", "d", "friend"),
  ("d", "a", "friend"),
  ("a", "e", "friend")
], ["src", "dst", "relationship"])

# COMMAND ----------

from graphframes.examples import Graphs
same_g = Graphs(sqlContext).friends()
print(same_g)

# COMMAND ----------

g = GraphFrame(vertices, edges)
print(g)

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.inDegrees)

# COMMAND ----------

youngest = g.vertices.groupBy().min("age")
display(youngest)

# COMMAND ----------

paths = g.bfs("name = 'Esther'", "age < 32")
display(paths)

# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)

# COMMAND ----------



# COMMAND ----------


