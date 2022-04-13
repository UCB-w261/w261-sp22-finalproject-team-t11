# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Trees

# COMMAND ----------

# Taken from Google Developers https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb

# Toy dataset.
# Format: each row is an example.
# The last column is the label.
# The first two columns are features.
# Feel free to play with it by adding more features & examples.
# Interesting note: I've written this so the 2nd and 5th examples
# have the same features, but different labels - so we can see how the
# tree handles this case.
# training_data = [
#     ['Green', 3, 'Apple'],
#     ['Yellow', 3, 'Apple'],
#     ['Red', 1, 'Grape'],
#     ['Red', 1, 'Grape'],
#     ['Yellow', 3, 'Lemon'],
# ]

# COMMAND ----------

df = spark.read.option('header', True).csv('/iris_csv.csv')
display(df)

# COMMAND ----------

training_data = []
test_data = []

train_, test_ = df.randomSplit([0.8, 0.2])

for i, item in enumerate(train_.collect()):
  training_data.append([float(item[0]), float(item[1]), float(item[2]), float(item[3]), item[4]])

for i, item in enumerate(test_.collect()):
  test_data.append([float(item[0]), float(item[1]), float(item[2]), float(item[3]), item[4]])
  
training_data

# COMMAND ----------

# Column labels.
# These are used only to print the tree.
header = ["sepal_lenght", "sepal_width", "petal_length", "petal_width", "label"]

# COMMAND ----------

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

# COMMAND ----------

#######
# Demo:
# 0 is our Sepal Length feature
unique_vals(training_data, 0)
#######

# COMMAND ----------

# Try now with 1, our Petal Length feature
unique_vals(training_data, 2)

# COMMAND ----------

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# COMMAND ----------

#######
# Demo:
class_counts(training_data)
#######

# COMMAND ----------

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

# COMMAND ----------

#######
# Demo:
is_numeric(7)
# is_numeric("Red")
#######

# COMMAND ----------

class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    def get_feature(self):
      # Get column integer for a particular question 
      return self.column

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

# COMMAND ----------

#######
# Demo:
# Let's write a question for a numeric attribute
Question(0, 6.0)

# COMMAND ----------

# How about one for a categorical attribute
q = Question(0, 2.5)
q

# COMMAND ----------

# Let's pick an example from the training set...
example = training_data[0]
# ... and see if it matches the question
q.match(example) # this will be True
#######

# COMMAND ----------

def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

# COMMAND ----------

#######
# Demo:
# Let's partition the training data based on whether rows are larger or less than 5.0.
true_rows, false_rows = partition(training_data, Question(0, 6.0))
# This will contain all 
true_rows

# COMMAND ----------

# This will contain everything else.
false_rows
#######

# COMMAND ----------

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl, count in counts.items():
        prob_of_lbl = count / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

# COMMAND ----------

#######
# Demo:
# Let's look at some example to understand how Gini Impurity works.
#
# First, we'll look at a dataset with no mixing.
no_mixing = [['Iris-versicolor'],
              ['Iris-versicolor']]
# this will return 0
gini(no_mixing)

# COMMAND ----------

# Now, we'll look at dataset with a 50:50 Iris-versicolor:Iris-virginica ratio
some_mixing = [['Iris-versicolor'],
               ['Iris-virginica']]
# this will return 0.5 - meaning, there's a 50% chance of misclassifying
# a random example we draw from the dataset.
gini(some_mixing)

# COMMAND ----------

# Now, we'll look at a dataset with many different labels
lots_of_mixing = [['Iris-setosa'],
                  ['Iris-versicolor'],
                  ['Iris-virginica'],
                  ['Iris-versicolor'],
                  ['Iris-virginica'],
                  ['Iris-setosa'],
                  ['Iris-setosa']]
# This will return 0.8
gini(lots_of_mixing)
#######

# COMMAND ----------

def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

# COMMAND ----------

#######
# Demo:
# Calculate the uncertainy of our training data.
current_uncertainty = gini(training_data)
current_uncertainty

# COMMAND ----------

# How much information do we gain by partioning on Sepal Length above 5.0?
true_rows, false_rows = partition(training_data, Question(0, 5.0))
info_gain(true_rows, false_rows, current_uncertainty)

# COMMAND ----------

# What about if we partioned on Sepal Length 6.0 instead?
true_rows, false_rows = partition(training_data, Question(0, 6.0))
info_gain(true_rows, false_rows, current_uncertainty)

# COMMAND ----------

# It looks like we learned more using '6.0' (0.1699), than '5.0' (0.0854).
# Why? Look at the different splits that result, and see which one
# looks more 'unmixed' to you.
true_rows, false_rows = partition(training_data, Question(0, 6.0))

# Here, the true_rows contain only Sepal Length above 6.0.
true_rows

# COMMAND ----------

# Here are the False rows
false_rows

# COMMAND ----------

# Create a global list of columns to keep track
# of features isued in tree
features = [i for i in range(len(training_data[0])-1)]
features

# COMMAND ----------

def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    #n_features = len(rows[0]) - 1  # number of columns

    ### TO DO ###
    
    ### TO DO ###

    return best_gain, best_question

# COMMAND ----------

#######
# Demo:
# Find the best question to ask first for our dataset.
best_gain, best_question = find_best_split(training_data)
best_question

# COMMAND ----------

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)

# COMMAND ----------

class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# COMMAND ----------

def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 
      1) Believe that it works. 
      2) Start by checking. for the base case (no further information gain).
      3) Prepare for giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.

    ### INSERT CODE HERE, 1-LINER ###
    
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    
    ### INSERT CODE HERE, IF STATEMENT, 2-LINER ###
      
    # If we reach here, we have found a useful feature / value
    # to partition on.
    
    ### CREATE THE PARTITION OF TRUE AND FALSE ROWS ###
    
    # Recursively build the true branch.
    
    ### MAKE A CALL TO THIS SAME DEF WITH TRUE ROWS ###
    
    # Recursively build the false branch.
    
    ### MAKE A CALL TO THIS SAME DEF WITH FALSE ROWS ###
    
    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)

# COMMAND ----------

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

# COMMAND ----------

my_tree = build_tree(training_data)

# COMMAND ----------

print_tree(my_tree)

# COMMAND ----------

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

# COMMAND ----------

#######
# Demo:
# The tree predicts the 1st row of our
# training data is an Iris-setosa.
classify(training_data[10], my_tree)
#######

# COMMAND ----------

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl, count in counts.items():
        probs[lbl] = str(int(count / total * 100)) + "%"
    return probs

# COMMAND ----------

#######
# Demo:
# Printing that a bit nicer
print_leaf(classify(training_data[2], my_tree))
#######

# COMMAND ----------

#######
# Demo:
# Printing that a bit nicer
print_leaf(classify(training_data[50], my_tree))
#######

# COMMAND ----------

#######
# Demo: Now with Unseen Data
print_leaf(classify(test_data[7], my_tree))
#######

# COMMAND ----------


