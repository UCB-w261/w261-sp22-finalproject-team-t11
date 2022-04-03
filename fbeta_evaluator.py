import random
from pyspark.ml.evaluation import Evaluator

class FBetaEvaluator(Evaluator):

    def __init__(self, predictionCol="prediction", labelCol="label", beta=1.0, labelWeighCol=None):
        self.predictionCol = predictionCol
        self.labelCol = labelCol
        self.beta = beta

    def _evaluate(self, dataset):
        # TODO: the col names are hardcoded. Use the parameter instead.
        TP = dataset.filter((dataset.label == 1) & (dataset.prediction == 1)).count()
        TN = dataset.filter((dataset.label == 0) & (dataset.prediction == 0)).count()
        FP = dataset.filter((dataset.label == 0) & (dataset.prediction == 1)).count()
        FN = dataset.filter((dataset.label == 1) & (dataset.prediction == 0)).count()
        
        if (TP+TN+FP+FN) != 0:
            accuracy = (TP + TN) / (TP+TN+FP+FN)
        else:
            accuracy = 0
        if (TP + FP) != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if ((self.beta**2) * precision + recall) != 0:
            f_beta = ((1+self.beta**2) * (precision * recall)) / ((self.beta**2) * precision + recall)
        else:
            f_beta = 0
        if (precision + recall) != 0:
            f_score = (precision * recall) / (precision + recall)
        else:
            f_score = 0
        
        return f_beta

    def isLargerBetter(self):
        return True