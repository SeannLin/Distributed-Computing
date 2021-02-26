from pyspark.sql import *
from pyspark.ml import *

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline

from builtins import round

from user_definition import *

def train_rf_cv(df_train, df_valid, n_tree, n_fold, n_digits):
    # Create model instance
    rf = RandomForestClassifier(numTrees=n_tree)

    # Set the metric
    evaluator = BinaryClassificationEvaluator()

    # create the cross validator
    cv = CrossValidator()\
        .setEstimator(rf)\
        .setEvaluator(evaluator)\
        .setNumFolds(n_fold)

    # setting for cross validator
    paramGrid = ParamGridBuilder().build()
    cv.setEstimatorParamMaps(paramGrid)

    # fit the model using cross validator
    cv_model = cv.fit(df_train)

    # prediction
    rf_predict = cv_model.bestModel.transform(df_valid)

    return round(evaluator.evaluate(rf_predict), n_digits)

def train_gbt_cv(df_train, df_valid, depth, n_fold, n_digits):
    # Create model instance
    gbt = GBTClassifier(maxDepth=depth)

    # Set the metric
    evaluator = BinaryClassificationEvaluator()

    # create the cross validator
    cv = CrossValidator()\
        .setEstimator(gbt)\
        .setEvaluator(evaluator)\
        .setNumFolds(n_fold)

    # setting for cross validator
    paramGrid = ParamGridBuilder().build()
    cv.setEstimatorParamMaps(paramGrid)

    # fit the model using cross validator
    cv_model = cv.fit(df_train)

    # prediction
    gbt_predict = cv_model.bestModel.transform(df_valid)

    return round(evaluator.evaluate(gbt_predict), n_digits)

ss = SparkSession.builder\
    .config('spark.driver.memory', '16g')\
    .config('spark.executor.memory', '16g')\
    .getOrCreate()

df_train = ss.read.parquet(train_folder)
print(df_train.count())
print()

df_valid = ss.read.parquet(valid_folder)
print(df_valid.count())
print()

best_score = 0
best_num = 0
for num in num_trees:
    score = train_rf_cv(df_train, df_valid, num, n_fold, n_digits)
    if score > best_score:
        best_score = score
        best_num = num

print(RandomForestClassifier.__name__)
print(best_num)
print(best_score)
print()

best_depth = 0
best_score = 0
for d in max_depth:
    score = train_gbt_cv(df_train, df_valid, d, n_fold, n_digits)
    if score > best_score:
        best_score = score
        best_depth = d

print(GBTClassifier.__name__)
print(best_depth)
print(best_score)

ss.stop()
