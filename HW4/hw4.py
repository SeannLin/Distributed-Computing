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

def rf_train(df_train, df_valid, num_trees):
    rf = RandomForestClassifier(numTrees=num_trees)
    rf_model = rf.fit(df_train)
    rf_predict = rf_model.transform(df_valid)
    evaluator = BinaryClassificationEvaluator()
    return round(evaluator.evaluate(rf_predict), n_digits)

best_score = 0
best_num = 0
for num in num_trees:
    score = rf_train(df_train, df_valid, num)
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
    gbt = GBTClassifier(maxDepth=d)
    model_gbt = gbt.fit(df_train)
    predict_gbt = model_gbt.transform(df_valid)
    evaluator = BinaryClassificationEvaluator()
    score = round(evaluator.evaluate(predict_gbt), n_digits)
    if score > best_score:
        best_score = score
        best_depth = d

print(GBTClassifier.__name__)
print(best_depth)
print(best_score)

ss.stop()
