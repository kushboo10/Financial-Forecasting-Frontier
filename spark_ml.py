import os

# WINDOWS PYTHON PATH
os.environ["PYSPARK_PYTHON"] = r"C:\Users\kushb\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\kushb\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# CREATE SPARK SESSION
spark = SparkSession.builder \
    .appName("SparkMLProject") \
    .getOrCreate()

df = spark.read.csv(
    r"C:\Users\kushb\Banking_Project\data\bank.csv",
    header=True,
    inferSchema=True
)

# SHOW DATA
df.show(5)

# ======================================================
# 13. MACHINE LEARNING WITH SPARK ML
# ======================================================

print("\nMACHINE LEARNING SECTION")

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# CREATE LABEL COLUMN
df = df.withColumn(
    "label",
    when(col("y") == "yes", 1).otherwise(0)
)

# CATEGORICAL COLUMNS
categorical_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

# STRING INDEXERS
indexers = [
    StringIndexer(
        inputCol=col_name,
        outputCol=col_name + "_index",
        handleInvalid="keep"
    )
    for col_name in categorical_cols
]

# ONE HOT ENCODERS
encoders = [
    OneHotEncoder(
        inputCol=col_name + "_index",
        outputCol=col_name + "_vec"
    )
    for col_name in categorical_cols
]

# NUMERIC COLUMNS
numeric_cols = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

# FEATURE COLUMNS
feature_cols = [c + "_vec" for c in categorical_cols] + numeric_cols

# VECTOR ASSEMBLER
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# LOGISTIC REGRESSION MODEL
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)

# PIPELINE
pipeline = Pipeline(
    stages=indexers + encoders + [assembler, lr]
)

# TRAIN TEST SPLIT
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# TRAIN MODEL
model = pipeline.fit(train_data)

# PREDICTIONS
predictions = model.transform(test_data)

print("\nPREDICTIONS")
predictions.select(
    "label",
    "prediction",
    "probability"
).show(10)

# MODEL EVALUATION
evaluator = BinaryClassificationEvaluator(
    labelCol="label"
)

auc = evaluator.evaluate(predictions)

print("\nAUC SCORE:", auc)

# ======================================================
# HYPERPARAMETER TUNING
# ======================================================

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

cv_model = crossval.fit(train_data)

best_model = cv_model.bestModel

final_predictions = best_model.transform(test_data)

final_auc = evaluator.evaluate(final_predictions)

print("\nFINAL TUNED AUC:", final_auc)

# ======================================================
# FEATURE IMPORTANCE / COEFFICIENTS
# ======================================================

lr_model = best_model.stages[-1]

print("\nMODEL COEFFICIENTS")
print(lr_model.coefficients)

print("\nMODEL INTERCEPT")
print(lr_model.intercept)

# ======================================================
# MODEL PERFORMANCE SUMMARY
# ======================================================

print("\nMODEL PERFORMANCE SUMMARY")
print("Initial AUC:", auc)
print("Tuned AUC:", final_auc)

if final_auc > 0.85:
    print("Excellent model performance")
elif final_auc > 0.75:
    print("Good model performance")
else:
    print("Model needs improvement")

# STOP SPARK SESSION
spark.stop()