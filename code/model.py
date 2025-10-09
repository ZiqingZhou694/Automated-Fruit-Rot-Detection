""" @Author(s): Aidan Eiler, Ziqing Zhou
 @Created: 10/6/2025
 @Modified: 10/8/2025

 Purpose:
     This program creates a machine learning model that uses the random forest algorithm to classify images of fruit as either fresh or rotten.
     Training uses the pre-extracted features stored in the Parquet dataset and produces a saved model for later use. The model is evaluated on a
     test set and various performance metrics are printed.

 Input:
     ./out/features_parquet/split=train - parquet dataset for training
     ./out/features_parquet/split=test - parquet dataset for evaluting the accuracy of the trained model
 Output:
     ./out/model - saved trained model"""

from pyspark.sql import SparkSession, functions as F
# for random forest algorithm
from pyspark.ml.classification import RandomForestClassifier
# for converting string labels to numeric
from pyspark.ml.feature import StringIndexer
# for evaluating model performance
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# for creating ML pipelines
from pyspark.ml import Pipeline
# for handling feature vectors
from pyspark.ml.linalg import Vectors, VectorUDT
import sys

TRAINING_DATASET_PATH = "./out/features_parquet/split=train"
TESTING_DATASET_PATH = "./out/features_parquet/split=test"
MODEL_STORAGE_PATH = "./out/model"


def log(msg):  # printf but with more style
    print(f"\n=== {msg} ===")


# initialize spark session, using extract_features.py as a template
log("initializing spark...")
spark = (
    SparkSession.builder
    .appName("model_training")
    .master("local[*]")  # Use all local cores
    .config("spark.driver.memory", "6g")  # Driver memory
    .config("spark.sql.shuffle.partitions", "8")  # Reduce shuffle partitions
    .config("spark.hadoop.hadoop.home.dir", "C:/hadoop")
    .config("spark.hadoop.io.native.lib.available", "false")
    # Force PySpark to use the same Python interpreter as current process
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)

# load training and test data (parquet)
log("loading training features...")
try:
    train_dataset = spark.read.parquet(TRAINING_DATASET_PATH)
    train_count = train_dataset.count()
    log(f"training samples: {train_count}")

    # logging label distribution. good for spotting imbalanced datasets
    print("Training label distribution:")
    train_dataset.groupBy("label").count().show()

except Exception as e:
    print(f"ERROR: failed to load training data: {e}")
    spark.stop()
    sys.exit(1)

log("loading testing features...")
try:
    test_dataset = spark.read.parquet(TESTING_DATASET_PATH)
    test_count = test_dataset.count()
    log(f"test samples: {test_count}")

    # logging label distribution. good for spotting imbalanced datasets
    print("Test label distribution:")
    test_dataset.groupBy("label").count().show()

except Exception as e:
    print(f"ERROR: failed to load test data: {e}")
    spark.stop()
    sys.exit(1)

# spark yells at us if the features column is not the right type. here we need to convert the array<float> type to VectorUDT type
log("converting features to vector format...")


# using this function because our user defined function (udf) needs a function and lamba is harder to read
def array_to_vector_func(arr):
    return Vectors.dense(arr)


# user defined function for converting features to vector form
array_to_vector = F.udf(array_to_vector_func, VectorUDT())

# apply function to datasets
train_dataset = train_dataset.withColumn(
    "features", array_to_vector(F.col("features")))
test_dataset = test_dataset.withColumn(
    "features", array_to_vector(F.col("features")))

# validate features column exists
log("validating features...")
if "features" not in train_dataset.columns:
    print("ERROR: 'features' column not found in training data")
    spark.stop()
    sys.exit(1)

# check feature vector size (should be 128 because 32 historgrams with 32 bins)
sample_feature = train_dataset.select("features").first()[0]
feature_size = len(sample_feature)
log(f"feature vector size: {feature_size}")

if feature_size != 128:
    print(f"WARNING: expected 128 features, got {feature_size}")

# set up the machine learning pipeline, this is where we define the model and how it works. the actual training will be done later
log("building ML pipeline...")

# Convert string labels (fresh/rotten) to numeric indices (0/1) which we need for MLlib
indexer = StringIndexer(
    inputCol="label", outputCol="true_value", handleInvalid="keep")


random_forest = RandomForestClassifier(
    featuresCol="features",
    labelCol="true_value",
    predictionCol="prediction",
    numTrees=100,  # number of decision trees
    maxDepth=10,  # maximum depth of each tree, if this is "too big" the model may overfit
    seed=0  # arbitrary seed for reproducibility
)

pipeline = Pipeline(stages=[indexer, random_forest])


# ===== Fine-tuning: (TrainValidationSplit) =====
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="true_value",
    predictionCol="prediction",
    metricName="accuracy"
)


paramGrid = (ParamGridBuilder()
    .addGrid(random_forest.numTrees, [100, 200])
    .addGrid(random_forest.maxDepth, [6, 8, 10])      
    .addGrid(random_forest.minInstancesPerNode, [1, 5, 10])
    .addGrid(random_forest.maxBins, [32, 64])
    .build())

tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    trainRatio=0.8,    # Split training data into 80% train / 20% validation
    parallelism=2
)

log("grid search training (RF)…")
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()
train_dataset.count()
test_dataset.count()  # Force caching
tvs_model = tvs.fit(train_dataset)
# cv_model = cv.fit(train_dataset)

# Retrieve best model
best_model = tvs_model.bestModel
rf_model = best_model.stages[1]  # RF model

# numTrees is usually an integer attribute; if missing, fall back to the tree list length
if hasattr(rf_model, "numTrees"):
    num_trees = rf_model.numTrees
else:
    num_trees = len(getattr(rf_model, "trees", []))

# Get parameters safely using getOrDefault (in case they are not defined)
max_depth = rf_model.getOrDefault(rf_model.maxDepth) if hasattr(rf_model, "maxDepth") else None
max_bins  = rf_model.getOrDefault(rf_model.maxBins)  if hasattr(rf_model, "maxBins") else None

print("\n=== Best Params (RF) ===")
print("numTrees:", num_trees)
print("maxDepth:", max_depth)
print("maxBins :", max_bins)

# Evaluate best model on test data
log("evaluating best model on test set…")
pred = best_model.transform(test_dataset)

acc_eval = MulticlassClassificationEvaluator(
    labelCol="true_value", predictionCol="prediction", metricName="accuracy")
precision_eval = MulticlassClassificationEvaluator(
    labelCol="true_value", predictionCol="prediction", metricName="weightedPrecision")
recall_eval = MulticlassClassificationEvaluator(
    labelCol="true_value", predictionCol="prediction", metricName="weightedRecall")
f1_eval = MulticlassClassificationEvaluator(
    labelCol="true_value", predictionCol="prediction", metricName="f1")

accuracy  = acc_eval.evaluate(pred)
precision = precision_eval.evaluate(pred)
recall    = recall_eval.evaluate(pred)
f1        = f1_eval.evaluate(pred)

print(f"\nTest Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

log("confusion matrix")
pred.groupBy("true_value", "prediction").count().orderBy("true_value","prediction").show()

# Save model and label mapping for later inference
log("saving best model…")
best_model.write().overwrite().save(MODEL_STORAGE_PATH)

import json, os
si_model = best_model.stages[0]      # StringIndexerModel
labels = si_model.labels             # Class label order
with open(os.path.join(MODEL_STORAGE_PATH, "labels.json"), "w") as f:
    json.dump(labels, f)

# Print summary statistics
log("TRAINING SUMMARY")
print(f"Training samples     : {train_count}")
print(f"Test samples         : {test_count}")
print(f"Feature dimensions   : {feature_size}")
print(f"Best RF -> numTrees  : {num_trees}")
print(f"Best RF -> maxDepth  : {max_depth}")
print(f"Best RF -> maxBins   : {max_bins}")
print(f"Test Accuracy        : {accuracy*100:.2f}%")
print(f"F1 Score             : {f1:.4f}")
print(f"Model saved          : {MODEL_STORAGE_PATH}")
print("Label order:", labels)

spark.stop()
print("\nPROGRAM COMPLETE")
