# model.py
# Purpose:
#   Train a RandomForest model on image features stored in HDFS Parquet,
#   evaluate it on a test set, and save the model + label mapping back to HDFS.
#   Uses Spark MLlib for distributed training and evaluation.

from pyspark.sql import SparkSession, Row, functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import sys

# =============================
# CONFIGURATION
# =============================
HDFS_BASE = "hdfs://localhost:9000"
# TRAINING_DATASET_PATH = f"{HDFS_BASE}/out/features_parquet/split=train"
TESTING_DATASET_PATH  = f"{HDFS_BASE}/out/features_parquet/test"
MODEL_STORAGE_PATH    = f"{HDFS_BASE}/out/model"

# =============================
# UTILITY LOG FUNCTION
# =============================
def log(msg):
    print(f"\n=== {msg} ===")

# =============================
# 1. Initialize Spark Session
# =============================
log("initializing Spark...")
spark = (
    SparkSession.builder
    .appName("RandomForest_HDFS")
    .config("spark.hadoop.fs.defaultFS", HDFS_BASE)
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)

# =============================
# 2. Load training and test data, training data omitted for testing purposes
# =============================
# log("loading training dataset from HDFS...")
# train_df = spark.read.parquet(TRAINING_DATASET_PATH)
# train_count = train_df.count()
# log(f"training samples: {train_count}")
# train_df.groupBy("label").count().show()

log("loading test dataset from HDFS...")
test_df = spark.read.parquet(TESTING_DATASET_PATH)
test_count = test_df.count()
log(f"test samples: {test_count}")
test_df.groupBy("label").count().show()

# =============================
# 3. Convert array<float> to VectorUDT
# =============================
log("converting features to VectorUDT...")
array_to_vector = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())
# train_df = train_df.withColumn("features", array_to_vector(F.col("features")))
test_df  = test_df.withColumn("features", array_to_vector(F.col("features")))

# =============================
# 4. Build ML pipeline
# =============================
log("building ML pipeline...")
indexer = StringIndexer(inputCol="label", outputCol="true_value", handleInvalid="keep")
rf = RandomForestClassifier(featuresCol="features", labelCol="true_value", predictionCol="prediction", seed=0)
pipeline = Pipeline(stages=[indexer, rf])

# =============================
# 5. TrainValidationSplit + Grid Search
# =============================
log("setting up grid search...")
evaluator = MulticlassClassificationEvaluator(labelCol="true_value", predictionCol="prediction", metricName="accuracy")
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [100, 200])
             .addGrid(rf.maxDepth, [8, 10])
             .addGrid(rf.maxBins, [32, 64])
             .build())

tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    trainRatio=0.8,
    parallelism=2
)

log("caching datasets...")
# train_df = train_df.cache()
test_df  = test_df.cache()
# train_df.count(); 
test_df.count()  # force caching

log("starting grid search training...")
tvs_model = tvs.fit(test_df)
best_model = tvs_model.bestModel
rf_model = best_model.stages[1]

# =============================
# 6. Print best RF parameters
# =============================
num_trees = getattr(rf_model, "numTrees", len(getattr(rf_model, "trees", [])))
max_depth = rf_model.getOrDefault(rf_model.maxDepth)
max_bins  = rf_model.getOrDefault(rf_model.maxBins)
print("\n=== Best RF Params ===")
print(f"numTrees: {num_trees}, maxDepth: {max_depth}, maxBins: {max_bins}")

# =============================
# 7. Evaluate on test set
# =============================
log("evaluating model on test set...")
pred = best_model.transform(test_df)
accuracy  = evaluator.evaluate(pred)
precision = MulticlassClassificationEvaluator(labelCol="true_value", predictionCol="prediction", metricName="weightedPrecision").evaluate(pred)
recall    = MulticlassClassificationEvaluator(labelCol="true_value", predictionCol="prediction", metricName="weightedRecall").evaluate(pred)
f1        = MulticlassClassificationEvaluator(labelCol="true_value", predictionCol="prediction", metricName="f1").evaluate(pred)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

log("confusion matrix:")
pred.groupBy("true_value","prediction").count().orderBy("true_value","prediction").show()

# =============================
# 8. Save model + labels to HDFS
# =============================
log("saving model to HDFS...")
best_model.write().overwrite().save(MODEL_STORAGE_PATH)

# Save StringIndexer labels.json to HDFS
log("saving labels.json to HDFS...")
si_model = best_model.stages[0]
labels = si_model.labels
# Convert labels list to a DataFrame
labels_rdd = spark.sparkContext.parallelize(labels).map(lambda l: Row(label=l))
labels_df = spark.createDataFrame(labels_rdd)

# Save to HDFS as text
labels_df.write.mode("overwrite").text(f"{MODEL_STORAGE_PATH}/labels.txt")

log("TRAINING SUMMARY")
# print(f"Training samples   : {train_count}")
print(f"Test samples       : {test_count}")
print(f"Feature dimensions : {len(test_df.select('features').first()[0])}")
print(f"Test Accuracy      : {accuracy*100:.2f}%")
print(f"Model saved        : {MODEL_STORAGE_PATH}")

spark.stop()
print("\nPROGRAM COMPLETE")
