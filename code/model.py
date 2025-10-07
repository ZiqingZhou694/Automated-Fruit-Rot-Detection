# @Author(s): Aidan Eiler
# @Created: 10/6/2025
# @Modified: 10/7/2025
#
# Purpose:
#     This program creates a machine learning model that uses the random forest algorithm to classify images of fruit as either fresh or rotten.
#     Training uses the pre-extracted features stored in the Parquet dataset and produces a saved model for later use. The model is evaluated on a
#     test set and various performance metrics are printed.

# Input:
#     ./out/features_parquet/split=train - parquet dataset for training
#     ./out/features_parquet/split=test - parquet dataset for evaluting the accuracy of the trained model
# Output:
#     ./out/model - saved trained model

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

# do the actual training
log("training model...")
try:
    model = pipeline.fit(train_dataset)
    log("training complete!")
except Exception as e:
    print(f"ERROR: training failed: {e}")
    spark.stop()
    sys.exit(1)

# test the model against our testing set
log("testing model against test set...")
try:
    # run predictions against the test set
    model_predictions = model.transform(test_dataset)

    # renaming columns created by spark (MLlib) for clarity
    model_predictions = (model_predictions
                         .withColumnRenamed("prediction", "model_prediction")
                         .withColumnRenamed("labelIndex", "true_value")
                         )

    # this prints out the first 10 predictions
    print("Sample predictions:")
    model_predictions.select("label", "true_value", "model_prediction",
                             "probability").show(10, truncate=False)

except Exception as e:
    print(f"ERROR: prediction failed: {e}")
    spark.stop()
    sys.exit(1)

# evaluate the model using various metrics
# we start off by creating an evaluator, then evaluate, then print the results
log("evaluateing model...")

# overall correctness
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="true_value",
    predictionCol="model_prediction",
    metricName="accuracy"
)
accuracy = accuracy_evaluator.evaluate(model_predictions)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# precision checks for FALSE POSITIVES
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="true_value",
    predictionCol="model_prediction",
    metricName="weightedPrecision"
)
precision = precision_evaluator.evaluate(model_predictions)
print(f"Precision: {precision:.4f}")

# recall checks for FALSE NEGATIVES
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="true_value",
    predictionCol="model_prediction",
    metricName="weightedRecall"
)
recall = recall_evaluator.evaluate(model_predictions)
print(f"Recall: {recall:.4f}")

# f1 score is a combination of precision and recall
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="true_value",
    predictionCol="model_prediction",
    metricName="f1"
)
f1 = f1_evaluator.evaluate(model_predictions)
print(f"F1 Score: {f1:.4f}")

# this prints out the confusion matrix. this creates a matrix for us to see how our model performed. https://en.wikipedia.org/wiki/Confusion_matrix
log("computing confusion matrix...")
confusion = model_predictions.groupBy("true_value", "model_prediction").count()
confusion.orderBy("true_value", "model_prediction").show()

# we want to store our trained model for later use
log("saving model...")
try:
    model.write().overwrite().save(MODEL_STORAGE_PATH)
    print(f"Model saved to: {MODEL_STORAGE_PATH}")
except Exception as e:
    print(f"ERROR: failed to save model: {e}")

# print stats
log("TRAINING SUMMARY")
print(f"Training samples: {train_count}")
print(f"Test samples: {test_count}")
print(f"Feature dimensions: {feature_size}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"Model saved: {MODEL_STORAGE_PATH}")

spark.stop()
print("\nPROGRAM COMPLETE")
