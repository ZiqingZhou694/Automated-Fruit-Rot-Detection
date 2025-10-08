Project title: Distributed Machine Learning System for Automated Fruit Rot Detection



Team members with LSU email addresses: 

Aidan Eiler - aeiler3@lsu.edu

Ziqing Zhou - zzhou24@lsu.edu

Jacob Rogers - jroge79@lsu.edu



Dataset(s) you plan to use with their URLs: Fruits fresh and rotten for classification: https:

//www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification/code



Big data framework(s) you plan to use: Hadoop MapReduce, Apache Spark



Short description (theme) of your application (in no more than five sentences): This project aims to identify rotten fruits using computer vision. We plan on using Hadoop ManReduce to extract features from the fruit images dataset and use Apache Spark train a neutral network to classify fruits as fresh/rotten. To enable scalability, fault tolerance, and parallelization, we intend to use a distributed computation architecture.

## ⚙️ Environment Setup (Windows PowerShell)

Before running any script, you must set PySpark to use your Python 3.11 interpreter:

```powershell
$env:PYSPARK_PYTHON = (py -3.11 -c "import sys; print(sys.executable)")
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
```

## 🧩 Steps to Run

### Manifest csv

```powershell
py code\manifest.py
```

### Feature Extraction

Generate image feature data from your dataset:

```powershell
py code\extract_features.py
```
### Model Training

Train a Random Forest classifier on extracted features:

 ```powershell
 py code\model.py
```

### Test Model on a Single Image

You can use either a local file or an online image URL.

Example (local file):
 ```powershell
py code\model_test.py "local image"
```

Example (image URL):
 ```powershell
py code\model_test.py "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg"
```