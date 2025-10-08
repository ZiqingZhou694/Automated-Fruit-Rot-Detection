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

Fresh:
 ```powershell
py code\model_test.py "https://i5.walmartimages.com/seo/Fresh-Envy-Apples-Each_32451a10-0563-426a-9a16-a8865b2c3774_3.b3be01fcc4c956f51fe3890589897d31.jpeg"
```

 ```powershell
py code\model_test.py "https://tiimg.tistatic.com/fp/1/007/291/natural-fresh-orange-fruit-411.jpg"
```

```powershell
py code\model_test.py "https://s3.amazonaws.com/grocery-project/product_images/clementine-6762922-2.jpeg"
```
```powershell
py code\model_test.py "https://i5.walmartimages.com/seo/Fresh-Banana-Each_5939a6fa-a0d6-431c-88c6-b4f21608e4be.f7cd0cc487761d74c69b7731493c1581.jpeg"
```

Rotten:
 ```powershell
py code\model_test.py "https://i1.sndcdn.com/artworks-9MBM0YJZia5Kb4OS-yGmyxw-t500x500.jpg"
```

 ```powershell
py code\model_test.py "https://images.stockcake.com/public/b/3/9/b39ce5ab-e625-4a6b-954e-754e0ab08dab_large/rotten-red-apple-stockcake.jpg"
```

 ```powershell
py code\model_test.py "https://media.istockphoto.com/id/520613602/photo/rotten-and-moldy-orange.jpg?s=612x612&w=0&k=20&c=NTg0uiZakxLhSbNSAmK7jPm4sdhNzSQ412gPSG5gxDA="
```
