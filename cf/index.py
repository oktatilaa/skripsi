import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import surprise
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset
from surprise import KNNBasic

pd.read_csv(“dataset”) 

# Deskripsikan dataset
rating.describe()

# Lihat informasi dataset
rating.info()

primary_cat = rating['primaryCategories'].value_counts()
primary_cat

# Visualisasikan kolom primaryCategories
dims = (10, 8)
fig, ax = plt.subplots(figsize=dims)
ax = sns.countplot(x=rating.primaryCategories)

# Ubah nama kolomnya agar lebih mudah 
rating.rename(columns={"reviews.rating":"reviewsRating"},inplace=True)

# Jadikan sebagai variabel dan lakukan perhitungan sejumlah nilai yang ada di kolom reviewRating 
rev_rating = rating['reviewsRating'].value_counts()
rev_rating

# Visualisasikan dari hasil perhitungan nilai
dims = (10, 8)
fig, ax = plt.subplots(figsize=dims)
ax = sns.countplot(x="reviewsRating", data=rating)


dims = (10, 8)
fig, ax = plt.subplots(figsize=dims)
ax = sns.countplot(x="primaryCategories",hue="reviewsRecommend", data=rating)


dims = (10, 8)
fig, ax = plt.subplots(figsize=dims)
ax = sns.boxplot(x="reviewsRating", y="primaryCategories", hue="reviewsRecommend", data=rating)

# Cek apakah ada yang null pada dataset
rating.isnull().sum()

reader = Reader()
data = Dataset.load_from_df(rating[['reviews.username','id','reviewsRating']], reader)

# Train & Test
trainset, testset = train_test_split(data, test_size=0.20, random_state=50)

# Gunakan fungsi svd() yang sudah disediakan pada surprise
algo_svd = SVD()
prediction_mf = algo_svd.fit(trainset).test(testset)

# Prediksi
prediction_mf

# Tes rekomendasinya
recom_svd = algo_svd.predict(uid='Jays',iid='AWMjT0WguC1rwyj_rFh3')
recom_svd

sim_options = {'name': 'pearson_baseline','shrinkage': 0}
algo = KNNBasic(sim_options=sim_options)
algo_knn = KNNBasic(k=50, sim_options=sim_options)
prediction_knn = algo_knn.fit(trainset).test(testset)

# Prediksi
prediction_knn

# Tes rekomendasinya
recom_knn = algo_knn.predict(uid='Jays',iid='AWMjT0WguC1rwyj_rFh3')
recom_knn


accuracy.mae(prediction_mf)
accuracy.fcp(prediction_mf)
accuracy.rmse(prediction_mf)


accuracy.mae(prediction_knn)
accuracy.fcp(prediction_knn)
accuracy.rmse(prediction_knn)


# Dataset yang akan dipakai untuk train test split dengan framework surprise
rating[['reviews.username','id','reviewsRating']]