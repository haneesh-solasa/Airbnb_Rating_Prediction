# Airbnb_Rating_Prediction
Predicting ratings for Airbnb for Dublin, Leinster, Ireland (NLP)

Dataset : https://drive.google.com/drive/folders/1AhBIcV0kkkyUcV9Uvr9wJlW_AQMK9EEZ

Steps to execute the code:

1) Reviews_Translate: Some of the reviews are in different language. So translate them to english using Langdetect and Google Trans library.
2) Reviews_Clean: Cleaned the different symbols in the reviews file.
3) Listings_Clean: Cleaned Nan values, Encode Categorical Values, Scaled Numerical Values and finally performed TF-IDF on the descriptive columns after merging both the dataframes.(Description, Neighborhood overview and Comments).
4) K-NN Crossvalidation: Performed Crossvalidation on the model to get the best K value.
5) Ridge Regression & KNN: Performed Crossvalidation on Ridge Regression. Trained the Ridge regression and K-NN models for predicting all the 7 ratings.

