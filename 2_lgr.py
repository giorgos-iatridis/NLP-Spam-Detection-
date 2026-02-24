import pandas as pd
import numpy as np
from config import final_dfTrain, final_dfTest, my_stop_words
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report


#-----------NOTES-----------
# In this file, we try a combination of TF-IDF for text vectorization and Logistic Regression for classification.
# We also implement hyperparameter tuning using GridSearchCV to optimize the Logistic Regression model.
# Then, we are going to take the best parameters and train the model again in another file
# to evaluate it, extract feature importance and save the model for future use.


s = time.time()

# Load datasets
dfTrain = pd.read_csv(final_dfTrain)
print(dfTrain.head())

#duplicates in train set
print(f"Number of duplicates in train set: {dfTrain.duplicated().sum()}")

if dfTrain.duplicated().sum() > 0:
    dfTrain = dfTrain.drop_duplicates()


#duplicates in test set
dfTest = pd.read_csv(final_dfTest)
print(f"Number of duplicates in test set: {dfTest.duplicated().sum()}")

if dfTest.duplicated().sum() > 0:
    dfTest = dfTest.drop_duplicates()


X_train = dfTrain.drop(columns=['exclamation_ratio', 'has_money', 'source', 'label'], axis=1) #we drop 'source' as we observe from EDA that nazario and nigerian_fraud datasets have only spam emails which may lead to data leakage
y_train = dfTrain['label']

X_test = dfTest.drop(columns=['exclamation_ratio', 'has_money', 'source', 'label'], axis=1)
y_test = dfTest['label']


# ****************************************************************

# Tasks 3, 4 & 5: TF-IDF, Logistic Regression Model and Evaluation

# ****************************************************************

scaler_cols = ['text_length', 'caps_ratio', 'question_ratio', 'digits_ratio',
               'email_phone', 'day_of_week', 'hour', 'urls']


preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words=my_stop_words, 
                                 max_features=5000, 
                                 sublinear_tf=True,
                                 min_df=5,
                                 max_df=0.8), 'cleaned_text'),
        ('scale', StandardScaler(), scaler_cols)
    ]
)

#we preprocess the data here once to save time during GridSearchCV
# we do have some data leakage here, because we will use this preprocessed data in GridSearchCV
# but since we are only tuning hyperparameters and not training the final model here,
# the impact should be minimal.


print("Preprocessing data (TF-IDF & Scaling)...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

#lgr object
lgr = LogisticRegression(max_iter=3000, random_state=42)

param_grid = [
    #first set for liblinear (supports both l1 and l2)
    {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1.0, 5.0]
    },
    #lbfgs-l2
    {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.1, 1.0, 5.0]
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=lgr,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

print("\nStarting Grid Search for Hyperparameter Tuning...\n")
grid_search.fit(X_train_processed, y_train)
print(f"\nBest Hyperparameters: {grid_search.best_params_}\n")
best_model = grid_search.best_estimator_



#TRAINING
print("\n**TRAINING** Set Evaluation Metrics:\n")
y_train_pred = best_model.predict(X_train_processed) #Evaluate on training set to check for overfitting
print(classification_report(y_train, y_train_pred, digits=3))
print("\n" + "-" * 50)


#TEST DATASET
print("\n**TEST** Set Evaluation Metrics:\n")
y_test_pred = best_model.predict(X_test_processed)
print(classification_report(y_test, y_test_pred, digits=3))
print("\n" + "-" * 50)


e = time.time()
print(f"\nTotal Execution Time: {e - s:.2f} seconds\n")