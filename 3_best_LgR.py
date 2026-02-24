import pandas as pd
import numpy as np
from config import final_dfTrain, final_dfTest, my_stop_words, lgr_model_path
import time
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report



#---------NOTES-----------
# This file trains and evaluates the Logistic Regression model using TF-IDF for text vectorization,
# with the best hyperparameters obtained from a previous GridSearchCV run in lgr.py.
# it's important to have the best model in a separate file for better organization and speed,
# so that we don't have to run hyperparameter tuning every time we want to evaluate the model.
# It also extracts and saves the top 20 important features based on the model coefficients.



s = time.time()

# Load datasets
dfTrain = pd.read_csv(final_dfTrain)
dfTest = pd.read_csv(final_dfTest)

 #we drop 'source' as we observe from EDA that nazario and nigerian_fraud datasets
 # have only spam emails which may lead to data leakage
 #'exclamation_ratio', 'has_money' from EDA seem to not correlate with the label
X_train = dfTrain.drop(columns=['exclamation_ratio', 'has_money', 'source', 'label'], axis=1)
y_train = dfTrain['label']

X_test = dfTest.drop(columns=['exclamation_ratio', 'has_money', 'source', 'label'], axis=1)
y_test = dfTest['label']



# ****************************************************************

# Tasks 3, 4 & 5: TF-IDF, Logistic Regression Model and Evaluation

# ****************************************************************

scaler_cols = [
    'text_length', 'caps_ratio', 'question_ratio', 'digits_ratio',
    'email_phone', 'day_of_week', 'hour', 'urls'
]

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


lgr = LogisticRegression(
    solver='liblinear',
    penalty='l2',
    C=5.0,
    l1_ratio=0.3,
    max_iter=3000,
    random_state=42
)

pipe_lgr = Pipeline([
    ('preprocessor', preprocessor),
    ('lgr', lgr)
])

print("\nFitting the model on the Training Set...\n")
pipe_lgr.fit(X_train, y_train)


#TEST DATASET
y_test_pred = pipe_lgr.predict(X_test) #Evaluate on test set
print("\n**TEST** Set Evaluation Metrics:\n")
print(classification_report(y_test, y_test_pred, digits=3))
print("\n" + "-" * 50)

e = time.time()
time_lgr = e - s
print(f"\nTotal Execution Time: {time_lgr:.1f} seconds\n")


#____FEATURE IMPORTANCE____

print("\n" + "="*40)
print("TOP FEATURES (Logistic Regression)")
print("="*40)

# Extract feature names and coefficients
feature_names = pipe_lgr.named_steps['preprocessor'].get_feature_names_out()
coeffs = pipe_lgr.named_steps['lgr'].coef_[0]



df_coeffs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coeffs,
})

#top-20 spam
top_spam = df_coeffs.sort_values(by='Coefficient', ascending=False).head(20)
print("\nTOP 20 SPAM FEATURES")
print(top_spam)

#top-20 ham
top_ham = df_coeffs.sort_values(by='Coefficient', ascending=True).head(20)
print("\nTOP 20 HAM FEATURES")
print(top_ham)


top10_spam_ham = pd.concat([top_spam, top_ham])

joblib.dump({
    "model": pipe_lgr,
    "feature_cols": X_train.columns.tolist(),
    "target": "label",
    "time(s)": time_lgr,
    "important_features": top10_spam_ham
}, lgr_model_path)