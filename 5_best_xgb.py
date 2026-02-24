import pandas as pd
import numpy as np
import time
import os
import joblib
from config import final_dfTrain, final_dfTest, google_news_vectors_path, my_stop_words, xgb_json_model_path, xgb_model_path

import xgboost as xgb
from gensim.models import KeyedVectors

from sklearn.metrics import classification_report



s = time.time()

# Load datasets
dfTrain = pd.read_csv(final_dfTrain)
dfTest = pd.read_csv(final_dfTest)


xTrain=dfTrain.drop(columns=['exclamation_ratio', 'has_money', 'source', 'label'], axis=1)
y_train = dfTrain['label'].values

# Load pre-trained Google News word2vec model
google = KeyedVectors.load_word2vec_format(google_news_vectors_path, binary=True)

leakage = set(my_stop_words)

# Function to vectorize text data using the pre-trained model
def vectorize_text(text, model, vector_size=300):

    words = str(text).split()
    valid_word_vectors = [
        model[word] for word in words 
        if word in model and word not in leakage
    ]
    
    if not valid_word_vectors:
        return np.zeros(vector_size)
    
    return np.mean(valid_word_vectors, axis=0)


# ************************************************

# Tasks 3, 4 & 5: Word2Vec, XGBoost and Evaluation

# ************************************************


print("Vectorizing text data and extracting metadata...")

#____TRAIN SET____
vectors_train = np.array([vectorize_text(text, google) for text in xTrain['cleaned_text']])
extra_train = xTrain.drop(columns=['cleaned_text']).values
X_train_final = np.hstack((vectors_train, extra_train))

#____TEST SET____
vectors_test = np.array([vectorize_text(text, google) for text in dfTest['cleaned_text']])
extra_test = dfTest.drop(columns=['exclamation_ratio', 'has_money', 'cleaned_text', 'source', 'label']).values
X_test_final = np.hstack((vectors_test, extra_test))
y_test = dfTest['label'].values

print("Vectorization complete.")


#____XGBoost Model____


xgboost = xgb.XGBClassifier(
    tree_method='hist',
    objective='binary:logistic',
    colsample_bytree=0.7, 
    learning_rate=0.1, 
    max_depth=5,
    n_estimators=1500, 
    subsample=0.8,
    reg_lambda=20,
    n_jobs=1,
    random_state=42,
)


print("\nFitting the model on the Training Set...\n")
xgboost.fit(X_train_final, y_train)


#TEST DATASET
y_test_pred = xgboost.predict(X_test_final) #Evaluate on test set
print("\n**TEST** Set Evaluation Metrics:\n")
print(classification_report(y_test, y_test_pred, digits=3))

e = time.time()
time_xgb = e - s
print(f"time: {time_xgb:.2f} seconds")


#save
xgboost.save_model(xgb_json_model_path)


joblib.dump({
    "model": xgboost,
    "target": "label",
    "time(s)": time_xgb,
}, xgb_model_path)