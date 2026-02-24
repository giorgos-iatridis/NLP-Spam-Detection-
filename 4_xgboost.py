import pandas as pd
import numpy as np
import time
from config import final_dfTrain, final_dfTest, google_news_vectors_path, my_stop_words

import xgboost as xgb
from gensim.models import KeyedVectors

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report



s = time.time()

# Load datasets
dfTrain = pd.read_csv(final_dfTrain)
dfTest = pd.read_csv(final_dfTest)


#train_val_split
xTrain=dfTrain.drop(columns=['exclamation_ratio', 'has_money', 'source', 'label'], axis=1)
y_train=dfTrain['label'].values
y_test=dfTest['label'].values


# Load pre-trained Google News word2vec model
print("Loading Word2Vec model...\n")
google = KeyedVectors.load_word2vec_format(google_news_vectors_path, binary=True)

#we transform our stop words to set for speed
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

print("Vectorization complete.")

#GridSearch parameters to check
param_grid = {
    'max_depth': [5],
    'learning_rate': [0.1],
    'n_estimators': [500, 1500],
    'subsample': [0.8],
    'reg_lambda': [20]
}



#____XGBoost Model____

xgb_model = xgb.XGBClassifier(
    tree_method='hist',
    objective='binary:logistic',
    n_jobs=1,
    random_state=42
)


#grid search
print("\nRunning GridSearch with 5-Fold Cross-Validation on Training Set...\n")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=kf,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_final, y_train)
print(f"Best Parameters from GridSearch: {grid_search.best_params_}\n")
xgboost = grid_search.best_estimator_

#TRAINING
y_train_pred = xgboost.predict(X_train_final) #Evaluate on training set to check for overfitting
print("\n**TRAINING** Set Evaluation Metrics:\n")
print(classification_report(y_train, y_train_pred, digits=3))


#TEST DATASET
y_test_pred = xgboost.predict(X_test_final) #Evaluate on test set
print("\n**TEST** Set Evaluation Metrics:\n")
print(classification_report(y_test, y_test_pred, digits=3))

e = time.time()
print(f"time: {(e - s)/60:.2f} minutes")