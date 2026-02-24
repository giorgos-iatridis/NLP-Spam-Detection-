import os

# Define base path
path_BASE = os.path.dirname(os.path.abspath(__file__))

# Define paths for raw, cleaned and preprocessed data
path_raw = os.path.join(path_BASE, "dataset", "raw")
path_preprocessed = os.path.join(path_BASE, "dataset", "preprocessed")

#RAW FILES
ceas = os.path.join(path_raw, "ceas.csv")
nazario = os.path.join(path_raw, "nazario.csv")
nigerian_fraud = os.path.join(path_raw, "nigerian_fraud.csv")
spam_assasin = os.path.join(path_raw, "spam_assasin.csv")

full_df = os.path.join(path_raw, "full_df.csv")


#PREPROCESSED FILES
final_dfTrain = os.path.join(path_preprocessed, "emails_train.csv")
final_dfTest = os.path.join(path_preprocessed, "emails_test.csv")

google_news_vectors_path = os.path.join(path_BASE, "google", "GoogleNews-vectors-negative300.bin.gz")

#words that may bias the models
my_stop_words = [
    'joseurl', 'ilug', 'ierant', 'opensuse', 'write',
    'date', 'tony', 'february', '2008', '2007', 'uai', 'spamexpert'
]

MODELS_DIR = os.path.join(path_BASE, "saved_models")

lgr_model_path = os.path.join(MODELS_DIR, "lgr_model.joblib")
xgb_model_path = os.path.join(MODELS_DIR, "xgboost_model.joblib")
xgb_json_model_path = os.path.join(MODELS_DIR, "xgboost_model.json")