# NLP Spam Detection Pipeline

An end-to-end Natural Language Processing (NLP) project for spam classification. This repository contains the complete pipeline, from data extraction and exploratory data analysis (EDA) to model training, hyperparameter tuning, and evaluation. 

The project compares a traditional machine learning approach (TF-IDF + Logistic Regression) against a tree-based ensemble method with dense embeddings (Word2Vec + XGBoost).

## 📂 Repository Structure

**It's very important to recreate the exact stracture that I present in `execution_details.txt` file. Without this format, the paths in the `config.py` are not going to work.**

The codebase is organized sequentially to reflect the machine learning lifecycle:

* **`0_ETL.py`**: Data extraction, text cleaning, tokenization, and preprocessing.
* **`1_EDA.ipynb`**: Exploratory Data Analysis, visualizing word frequencies, message lengths, and data distributions.
* **`2_lgr.py`**: Baseline Logistic Regression model training using TF-IDF vectorization.
* **`3_best_LgR.py`**: Hyperparameter tuning for the Logistic Regression model to optimize performance.
* **`4_xgboost.py`**: Baseline XGBoost model training utilizing Word2Vec embeddings.
* **`5_best_xgb.py`**: Hyperparameter tuning for the XGBoost model.
* **`6_metrics.ipynb`**: Final evaluation notebook comparing the models across key metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
* **`spam_detection.pdf`**: Detailed project report documenting the methodology, decisions, and final results.
* **`execution_details.txt`**: Additional execution instructions.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **NLP & Vectorization:** Scikit-learn (TF-IDF), Gensim/Spacy (Word2Vec)
* **Machine Learning:** Scikit-learn (Logistic Regression), XGBoost
* **Data Analysis:** Pandas, NumPy, Jupyter Notebooks

## 🚀 Execution Flow
To reproduce the results, run the scripts in their numerical order. Start by executing the ETL pipeline (`0_ETL.py`) to generate the cleaned datasets, followed by the model training scripts. Final comparisons can be viewed in the `6_metrics.ipynb` notebook.
