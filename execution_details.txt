DESCRIPTION
-----------
This folder contains the source code, data, and trained models developed 
within the framework of this project.

DIRECTORY STRUCTURE
-------------------
.
├── dataset/
│   ├── raw/                  # Original, raw data (CEAS, Nazario, etc.)
│   └── preprocessed/          # Processed data (emails_train.csv, emails_test.csv)
│
├── Google/
│   └── GoogleNews-vectors...  # Pre-trained Word2Vec model (.bin.gz file)
                               # Not included due to file size (see Installation Instructions)
│
├── saved_models/              # Folder for storing models (.joblib / .json)
│
├── 0_ETL.py                   # Cleaning and preprocessing code (Extract-Transform-Load)
├── 1_EDA.ipynb                # Jupyter Notebook for Exploratory Data Analysis
├── 2_lgr.py                   # Logistic Regression - TF-IDF, initial code for tuning
├── 3_best_LgR.py              # Training & Evaluation of the optimal Logistic Regression
├── 4_xgboost.py               # XGBoost - Word2Vec
├── 5_best_xgb.py              # Training & Evaluation of the optimal XGBoost
├── 6_metrics.ipynb            # Jupyter Notebook for final comparison
├── config.py                  # Configuration file (paths, stopwords, etc.)
├── requirements.txt           # List of required Python libraries
└── README.txt                 # This instruction file


INSTALLATION & REQUIREMENTS
---------------------------
1. The use of a virtual environment is recommended.
2. Install the libraries:
   pip install -r requirements.txt

3. Download the spaCy language model:
   python -m spacy download en_core_web_sm

4. Google Word2Vec:
   Download the GoogleNews-vectors-negative300.bin.gz file (Google's pre-trained model) 
   from the following link:

   https://github.com/mmihaltz/word2vec-GoogleNews-vectors
   
   and place it in the "Google" folder. The application reads the compressed file 
   directly, so decompression is not required.


EXECUTION ORDER
---------------
The files must be executed in the following order to reproduce the results:

STEP 1: Preprocessing
   Run `0_ETL.py`. (Execution time ~24 minutes, depending on the OS)
   - Reads the raw data.
   - Performs cleaning, NLP processing, and split.
   - Saves `emails_train.csv` and `emails_test.csv` in the `dataset/preprocessed/` folder.

STEP 2: Exploratory Analysis (Optional)
   Open and run `1_EDA.ipynb` to view the charts and statistics 
   described in the project.

STEP 3: Model Training
   
   A. LOGISTIC REGRESSION:
      - `2_lgr.py`: Performs GridSearch to find parameters (execution time ~1 minute).
      - `3_best_LgR.py`: Trains the optimal model directly and displays the results.
        (Execution of `3_best_LgR.py` is recommended for quick verification, execution time ~20 seconds).

   B. XGBOOST:
      - `4_xgboost.py`: Performs GridSearch for XGBoost (execution time ~15 minutes).
      - `5_best_xgb.py`: Trains the optimal XGBoost with Word2Vec and displays the results.
        (Execution of `5_best_xgb.py` is recommended for quick verification, execution time ~5 minutes).

STEP 4: Comparison & Metrics
   Open `6_metrics.ipynb`.
   - Loads the saved models from the `saved_models/` folder.
   - Generates classification reports, confusion matrices, error analysis, and model wins/losses.

NOTES
-----
- The `config.py` file contains all file paths.
