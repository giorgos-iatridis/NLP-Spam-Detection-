import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from config import ceas, nazario, nigerian_fraud, spam_assasin, path_preprocessed, path_raw
import spacy
import time
import datetime

def clean_text(text):


    #transform to lowercase
    text = str(text).lower()

    #we remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    #define a regex pattern to identify URLs
    #we could use tldextract library for more comprehensive tld list, but for simplicity we define a basic list here
    tlds = r'(com|org|net|edu|gov|mil|gr|eu|uk|ru|cn|de|fr|it|es|info|biz|xyz|top|online|site|co|io|ai|me|us)'
    
    #we identify links with classic patterns (http, https, www)
    pattern_links = r'https?://\S+|www\.\S+'
    
    #we also identify naked domains (example.com, example.gr)
    pattern_naked = r'[a-z0-9.-]+\.' + tlds + r'\b'
    
    #combine both patterns
    final_url_pattern = f'{pattern_links}|{pattern_naked}'

    # substitute identified URLs with the token 'url'
    text = re.sub(final_url_pattern, 'url', text)

    
    #Remove punctuation and special characters, keeping only alphanumeric characters and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    #substitute multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return the cleaned text
    return text





def extract_features(df):

    print('Feature Engineering Started...')

    #we fill NA texts (if any) with empty strings to avoid errors in the following calculations
    temp_body = df['body'].fillna('').astype(str)

    #column text length
    df['text_length'] = temp_body.str.len()

    #ratio of caps, exclamation marks, question marks per text length
    df['caps_ratio'] = temp_body.str.count(r'[A-Z]') / df['text_length'].replace(0, 1)
    df['exclamation_ratio'] = temp_body.str.count('!') / df['text_length'].replace(0, 1)
    df['question_ratio'] = temp_body.str.count('\?') / df['text_length'].replace(0, 1)

    #boolean column indicating presence of money symbols
    df['has_money'] = temp_body.str.contains(r'[\$\€\£\¥]', regex=True).fillna(False).astype(int)

    #digits per text length
    df['digits_ratio'] = temp_body.str.count(r'\d') / df['text_length'].replace(0, 1)

    #has email/phone boolean
    df['email_phone'] = temp_body.str.contains(r'[\w\.-]+@[\w\.-]+|\+?\d[\d -]{8,}\d', regex=True).astype(int)
    
    ##we fill missing values with empty strings
    df['subject'] = df['subject'].fillna('')
    
    #clean_text function application
    cols_to_clean = ['subject', 'body']
    print('Cleaning text data...', "\n")
    for col in cols_to_clean:
        df[col] = df[col].apply(clean_text)
        
    #split date to hour and day_of_week
    # 0: Monday, 6: Sunday
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    
    #fill missing values in 'hour' and 'day_of_week' if any
    df['hour'] = df['hour'].fillna(-1).astype(int)
    df['day_of_week'] = df['day_of_week'].fillna(-1).astype(int)
    
    #Identify INVALID years
    #if year < 1990 or year > current
    current_year = datetime.datetime.now().year
    mask_invalid = (df['date'].dt.year < 1990) | (df['date'].dt.year > current_year)
    
    # Overwrite hour and day with -1 for these invalid dates
    if mask_invalid.sum() > 0:
        print(f"Warning: Found {mask_invalid.sum()} messages with invalid dates. Setting hour/day to -1.")
        df.loc[mask_invalid, 'hour'] = -1
        df.loc[mask_invalid, 'day_of_week'] = -1


    # Combine Text
    df['cleaned_message'] = df['subject'] + ' ' + df['body']


    new_features = [
        'text_length', 'caps_ratio', 'exclamation_ratio', 
        'question_ratio', 'has_money', 'digits_ratio', 'email_phone'
    ]

    cols_to_keep = new_features + ['day_of_week', 'hour', 'cleaned_message', 'urls', 'label', 'source']

    
        
    return df[cols_to_keep]





def nlp_process(df, model):

    print('Data transformation started', "\n")

    lem_emails = []
    
    # Truncate overly long messages to avoid exceeding spaCy's max length
    df['cleaned_message'] = df['cleaned_message'].astype(str).apply(lambda x: x[:900000] if len(x) > 900000 else x)

    #clean sms messages
    #we keep sentences that include only the routes of every word without stopwords and punctuations
    sentences = model.pipe(df['cleaned_message'], batch_size=2000, disable=["parser", "ner"])

    for sentence in sentences:
        #Lemmatization and removing stopwords and punctuations
        words = [token.lemma_ for token in sentence if not token.is_stop and not token.is_punct and not token.is_space]

        #join the cleaned words back to a single string
        lem_emails.append(" ".join(words))
    
    df['cleaned_text'] = lem_emails

    print(f'Shape of raw_data before removing empty messages: {df.shape}', "\n")
    #after cleaning, maybe some messages are empty,so we secure that we remove these rows
    df['cleaned_text'] = df['cleaned_text'].replace('', np.nan) #we replace empty cells with NaN
    df = df.dropna(subset=['cleaned_text'])
    print(f'Shape of raw_data after removing empty messages: {df.shape}', "\n")
    df = df.drop(columns=['cleaned_message'], axis=1)



    return df

    

# We need 'if __name__' to be sure that this code runs only when we run this file directly 
# and not when we import it as a module in another file. 
# We do this to save performance, speed and memory, so when we call it as a module in another file, 
# it doesn't run all the code every time.
if __name__ == "__main__": 
    
    s = time.time()

    print('ETL process started', "\n")


    #Reading the train datasets
    ceas = pd.read_csv(ceas)
    ceas['source'] = 'ceas'
    nazario = pd.read_csv(nazario)
    nazario['source'] = 'nazario'
    nigerian_fraud = pd.read_csv(nigerian_fraud)
    nigerian_fraud['source'] = 'nigerian_fraud'
    spam_assasin = pd.read_csv(spam_assasin)
    spam_assasin['source'] = 'spam_assasin'
    

    print('Initial Datasets Loaded', "\n")

    #Info/Describe of initial datasets
    datasets = {'ceas': ceas, 'nazario': nazario, 'nigerian_fraud': nigerian_fraud, 'spam_assasin': spam_assasin}
    for name, dataset in datasets.items():
        print(f'Dataset: ----{name}----')        
        print('Info: \n', dataset.info(), "\n")
        print('Describe: \n', dataset.describe(), "\n")



    #merging them
    full_df = pd.concat([ceas, nazario, nigerian_fraud, spam_assasin], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle the dataset
    full_df = full_df[['date', 'subject', 'body', 'urls', 'label', 'source']]
    full_df.drop_duplicates(subset=['subject', 'body'], inplace=True)
    print('Datasets Merged', "\n")
    full_df.to_csv(path_raw + '/full_df.csv')
    print('Merged Dataset INFO: \n', full_df.info(), '\n')

    #spliting to train and test sets
    dfTrain, dfTest = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['label'])
    dfTrain = dfTrain.copy()
    dfTest = dfTest.copy()

    print('Datasets Split to Train and Test sets', "\n")


    #we create an object of trained spacy's model for english language
    eng_model = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    eng_model.max_length = 2000000
    print('Spacy Model Loaded', "\n")


    #-------------------------------------------------------------------

    # TASK/Question 2.b) of the project:

    #-------------------------------------------------------------------

    dfTrain = extract_features(dfTrain)
    dfTrain = nlp_process(dfTrain, eng_model)

    #from EDA we see that there are some duplicate messages, so we remove them
    dfTrain.drop_duplicates(subset=['cleaned_text'], inplace=True)

    dfTest = extract_features(dfTest)
    dfTest = nlp_process(dfTest, eng_model)
    dfTest.drop_duplicates(subset=['cleaned_text'], inplace=True)

    # Save preprocessed training data
    dfTrain.to_csv(path_preprocessed + "/emails_train.csv", index=False)

    # Save preprocessed testing data
    dfTest.to_csv(path_preprocessed + "/emails_test.csv", index=False)

    e = time.time()

    print(f"Total ETL time: {(e - s)/60} minutes")