import re, string, unicodedata
import nltk
#import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

# def replace_contractions(text):
#     """Replace contractions in string of text"""
#     return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

# def preprocess(sample):
#     sample = remove_URL(sample)
#     sample = replace_contractions(sample)
#     # Tokenize
#     words = nltk.word_tokenize(sample)

#     # Normalize
#     return normalize(words)


# if __name__ == "__main__":
#     sample = "Blood test for Down's syndrome hailed http://bbc.in/1BO3eWQ"               
    
#     sample = remove_URL(sample)
#     sample = replace_contractions(sample)

#     # Tokenize
#     words = nltk.word_tokenize(sample)
#     print(words)

#     # Normalize
#     words = normalize(words)
#     print(words)

def NLPpipeLine(df_data,col1,col2):
    #Tokenize
    df_data[col1] = df_data[col1].apply(nltk.word_tokenize)
    df_data[col2] = df_data[col2].apply(nltk.word_tokenize)
    
    # Normalize
    #words = normalize(words)
    df_data[col1] = df_data[col1].apply(normalize)
    df_data[col2] = df_data[col2].apply(normalize)
    print(df_data['req1'].head())
    input("hit enter")

    #print(words)

    return df_data


def Test():
    # load CSV
    df_data = pd.read_csv("Requires_enhancement_2_enhancementData.csv")
    #print(df_data['req1'].head(5))
    df_data = NLPpipeLine(df_data,'req1','req2')
    winsound.Beep(frequency, duration)
    df_data.to_csv("Processed_Requires_enhancement_2_enhancementData.csv")

import os
def processAllFile():
    entries = os.listdir('Data/')
    print(entries)
    df_data = pd.read_csv("Data/"+entries[0])
    print(df_data['req1'].head(5))

    input("hit enter")
    for filename in entries:
        print("Processing ", filename)
        # load CSV
        df_data = pd.read_csv("Data/"+filename)
        #print(df_data['req1'].head(5))
        df_data = NLPpipeLine(df_data,'req1','req2')
        df_data.to_csv("Processed"+filename)
        print("Done")
        winsound.Beep(frequency, duration)
    #print(entries)

def processAllFileListLines():
    entries = os.listdir('Data/')
    print(entries)
    size = 0
    df_data = pd.read_csv("Processed_Requires_enhancement_2_enhancementData.csv")
    print(len(df_data))
    for i in entries:
        df_data = pd.read_csv("Data/"+i)
        print(i,len(df_data))
        size = size +len(df_data)
    print(size)
    

processAllFileListLines()


#Test()
#processAllFile()

