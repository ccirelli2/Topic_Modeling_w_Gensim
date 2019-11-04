'''TOPIC MODELING WITH GENSIM

url2 tutorial       https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
LDA                 LDA’s approach to topic modeling is it considers each document as a 
                    collection of topics in a certain proportion. And each topic as a 
                    collection of keywords, again, in a certain proportion.               
                    Topic:  A topic is nothing but a collection of dominant keywords that 
                    are typical representatives. Just by looking at the keywords, 
                    you can identify what the topic is all about.
                    ** so it doesn't actually give you a topic but a group of colocated key words
                    that represent a topic. The reader must infer from these key words the
                    "meaning of the topic". 
'''

# IMPORT PYTHON PACKAGES ------------------------------------
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Spacy 
import spacy

# Plotting
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
import matplotlib.pyplot as plt

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.ERROR)

# Warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# IMPORT DATASETS ----------------------------------------------

# Stop Words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Import Dataset
''' Columns:   content, target, target_names
    Ex:         "0":"From: lerxst@wam.umd.edu (where's my thing)\n
                Subject: WHAT car is this!?
                Nntp-Posting-Host: rac3.wam.umd.edu\nOrganization: University of Maryland, 
                College Park\nLines: 
                15\n\n I was wondering if anyone out there could enlighten 
                me on this car I saw\nthe other day. It was a 2-door sports car, looked to be 
                from the late 60s\/\nearly 70s. It was called a Bricklin. The doors were really 
                small. In addition,\nthe front bumper was separate from the rest of the body. 
                This is \nall I know. If anyone can tellme a model name, engine specs, years\nof 
                production, where this car is made, history, or whatever info you\nhave on 
                this funky looking car, please e-mail.\n\nThanks,\n- IL\n   
                ---- brought to you by your neighborhood Lerxst ----\n\n\n\n\n"
'''
 
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')


# REMOVE EMAILS AND NEW LINE CHARACTERS  -----------------------------
data = df.content.values.tolist()
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove Single Quotes
data = [re.sub("\'", "", sent) for sent in data]
#pprint(data[:1])



# TOKENIZATION AND WORD CLEANUP -------------------------------------

tokenize_txt    = [gensim.utils.simple_preprocess(str(sentence), deacc=True) for sentence in data]
''' min_count   = Ignore all words and bigrams with total collected count lower than this value.
    threshold   = Represent a score threshold for forming the phrases (see score calc)'''

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] \
            for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(type(data_lemmatized))


# CREATE A DICTION AND CORPUS NEEDED FOR TOPIC MODELING ------------------------------

# Create Dictionary (returns mapping of [word_id, word_frequency], input for IDA model
id2word = corpora.Dictionary(data_lemmatized)

print(id2word)
# Create Corpus
texts = data_lemmatized

# Corpus
corpus=[id2word.doc2bow(text) for text in texts]




# BUILD A TOPIC MODEL -----------------------------------------------------------------
''' Documentation = https://radimrehurek.com/gensim/models/ldamodel.html
    Random:         I think because the initial assignment is random
    Update_every:   Number of documents to be iterated through for each update. 0 for batch 
                    learning and > 1 for online iterative learning. 
    chunksize:      Number of documents to be used in each training chunk
    passes:         Number of passes through the corpus during training
    per_word_topics If true, the model compute a list of topics in order of importane. 

'''
lda_model = gensim.models.ldamodel.LdaModel(    corpus      = corpus, 
                                                id2word     = id2word,
                                                num_topics  = 20, 
                                                random_state= 100, 
                                                update_every= 1, 
                                                chunksize   = 100,
                                                passes      = 10, 
                                                alpha       = 'auto', 
                                                per_word_topics=True)


pprint(lda_model.print_topics())



# VIZUALIZE THE TOPICS - KEYWORDS


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis











