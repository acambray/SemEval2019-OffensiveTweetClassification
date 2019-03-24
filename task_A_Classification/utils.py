import numpy as np
import re
import string
import html
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
from symspellpy.symspellpy import SymSpell, Verbosity


def process_tweet(tweet,
                  remove_USER_URL=True,
                  remove_punctuation=True,
                  remove_stopwords=True,
                  remove_HTMLentities=True,
                  remove_hashtags=True,
                  appostrophe_handling=True,
                  lemmatize=True,
                  trial=False,
                  sym_spell=None,
                  reduce_lengthenings=True,
                  segment_words=True,
                  correct_spelling=True,
                  ):
    """
    This function receives tweets and returns clean word-list
    """
    ### Handle USERS and URLS ################################################
    if remove_USER_URL:
        if trial:
            tweet = re.sub(r'@\w+ ?', '', tweet)
            tweet = re.sub(r'http\S+', '', tweet)
        else:
            tweet = re.sub(r"@USER", "<>", tweet)
            tweet = re.sub(r"URL", "", tweet)
    else:
        if trial:
            tweet = re.sub(r'@\w+ ?', '<usertoken> ', tweet)
            tweet = re.sub(r'http\S+', '<urltoken> ', tweet)
        else:
            tweet = re.sub(r"@USER", "<usertoken>", tweet)
            tweet = re.sub(r"URL", "<urltoken>", tweet)

    ### Remove HTML Entiti es #################################################
    if remove_HTMLentities:
        tweet = html.unescape(tweet)

    ### REMOVE HASHTAGS? #####################################################
    if remove_hashtags:
        tweet = re.sub(r'#\w+ ?', '', tweet)

    ### Convert to lower case: Hi->hi, MAGA -> maga ##########################
    tweet = tweet.lower()

    ### Cleaning: non-ASCII filtering, some appostrophes, separation #########
    tweet = re.sub(r"’", r"'", tweet)
    tweet = re.sub(r"[^A-Za-z0-9'^,!.\/+-=@]", " ", tweet)
    tweet = re.sub(r"what's", "what is ", tweet)
    tweet = re.sub(r"\'s", " ", tweet)
    tweet = re.sub(r"\'ve", " have ", tweet)
    tweet = re.sub(r"n't", " not ", tweet)
    #     tweet = re.sub(r"i'm", "i am ", tweet)
    tweet = re.sub(r"\'re", " are ", tweet)
    tweet = re.sub(r"\'d", " would ", tweet)
    tweet = re.sub(r"\'ll", " will ", tweet)
    tweet = re.sub(r",", " ", tweet)
    tweet = re.sub(r"\.", " ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\/", " ", tweet)
    tweet = re.sub(r"\^", " ^ ", tweet)
    tweet = re.sub(r"\+", " + ", tweet)
    tweet = re.sub(r"\-", " - ", tweet)
    tweet = re.sub(r"\=", " = ", tweet)
    tweet = re.sub(r"(\d+)(k)", r"\g<1>000", tweet)
    tweet = re.sub(r":", " : ", tweet)
    tweet = re.sub(r" e g ", " eg ", tweet)
    tweet = re.sub(r" b g ", " bg ", tweet)
    tweet = re.sub(r" u s ", " american ", tweet)
    tweet = re.sub(r"\0s", "0", tweet)
    tweet = re.sub(r" 9 11 ", "911", tweet)
    tweet = re.sub(r"e - mail", "email", tweet)
    tweet = re.sub(r"j k", "jk", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)

    ### Remove Punctuation ###################################################
    if remove_punctuation:
        translator = str.maketrans('', '', ''.join(list(set(string.punctuation) - set("'"))))
        tweet = tweet.translate(translator)

    # Tokenize sentence for further word-level processing
    tokenizer = TweetTokenizer()
    words = tokenizer.tokenize(tweet)

    ### Apostrophe handling:    you're   -> you are  ########################
    APPO = {"aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
            "i'd": "I would", "i'll": "I will", "i'm": "I am", "isn't": "is not", "it's": "it is", "it'll": "it will",
            "i've": "I have", "let's": "let us", "mightn't": "might not", "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
            "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are", "they've": "they have", "we'd": "we would", "we're": "we are", "weren't": "were not",
            "we've": "we have", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
            "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "'re": " are",
            "wasn't": "was not", "we'll": " will"}
    if appostrophe_handling:
        words = [APPO[word] if word in APPO else word for word in words]

    tweet = ' '.join(words)
    words = tokenizer.tokenize(tweet)

    ### Lemmatisation:          drinking -> drink ###########################
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word, "v") for word in words]

    ### Remove stop words:      is, that, the, ... ##########################
    if remove_stopwords:
        eng_stopwords = set(stopwords.words("english"))
        words = [w for w in words if not w in eng_stopwords]

    ### Reduce lengthening:    aaaaaaaaah -> aah, bleeeeerh -> bleerh #################
    if reduce_lengthenings:
        pattern = re.compile(r"(.)\1{2,}")
        words = [pattern.sub(r"\1\1", w) for w in words]

    if sym_spell and (segment_words or correct_spelling):

        ### Segment words:    thecatonthemat -> the cat on the mat ####################
        if segment_words:
            words = [sym_spell.word_segmentation(word).corrected_string for word in words]

        ### Correct spelling: birberals -> liberals ######################
        if correct_spelling:
            def correct_spelling_for_word(word):
                suggestions = sym_spell.lookup(word, Verbosity.TOP, 2)

                if len(suggestions) > 0:
                    return suggestions[0].term
                return word

            words = [correct_spelling_for_word(word) for word in words]

    clean_tweet = " ".join(words)
    clean_tweet = re.sub("  ", " ", clean_tweet)
    clean_tweet = clean_tweet.lower()

    return clean_tweet

from random import shuffle
def under_sample(X, y):
    idx_0 = np.where(y == 0)[0].tolist()
    idx_1 = np.where(y == 1)[0].tolist()

    N = np.min([len(idx_0), len(idx_1)])
    idx = idx_0[:N] + idx_1[:N]
    shuffle(idx)

    X = X[idx].reshape(-1)
    y = y[idx].reshape(-1, 1)

    return X, y


from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
