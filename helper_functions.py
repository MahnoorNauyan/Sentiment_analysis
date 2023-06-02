import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from string import punctuation
import re
lemmatize=WordNetLemmatizer()
st = nltk.PorterStemmer()


class Cleaning:



    def cleaning_punctuations(self, tweet):
        '''This function removes punctuaions from tweets'''
        return re.sub(r"[()!?]", "", tweet)

    def cleaning_ats(self, tweet):
        '''This function removes @ signs from tweets'''
        return re.sub("@[A-Za-z0-9_]+", "", tweet)

    def cleaning_httpss(self, tweet):
        '''This function removes https from tweets'''
        return re.sub(r"http.\S+", "", tweet)

    def cleaning_hashs(self, tweet):
        '''This function removes #s from tweets'''
        return re.sub("#[A-Za-z0-9_]+", "", tweet)

    def cleaning_www(self, tweet):
        '''This function removes wwws from tweets'''
        return re.sub("www.\S+", "", tweet)

    def tokenizing(self, tweet):
        return word_tokenize(tweet)

    def stemming_on_text(self, test):
        text = [st.stem(word) for word in test]
        return text

    # df['stemmed_tweets']= df['token_tweets'].apply(lambda x: stemming_on_text(x))

    def cleaner_pipeline(self, tweet):
        '''This function calls all prior functions '''
        tweet = tweet.lower()
        punct = self.cleaning_punctuations(tweet)
        ats = self.cleaning_ats(punct)
        httpss = self.cleaning_httpss(ats)
        hashs = self.cleaning_hashs(httpss)
        wwws = self.cleaning_www(hashs)
        tokens = self.tokenizing(wwws)
        stem = self.stemming_on_text(tokens)
        return stem

# obj1=Cleaning()

# df['Tweet_Cleaned']=df['Tweet'].apply(lambda tweet:obj1.cleaner_pipeline(tweet))