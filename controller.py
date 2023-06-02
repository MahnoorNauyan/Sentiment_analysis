from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns #for plotting(visualization library)
import matplotlib.pyplot as plt
from helper_functions import Cleaning
import pickle

class Pipeline:


    def __init__(self, df):
        self.df = df
        self.clean = Cleaning()
        self.SVCmodel = LinearSVC()
        self.vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)

    def senti_conversion(self):
        self.df.loc[self.df["Sentiment"] == 0, "Sentiment"] = -1
        self.df.head(3)
        # positive
        self.df.loc[self.df["Sentiment"] == 4, "Sentiment"] = 1
        self.df.tail(3)
        return self.df["Sentiment"]

    def clean_tweets(self):
        self.df['Cleaned'] = self.df['Tweet'].apply(lambda x: self.clean.cleaner_pipeline(x))
        return self.df

    def preprocess(self):
        self.x = self.df.Cleaned
        self.y = self.df.Sentiment
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.05,
                                                                                random_state=26105111)
        self.x_train = self.x_train.apply(lambda x: ' '.join(x))
        self.x_test = self.x_test.apply(lambda x: ' '.join(x))
        # print(self.x_train.head(2))

    def training(self):
        self.vectoriser.fit(self.x_train)
        self.x_train = self.vectoriser.transform(self.x_train)
        self.x_test = self.vectoriser.transform(self.x_test)
        self.SVCmodel.fit(self.x_train, self.y_train)  # model training
        print('No. of feature_words: ', len(self.vectoriser.get_feature_names()))

    def model_Evaluate(self):
        self.y_pred = self.SVCmodel.predict(self.x_test)  # predict values for test from dataset
        # Classification report
        print(classification_report(self.y_test,
                                    self.y_pred))  # gives performance evaluation metrics(precision,recall,f1 score)
        # Confusion Matrix
        self.cf_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.categories = ['Negative', 'Positive']
        self.group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        self.group_percentages = ['{0:.2%}'.format(value) for value in
                                  self.cf_matrix.flatten() / np.sum(self.cf_matrix)]
        self.labels = [f'{v1} {v2}' for v1, v2 in zip(self.group_names, self.group_percentages)]
        self.labels = np.asarray(self.labels).reshape(2, 2)  # converting input to array 2 rows 2 columns
        sns.heatmap(self.cf_matrix, annot=self.labels, cmap='Blues', fmt='',
                         # heatmap gives 2D graphical representation where matrix contains colors
                         xticklabels=self.categories, yticklabels=self.categories)
        plt.xlabel("Predicted values", fontdict={'size': 14},
                        labelpad=10)  # x label functon for taking label of x axis
        plt.ylabel("Actual values", fontdict={'size': 14},
                        labelpad=10)  # y label function for taking label of y axis
        plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)  # T##

    def saving_model(self):
        self.model = self.SVCmodel
        self.filename = 'final_model.sav'
        pickle.dump(self.model, open(self.filename, 'wb'))
        self.loaded_model = pickle.load(open(self.filename, 'rb'))
        self.result = self.loaded_model.score(self.x_test, self.y_test)
        print(self.result)

    def predict_tweet(self, text):
        self.cleaned_tweet = self.clean.cleaner_pipeline(text)  # Cleaning()
        self.cleaned_tweet = ' '.join(self.cleaned_tweet)
        self.transformed_text = self.vectoriser.transform([self.cleaned_tweet])
        self.sentiment = self.SVCmodel.predict(self.transformed_text)
        if self.sentiment == -1:
            print("Tweet is negative.")
        else:
            print("Tweet is positive.")


