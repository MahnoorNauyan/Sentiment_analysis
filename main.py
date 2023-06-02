from controller import Pipeline
from helper_functions import Cleaning
import pandas as pd
import pickle

df = pd.read_csv('tsa.csv',encoding='latin-1')

pickle.load(open(r'/home/mahnoor/Personal/Mahnoor/save_data/RB projects/SentimentAnalysis/model/final_model.sav', 'rb'))
with open (r'/home/mahnoor/Personal/Mahnoor/save_data/RB projects/SentimentAnalysis/model/final_model.sav','rb') as f:
    model=pickle.load(f)
with open (r'/home/mahnoor/Personal/Mahnoor/save_data/RB projects/SentimentAnalysis/model/vectorizer.pickle','rb') as a:
    model_vec=pickle.load(a)

obj=Pipeline(df)
print("- "*10, "sentiment coversion", "- "*10)
obj.senti_conversion()
print("- "*10, "CLEANING", "- "*10)
obj.clean_tweets()#cleaning
print("- "*10, "PreProcessing", "- "*10)
obj.preprocess()#preprocessing
print("- "*10, "Training", "- "*10)
obj.training()#training

print("- "*10, "Evaluation", "- "*10)
obj.model_Evaluate()#Evaluation of model
print("- "*10, "saving model", "- "*10)
obj.saving_model()#saving model

text = 'i love my  country'
obj.predict_tweet(text)
