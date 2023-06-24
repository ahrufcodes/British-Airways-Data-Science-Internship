import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load the CSV data into a DataFrame
data = pd.read_csv("BA_reviews.csv")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each review
sentiments = []
for review in data['reviews']:
    sentiment = sia.polarity_scores(review)
    sentiments.append(sentiment)

# Add the sentiment scores to the DataFrame
data['positive_sentiment'] = [score['pos'] for score in sentiments]
data['negative_sentiment'] = [score['neg'] for score in sentiments]
data['neutral_sentiment'] = [score['neu'] for score in sentiments]
data['compound_sentiment'] = [score['compound'] for score in sentiments]

# Print the updated DataFrame with sentiment scores
print(data.head())
