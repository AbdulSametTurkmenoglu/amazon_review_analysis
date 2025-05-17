from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

# Suppress warnings
filterwarnings('ignore')

# Display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Load dataset
df = pd.read_excel("datasets/amazon.xlsx")

# Display first rows and info
df.head()
df.info()

# Lowercase conversion
df['Review'] = df['Review'].str.lower()

# Remove punctuation
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# Remove digits
df['Review'] = df['Review'].str.replace('\d', '')

# Remove stopwords
stop_words = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))

# Remove rare words (last 1000 in frequency)
sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Lemmatization
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Term frequency
tf = df['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["word", "tf"]

# Plot most frequent words
tf[tf["tf"] > 500].plot.bar(x="word", y="tf")
plt.show()

# Generate word cloud
text = " ".join(i for i in df.Review)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Preview processed text
df["Review"].head()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example sentiment scores
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))                      # full sentiment score dict
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])         # only compound score
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")  # label

# Add sentiment score and label to dataframe
df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# Encode sentiment labels
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

# Define target and features
y = df["sentiment_label"]
X = df["Review"]

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF vectorization
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

# Logistic Regression
log_model = LogisticRegression().fit(X_tf_idf_word, y)

# Cross-validation accuracy
print(cross_val_score(log_model, X_tf_idf_word, y, scoring="accuracy", cv=5).mean())

# Predict sentiment of a random review
random_review = pd.Series(df["Review"].sample(1).values)
random_review = CountVectorizer().fit(X).transform(random_review)
print(log_model.predict(random_review))

# Random Forest Classifier
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
print(cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean())
