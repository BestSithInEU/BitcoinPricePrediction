import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentAnalysis:
    def __init__(self, path, delimiter, seed=42):
        self.df = pd.read_csv(path, delimiter=delimiter)
        self.seed = seed
        self.stopwords = set(stopwords.words("english"))
        self.sid = SentimentIntensityAnalyzer()
        self.df["Date"] = pd.to_datetime(self.df["Date"])

    def preprocess(self):
        self.df = self.df.rename(columns={"News": "content"})
        self.df["cleantext"] = [
            self.tweet_to_words(item) for item in tqdm(self.df["content"])
        ]
        self.df = self.compute_vader_scores(self.df, "cleantext")
        return self.df

    def tweet_to_words(self, tweet):
        text = tweet.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        words = text.split()
        words = [w for w in words if w not in self.stopwords]
        words = [self.stemmer.stem(w) for w in words]
        return words

    def unlist(self, list):
        words = " ".join(list)
        return words

    def compute_vader_scores(self, df, label):
        sid = SentimentIntensityAnalyzer()
        self.df["vader_neg"] = self.df[label].apply(
            lambda x: sid.polarity_scores(self.unlist(x))["neg"]
        )
        self.df["vader_neu"] = self.df[label].apply(
            lambda x: sid.polarity_scores(self.unlist(x))["neu"]
        )
        self.df["vader_pos"] = self.df[label].apply(
            lambda x: sid.polarity_scores(self.unlist(x))["pos"]
        )
        self.df["vader_comp"] = self.df[label].apply(
            lambda x: sid.polarity_scores(self.unlist(x))["compound"]
        )
        self.df["cleantext2"] = self.df[label].apply(lambda x: self.unlist(x))

        class0 = []
        for i in range(len(self.df)):
            if self.df.loc[i, "vader_neg"] > 0:
                class0 += [0]
            elif self.df.loc[i, "vader_pos"] > 0:
                class0 += [2]
            else:
                class0 += [1]
        self.df["class"] = class0

        return self.df

    def process_inputs(self, max_len):
        self.train_input_ids, self.train_attention_masks = self.tokenize_roberta(
            self.X_train, max_len
        )
        self.val_input_ids, self.val_attention_masks = self.tokenize_roberta(
            self.X_valid, max_len
        )
        self.test_input_ids, self.test_attention_masks = self.tokenize_roberta(
            self.X_test, max_len
        )

    def aggregate_by_date(self):
        # Define aggregation function
        aggregation_function = {
            "vader_neg": lambda x: float(x[x > 0].count()),
            "vader_neu": lambda x: float(x[x > 0].count()),
            "vader_pos": lambda x: float(x[x > 0].count()),
        }

        # Group by date and apply aggregation function
        aggregated_df = self.df.groupby("Date").agg(aggregation_function).reset_index()

        # Rename columns
        aggregated_df.columns = ["Date", "vader_neg", "vader_neu", "vader_pos"]

        self.df = aggregated_df

        return self.df

    def save_df(self, filename):
        self.aggregate_by_date()
        self.df["Date"] = self.df["Date"].dt.strftime("%b %d, %Y")
        self.df.to_csv(filename, index=False)
