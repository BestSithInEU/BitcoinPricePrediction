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
    """
    A class for performing sentiment analysis on text data using VADER sentiment analysis and RoBERTa tokenization.

    The SentimentAnalysis class provides methods for data preprocessing, sentiment score computation,
    tokenization using RoBERTa, and aggregation of sentiment scores. It also includes functionality to save
    the processed data to a CSV file.

    Parameters
    ----------
    df : DataFrame
        The input dataset containing text data.
    seed : int
        Random seed used for reproducibility.
    stopwords : set
        Set of stopwords for text cleaning.
    sid : SentimentIntensityAnalyzer
        An instance of the SentimentIntensityAnalyzer from the NLTK library.
    """

    def __init__(self, path, delimiter, seed=42):
        self.df = pd.read_csv(path, delimiter=delimiter)
        self.seed = seed
        self.stopwords = set(stopwords.words("english"))
        self.sid = SentimentIntensityAnalyzer()
        self.df["Date"] = pd.to_datetime(self.df["Date"])

    def preprocess(self):
        """
        Preprocesses the data by cleaning text and computing VADER sentiment scores.

        Returns
        -------
        DataFrame
            The processed dataframe.
        """

        self.df = self.df.rename(columns={"News": "content"})
        self.df["cleantext"] = [
            self.tweet_to_words(item) for item in tqdm(self.df["content"])
        ]
        self.df = self.compute_vader_scores(self.df, "cleantext")
        return self.df

    def tweet_to_words(self, tweet):
        """
        Cleans a tweet by removing non-alphanumeric characters and stopwords,
        and applies stemming.

        Parameters
        ----------
        tweet : str
            A tweet.

        Returns
        -------
        list
            A list of cleaned words from the tweet.
        """

        text = tweet.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        words = text.split()
        words = [w for w in words if w not in self.stopwords]
        words = [self.stemmer.stem(w) for w in words]
        return words

    def unlist(self, list):
        """
        Joins a list of words into a string.

        Parameters
        ----------
        list : list
            A list of words.

        Returns
        -------
        str
            A string with words separated by spaces.
        """

        words = " ".join(list)
        return words

    def compute_vader_scores(self, df, label):
        """
        Computes VADER sentiment scores (negative, neutral, positive, compound) for each tweet.

        Parameters
        ----------
        df : DataFrame
            The dataframe.
        label : str
            The column in the dataframe containing the text to analyze.

        Returns
        -------
        DataFrame
            The dataframe with the added VADER sentiment scores.
        """

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
        """
        Tokenizes inputs using RoBERTa for training, validation, and testing.

        Parameters
        ----------
        max_len : int
            Maximum length for tokenization.
        """

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
        """
        Aggregates VADER sentiment scores by date.

        Returns
        -------
        DataFrame
            The aggregated dataframe.
        """

        aggregation_function = {
            "vader_neg": lambda x: float(x[x > 0].count()),
            "vader_neu": lambda x: float(x[x > 0].count()),
            "vader_pos": lambda x: float(x[x > 0].count()),
        }

        aggregated_df = self.df.groupby("Date").agg(aggregation_function).reset_index()
        aggregated_df.columns = ["Date", "vader_neg", "vader_neu", "vader_pos"]
        self.df = aggregated_df

        return self.df

    def save_df(self, filename):
        """
        Aggregates the dataframe by date and saves it to a CSV file.

        Parameters
        ----------
        filename : str
            The name of the output CSV file.
        """

        self.aggregate_by_date()
        self.df["Date"] = self.df["Date"].dt.strftime("%b %d, %Y")
        self.df.to_csv(filename, index=False)
