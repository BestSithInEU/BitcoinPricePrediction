import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping
from transformers import (
    RobertaTokenizerFast,
    TFRobertaModel,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentAnalysis:
    def __init__(self, path, delimiter, seed=42):
        self.df = pd.read_csv(path, delimiter=delimiter)
        self.seed = seed
        self.stopwords = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.sid = SentimentIntensityAnalyzer()
        self.tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.ohe = OneHotEncoder()
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

    def split_data(self):
        X = self.df["cleantext2"].values
        y = self.df["class"].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.seed
        )
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            self.X_test, self.y_test, test_size=0.5, random_state=self.seed
        )

    def encode_labels(self):
        self.y_train = self.ohe.fit_transform(
            np.array(self.y_train).reshape(-1, 1)
        ).toarray()
        self.y_valid = self.ohe.transform(
            np.array(self.y_valid).reshape(-1, 1)
        ).toarray()
        self.y_test = self.ohe.transform(np.array(self.y_test).reshape(-1, 1)).toarray()

    def tokenize_roberta(self, data, max_len):
        input_ids = []
        attention_masks = []
        for i in range(len(data)):
            encoded = self.tokenizer_roberta.encode_plus(
                data[i],
                add_special_tokens=True,
                max_length=max_len,
                padding="max_length",
                return_attention_mask=True,
            )
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
        return np.array(input_ids), np.array(attention_masks)

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

    def create_model(self, max_len):
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = tf.keras.losses.CategoricalCrossentropy()
        accuracy = tf.keras.metrics.CategoricalAccuracy()
        roberta_model = TFRobertaModel.from_pretrained("roberta-base")
        input_ids = tf.keras.Input(shape=(max_len,), dtype="int32")
        attention_masks = tf.keras.Input(shape=(max_len,), dtype="int32")
        output = roberta_model([input_ids, attention_masks])
        output = output[1]
        output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(output)
        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks], outputs=output
        )
        model.compile(opt, loss=loss, metrics=accuracy)
        return model

    def train_model(self, model, epochs=3, batch_size=32):
        early_stop = EarlyStopping(monitor="val_loss", patience=3)
        history = model.fit(
            [self.train_input_ids, self.train_attention_masks],
            self.y_train,
            validation_data=(
                [self.val_input_ids, self.val_attention_masks],
                self.y_valid,
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
        )
        return history

    def predict(self, model):
        result = model.predict([self.test_input_ids, self.test_attention_masks])
        y_pred = np.zeros_like(result)
        y_pred[np.arange(len(y_pred)), result.argmax(1)] = 1
        return y_pred

    def conf_matrix(self, y, y_pred, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        labels = ["Negative", "Neutral", "Positive"]
        ax = sns.heatmap(
            confusion_matrix(y, y_pred),
            annot=True,
            cmap="Blues",
            fmt="g",
            cbar=False,
            annot_kws={"size": 25},
        )
        plt.title(title, fontsize=20)
        ax.xaxis.set_ticklabels(labels, fontsize=17)
        ax.yaxis.set_ticklabels(labels, fontsize=17)
        ax.set_ylabel("Test", fontsize=20)
        ax.set_xlabel("Predicted", fontsize=20)
        plt.show()

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
