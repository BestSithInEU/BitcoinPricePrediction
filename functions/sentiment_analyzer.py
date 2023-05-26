from emoji import demojize
from langdetect import detect
from transformers import (
    AutoTokenizer,
    BertTokenizerFast,
    TFAutoModelForSequenceClassification,
)

import pandas as pd
import pandas as pd
import re, string
import tensorflow as tf


class SentimentAnalysis:
    def __init__(
        self,
        model_name_en="textattack/bert-base-uncased-imdb",
        model_name_es="dccuchile/bert-base-spanish-wwm-uncased",
    ):
        self.tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
        self.tokenizer_es = AutoTokenizer.from_pretrained(model_name_es)
        self.model_en = TFAutoModelForSequenceClassification.from_pretrained(
            model_name_en, from_pt=True
        )
        self.model_es = TFAutoModelForSequenceClassification.from_pretrained(
            model_name_es, from_pt=True
        )
        self.banned_list = string.punctuation + "Ã" + "±" + "ã" + "¼" + "â" + "»" + "§"

    def strip_emoji(self, text):
        return demojize(text, delimiters=("", ""))

    def strip_all_entities(self, text):
        text = text.replace("\r", "").replace("\n", " ").replace("\n", " ").lower()
        text = re.sub(r"(?:\@|https?\://)\S+", "", text)
        text = re.sub(r"[^\x00-\x7f]", r"", text)
        table = str.maketrans("", "", self.banned_list)
        text = text.translate(table)
        return text

    def clean_hashtags(self, tweet):
        new_tweet = " ".join(
            word.strip()
            for word in re.split(
                "#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)", tweet
            )
        )
        new_tweet2 = " ".join(word.strip() for word in re.split("#|_", new_tweet))
        return new_tweet2

    def filter_chars(self, a):
        sent = []
        for word in a.split(" "):
            if ("$" in word) | ("&" in word):
                sent.append("")
            else:
                sent.append(word)
        return " ".join(sent)

    def remove_mult_spaces(self, text):
        return re.sub("\s\s+", " ", text)

    def get_sentiment(self, text):
        lang = detect(text)

        if lang == "en":
            inputs = self.tokenizer_en.encode_plus(
                text, return_tensors="tf", truncation=True, padding=True
            )
            logits = self.model_en(inputs)[0]
        elif lang == "es":
            inputs = self.tokenizer_es.encode_plus(
                text, return_tensors="tf", truncation=True, padding=True
            )
            logits = self.model_es(inputs)[0]
        else:
            print("Language is neither English nor Spanish.")
            return None

        probs = tf.nn.softmax(logits, axis=1).numpy()[0]

        return probs

    def process_text_data(self, data_filepath):
        df = pd.read_csv(data_filepath, delimiter=";")
        df = df.drop(columns="Score")

        texts_new = [
            self.remove_mult_spaces(
                self.filter_chars(
                    self.clean_hashtags(self.strip_all_entities(self.strip_emoji(t)))
                )
            )
            for t in df.News
        ]
        df["text_clean"] = texts_new

        text_len = [len(text.split()) for text in df.text_clean]
        df["text_len"] = text_len
        df = df[df["text_len"] > 4]

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        token_lens = [
            len(tokenizer.encode(txt, max_length=512, truncation=True))
            for txt in df["text_clean"].values
        ]
        df = (
            df.assign(token_lens=token_lens)
            .sort_values(by="token_lens", ascending=False)
            .sample(frac=1)
            .reset_index(drop=True)
        )

        sentiments = df["text_clean"].apply(self.get_sentiment).to_list()
        df["neg_prob"], df["neu_prob"], df["pos_prob"] = zip(*sentiments)

        return df
