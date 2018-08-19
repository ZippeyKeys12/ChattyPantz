import os
import pickle
import tempfile
from zipfile import ZipFile

import spacy
from gensim.models import KeyedVectors, Word2Vec
from keras.layers import LSTM, Bidirectional, Dense, Embedding
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from markovify import Chain
from markovify import Text as MarkovText
from nltk.tokenize import sent_tokenize, word_tokenize

nlp = spacy.load("en")


class Bottimus(MarkovText):
    @staticmethod
    def load(filename: str):
        with ZipFile(filename) as archive, archive.open(
            "markov.p", "rb"
        ) as mc, archive.open("lstm.h5", "rb") as lstm:
            result = Bottimus(
                chain=pickle.load(mc.read()),
                word_vectors=KeyedVectors.load("word2vec.kv", mmap="r"),
                lstm=load_model(lstm),
            )
        return result

    def __init__(
        self,
        input_text: list(str) = None,
        state_size: int = 2,
        chain: Chain = None,
        retain_original: bool = True,
        word_vectors: KeyedVectors = None,
        vocab_size: int = None,
        vocab_vector_dim: int = None,
        lstm: Sequential = None,
        lstm_nodes: int = None,
        dropout_rate: float = None,
        learning_rate: float = None,
        loss: str = None,
        metrics: list(str) = None,
        epochs: int = None,
        max_words: int = None,
    ):
        self.vocab_size = vocab_size or 5000
        self.vocab_vector_dim = vocab_vector_dim or 300

        self.lstm_nodes = lstm_nodes or 256
        self.dropout_rate = dropout_rate or .2
        self.learning_rate = learning_rate or .1
        self.loss = loss or "cosine_proximity"
        self.metrics = metrics or ["accuracy"]
        self.epochs = epochs or 20
        self.max_words = max_words or 100

        MarkovText.__init__(
            self,
            self.sentence_join(input_text),
            state_size=state_size,
            chain=chain,
            retain_original=retain_original,
        )

        if input_text:
            corpus = []
            for index, sentence in enumerate(input_text):
                if index % 2 == 0:
                    corpus.append(word_tokenize(sentence))
                else:
                    corpus.append([word.lemma_ for word in nlp(sentence)])
            self.word_vectors = self.compile_word2vec(corpus)
            self.lstm = self.compile_lstm(corpus)
        else:
            self.word_vectors = word_vectors
            self.lstm = lstm

    def compile_word2vec(self, corpus: list(list(str))) -> KeyedVectors:
        word_vectors = Word2Vec(
            corpus,
            size=self.vocab_vector_dim,
            window=5,
            min_count=1,
            workers=16,
            max_final_vocab=self.vocab_size,
        ).wv
        self.vocab_size = len(word_vectors.vocab)
        return word_vectors

    def compile_lstm(self, corpus: list(list(str))) -> Sequential:
        model = Sequential()
        for _ in range(3):
            model.add(
                Bidirectional(
                    LSTM(
                        self.lstm_nodes,
                        input_length=self.max_words,
                        input_dim=self.vocab_vector_dim,
                        output_dim=self.vocab_vector_dim,
                        activation="sigmoid",
                        dropout=self.dropout_rate,
                        # init="glorot_normal",
                        # inner_init="glorot_normal",
                        return_sequences=True,
                    )
                )
            )
        # model.add(Dense(self.vocab_size))
        model.compile(
            loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=self.metrics
        )
        return model

    separator: str = "::"

    def sentence_split(self, text: str) -> list(str):
        return sent_tokenize(text)

    def word_split(self, sentence: str) -> list(str):
        tokens = []
        for token in nlp(sentence, disable=["tagger", "parser", "ner", "textcat"]):
            orth = token.orth_
            if (
                orth.isspace()
                or token.like_url
                or orth.startswith("@")
                or orth.startswith("#")
            ):
                continue
            else:
                tokens.append(self.separator.join((orth, token.pos_)))
        return tokens

    def word_join(self, words: list(str)) -> str:
        sentence = " ".join(word.split(self.separator)[0] for word in words)
        return sentence

    def make_sentence(
        self, statement: str = None, init_state=None, **kwargs: dict
    ) -> str:
        if not statement:
            return MarkovText.make_sentence(self, init_state=init_state, kwargs=kwargs)

    def save(self, filename: str):
        with ZipFile(filename, "w") as archive:
            with tempfile.NamedTemporaryFile(suffix=".p", delete=True) as fd:
                pickle.dump(self.chain, fd)
                archive.write(fd.name, "markov.p")
            with tempfile.NamedTemporaryFile(suffix=".kv", delete=True) as fd:
                self.word_vectors.save(fd)
                archive.write(fd.name, "word2vec.kv")
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as fd:
                self.lstm.save(fd.name)
                archive.write(fd.name, "lstm.h5")


input_text = """
Who are you?
I am a bot.
Why are you here?
To entertain you.
"""
filename = "archive.sav"
if os.path.isfile(filename):
    Prime = Bottimus.load(filename)
else:
    Prime = Bottimus()
