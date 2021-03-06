import os
import pickle
import re
from typing import Dict, List, Union

import spacy
from chatterbot import ChatBot
from gensim.models import FastText, KeyedVectors, Phrases, Word2Vec
from gensim.models.phrases import Phraser
from markovify import Chain
from markovify import Text as MarkovText
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.initializers import lecun_uniform
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Input, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tracery import Grammar
from tracery.modifiers import base_english

import pickle_keras


class Bottimus(MarkovText):
    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            result = pickle.load(f)
        return result

    def __init__(
        self,
        # Data
        generator_text: Union[str, List[str]] = None,
        responder_text: List[str] = None,
        command_text: List[str] = None,
        grammar: Union[Dict[str, str], Grammar] = None,
        # Models
        chain: Union[Dict[str], MarkovText] = None,
        phraser: Union[Dict[str], Phraser] = None,
        word_vectors: Union[Dict[str], KeyedVectors] = None,
        nn: Union[Dict[str], Model] = None,
        # Chatterbot
        commander: ChatBot = None,
        **kwargs: Dict[str, int],
    ):
        # Defaults
        kwargs.update(
            {"word_vector_size": 256, "min_count": 5, "max_vocab_size": 40000000}
        )

        self.nlp = spacy.load("en")

        corpus = list(map(self.word_split, responder_text))

        # Chain
        if (not chain) or isinstance(chain, dict):
            chain = chain or {}
            for Key, Value in {"state_size": 2, "retain_original": True}.items():
                chain.setdefault(Key, Value)

            MarkovText.__init__(
                self,
                None,
                state_size=chain["state_size"],
                parsed_sentences=corpus + list(self.generate_corpus(generator_text)),
                retain_original=chain["retain_original"],
            )
        else:
            MarkovText.__init__(
                self,
                None,
                state_size=chain.state_size,
                chain=chain,
                parsed_sentences=chain.parsed_sentences,
                retain_original=chain.retain_original,
            )

        corpus = [
            [word.split(self.separator)[0] for word in sentence] for sentence in corpus
        ]

        # Phraser
        if (not phraser) or isinstance(phraser, dict):
            phraser = phraser or {}
            for Key, Value in {"gram_size": 3, "scoring": "default"}.items():
                phraser.setdefault(Key, Value)

            for _ in range(phraser["gram_size"]):
                self.phraser = Phraser(
                    Phrases(
                        corpus,
                        min_count=kwargs["min_count"],
                        max_vocab_size=kwargs["max_vocab_size"],
                        scoring=phraser["scoring"],
                    )
                )
                corpus = self.phraser[corpus]
        else:
            self.phraser = phraser
            corpus = self.phraser[corpus]

        # Word Vectors
        if (not word_vectors) or isinstance(word_vectors, dict):
            word_vectors = word_vectors or {}
            for Key, Value in {
                "embedding_model": "fasttext",
                "window": 5,
                "workers": 3,
            }.items():
                word_vectors.setdefault(Key, Value)

            self.word_vectors = {"fasttext": FastText, "word2vec": Word2Vec}[
                word_vectors["embedding_model"].lower()
            ](
                corpus,
                size=kwargs["word_vector_size"],
                window=word_vectors["window"],
                min_count=1,  # kwargs["min_count"],
                workers=word_vectors["workers"],
                max_vocab_size=kwargs["max_vocab_size"],
            ).wv
        else:
            self.word_vectors = word_vectors

        # LSTM RNN
        if (not nn) or isinstance(nn, dict):
            nn = nn or {}
            for Key, Value in {
                "cell_type": "LSTM",
                # "num_layers": 3, Perhaps later
                "max_words": 100,
                "sentence_vector_size": 300,
                "activation": "tanh",
                "dropout_rate": .2,
                "loss": "categorical_crossentropy",
                "learning_rate": .0005,
                "metrics": ["accuracy"],
            }.items():
                nn.setdefault(Key, Value)

            input_statement = Input(
                shape=(nn["max_words"], kwargs["word_vector_size"]),
                name="input_statement",
            )
            input_response = Input(
                shape=(nn["max_words"], kwargs["word_vector_size"]),
                name="input_response",
            )

            self.nn = Model(
                inputs=[input_statement, input_response],
                outputs=[
                    Dense(kwargs["max_vocab_size"], activation="softmax")(
                        Dense(kwargs["max_vocab_size"] / 2, activation="relu")(
                            concatenate(
                                [
                                    Bidirectional(
                                        {"LSTM": LSTM, "GRU": GRU}[nn["cell_type"]](
                                            units=nn["sentence_vector_size"],
                                            input_shape=(
                                                nn["max_words"],
                                                kwargs["word_vector_size"],
                                            ),
                                            activation=nn["activation"],
                                            dropout=nn["dropout_rate"],
                                            kernel_initializer=lecun_uniform(),
                                        )
                                    )(input_statement),
                                    Bidirectional(
                                        {"LSTM": LSTM, "GRU": GRU}[nn["cell_type"]](
                                            units=nn["sentence_vector_size"],
                                            input_shape=(
                                                nn["max_words"],
                                                kwargs["word_vector_size"],
                                            ),
                                            activation=nn["activation"],
                                            dropout=nn["dropout_rate"],
                                            kernel_initializer=lecun_uniform(),
                                        )
                                    )(input_response),
                                ],
                                axis=1,
                            )
                        )
                    )
                ],
            )
            self.nn.compile(
                loss=nn["loss"],
                optimizer=Adam(lr=nn["learning_rate"]),
                metrics=nn["metrics"],
            )
        else:
            self.nn = nn

        # Commander
        self.commander = commander or ChatBot(
            "Commander",
            preprocessors=[
                "chatterbot.preprocessors.clean_whitespace",
                "chatterbot.preprocessors.convert_to_ascii",
            ],
            trainer="chatterbot.trainers.ListTrainer",
            logic_adapters=[
                {"import_path": "chatterbot.logic.BestMatch"},
                {
                    "import_path": "chatterbot.logic.LowConfidenceAdapter",
                    "threshold": 0.65,
                    "default_response": "FAIL",
                },
            ],
        )
        if command_text:
            self.commander.train(command_text)

        # Grammar
        if (not grammar) or isinstance(grammar, dict):
            grammar = grammar or {}
            for Key, Value in {}.items():
                grammar.setdefault(Key, Value)

            self.grammar = Grammar(grammar)
            self.grammar.add_modifiers(base_english)
        else:
            self.grammar = grammar

    def sentence_split(self, text: str) -> list:
        return sent_tokenize(text)

    separator: str = "::"

    def word_split(self, sentence: str, pos: bool = True) -> List[str]:
        if pos:
            tokens = []
            for token in self.nlp(
                sentence, disable=["tagger", "parser", "ner", "textcat"]
            ):
                orth = token.orth_
                if orth.isspace() or token.like_url or orth.startswith("#"):
                    continue
                elif orth.startswith("@"):
                    tokens.append("#username#")
                else:
                    tokens.append(self.separator.join((orth, token.pos_)))
        else:
            tokens = word_tokenize(sentence)
        return tokens

    def word_join(self, words: List[str], pos: bool = True) -> str:
        if pos:
            sentence = " ".join(word.split(self.separator)[0] for word in words)
        else:
            sentence = " ".join(words)
        return sentence

    rule_pattern = re.compile("#(\\w|\\.)+#")

    def make_sentence(
        self, statement: str = None, init_state=None, **kwargs: dict
    ) -> str:
        if statement:
            response = str(self.commander.get_response(statement))
            if response == "FAIL":
                return "NI YET"
        else:
            response = MarkovText.make_sentence(
                self, init_state=init_state, kwargs=kwargs
            )
        return self.grammar.flatten(response)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)
