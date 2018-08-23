import os, spacy

from bot import Bottimus

filename = "archive.sav"
if os.path.isfile(filename):
    Prime = Bottimus.load(filename)
else:
    Prime = Bottimus(
        generator_text=[],
        responder_text=[
            "Who are you?",
            "I am a bot.",
            "Why are you here?",
            "To entertain you.",
        ],
        command_text=[],
        grammar={
            "director": "#d_first# #d_last#",
            "d_first": [
                "J. J.",
                "Ben",
                "Woody",
                "Robert",
                "Michael",
                "P. T.",
                "Wes",
                "Darren",
                "Richard",
                "Warren",
                "Luc",
                "Kathryn",
                "Brad",
                "Neill",
                "Bong",
                "Danny",
                "Kenneth",
                "Mel",
                "Tim",
                "James",
                "Martin",
                "Jane",
                "Frank",
                "John",
                "Jackie",
                "Charlie",
                "Damien",
                "Jack",
                "George",
                "Joel",
                "Ethan",
                "Francis Ford",
                "Sofia",
                "Kevin",
                "David",
                "Alfonso",
                "Guillermo",
                "Brian",
                "Walt",
                "Ava",
                "Clint",
                "Roland",
                "Jon",
                "Jodie",
                "William",
                "Antoine",
            ],
            "d_last": [
                "Abrams",
                "Affleck",
                "Allen",
                "Altman",
                "Anderson",
                "Aronofsky",
                "Attenborough",
                "Bay",
                "Beatty",
                "Besson",
                "Bigelow",
                "Bird",
                "Blomkamp",
                "Joon-ho",
                "Boyle",
                "Branagh",
                "Brooks",
                "Burton",
                "Cameron",
                "Campbell",
                "Campion",
                "Capra",
                "Carpenter",
                "Chan",
                "Chaplin",
                "Chazelle",
                "Clayton",
                "Clooney",
                "Coen",
                "Coppola",
                "Costner",
                "Craven",
                "Crichton",
                "Cronenberg",
                "Cuarón",
                "Curtis",
                "del Toro",
                "De Palma",
                "DeVito",
                "Disney",
                "Donner",
                "DuVernay",
                "Eastwood",
                "Emmerich",
                "Favreau",
                "Fincher",
                "Foley",
                "Ford",
                "Foster",
                "Friedkin",
                "Fuqua",
            ],
        },
    )
