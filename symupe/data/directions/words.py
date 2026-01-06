""" Utility functions for word-based direction markings. """
from __future__ import annotations

import string


def word_regularization(word):
    if word:
        for symbol in string.punctuation:
            word = word.replace(symbol, " ")
        word = word.replace("  ", " ")
        return word.strip().lower()
    else:
        return None


def extract_main_keyword(key):
    if isinstance(key, tuple):
        return key[0]
    return key


def extract_direction_with_class(dir_word: str, direction_maps: dict, classes: list | None = None):
    for cls, keywords in direction_maps.items():
        if classes and cls not in classes or cls == "ALL":
            continue
        for key in keywords:
            if isinstance(key, tuple) and dir_word in key:
                return f"{cls}/{key[0]}"
            elif dir_word == key:
                return f"{cls}/{key}"
    return


def extract_direction_by_keys(dir_word, keywords):
    for key in keywords:
        if isinstance(key, tuple) and dir_word in key:
            return key[0]
        elif dir_word == key:
            return key
    return


def extract_all_directions_by_keys(dir_word, keywords):
    directions = []
    for key in keywords:
        if isinstance(key, tuple) and dir_word in key:
            directions.append(key[0])
        elif dir_word == key:
            directions.append(key)
    return directions


def check_direction_by_keywords(dir_word, keywords):
    dir_word = word_regularization(dir_word)
    if dir_word in keywords:
        return True
    else:
        word_split = dir_word.split(" ")
        for w in word_split:
            if w in keywords:
                return True

    for key in keywords:  # words like "sempre più mosso"
        if len(key) > 2 and key in dir_word:
            return True

    return False
