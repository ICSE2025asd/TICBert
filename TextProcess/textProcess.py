import re
import string
from string import punctuation, digits
from TextProcess import processOp

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    stop_word_set: set
    wnl: WordNetLemmatizer
    remove_digits: str.maketrans
    html_pattern: re.compile
    punctuation_table: str.maketrans

    def __init__(self):
        self.stop_word_set = set(stopwords.words('english') + list(string.ascii_lowercase))
        self.wnl = WordNetLemmatizer()
        self.html_pattern = re.compile(r'<[^>]+|[^<]+>', re.S)

    def text_process(self, text):
        # remove url and stacks
        text = processOp.removeUrlAndStack(text)

        # remove http labels
        text = processOp.removeSpecialPatterns(self.html_pattern, text)

        return text

