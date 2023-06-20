from util.tokenize.chinese_char import ChineseCharTokenizer
from util.tokenize.english_char import EnglishCharTokenizer

REGISTER_TOKENIZER = {
    "chinese_char": ChineseCharTokenizer,
    "english_char": EnglishCharTokenizer,
}

