from string import punctuation
import re

RADICAL_INDEX = 0
COMPONENT_INDEX = 1
OOV_CHAR = "□"

MIN = -1e32
HALF_MIN = -1e4
MAX = 1e32
EPS = 1e-7

chinese_punct = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟–—‘'‛“”„‟…‧."
english_punct = punctuation
PUNCT = set(chinese_punct + english_punct)

consonant_inits = {"q", "w", "r", "t", "y", "o", "p", "s", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "b", "n", "m"}
reAlNUM = re.compile(r"^[a-zA-Z0-9]+$")