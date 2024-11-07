from collections import defaultdict
import pickle

from pypinyin import pinyin, Style
from pypinyin_dict.pinyin_data import ktghz2013
from pypinyin_dict.phrase_pinyin_data import large_pinyin
import json
from tqdm import tqdm

from lmcsc.common import PUNCT, RADICAL_INDEX, COMPONENT_INDEX, OOV_CHAR


# load better pinyin data
ktghz2013.load()
# load better phrase pinyin data
large_pinyin.load()


class TransformationType:
    def __init__(self, vocab, is_bytes_level, shape_similar_threshold=0.45):
        self.similar_shape_dict = self.load_dict("data/similar_shape_dict.json")

        # https://github.com/gingasan/lemon/blob/main/confus/stroke.json
        self.shape_confusion_dict = self.load_dict("data/shape_confusion_dict.json")

        # CSCD-IME
        self.similar_consonant_dict = self.load_dict("data/similar_consonant_dict.json")

        # 
        self.similar_vowel_dict = self.load_dict("data/similar_vowel_dict.json")
        self.similar_spell_dict = self.load_similar_spell_dict(
            "data/pinyin_distance_matrix.pkl"
        )

        # Prone to confusion
        self.prone_to_confusion_dict = self.load_dict("data/prone_to_confusion_dict.json")

        self.vocab = vocab
        self.is_bytes_level = is_bytes_level
        self.shape_similar_threshold = shape_similar_threshold

        self.build_inverse_index()
        self.cache = None
        self.same_pinyin_cache = {}
        self.similar_pinyin_cache = {}
        self.similar_shape_cache = {}

    def load_dict(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_similar_spell_dict(self, file_name):
        self.spell_distance_matrix = pickle.load(open(file_name, "rb"))
        similar_spell_dict = defaultdict(set)
        for pair, distance in self.spell_distance_matrix.items():
            if distance <= 1.0:
                similar_spell_dict[pair[0]].add(pair[1])
                similar_spell_dict[pair[1]].add(pair[0])
        return similar_spell_dict

    def build_inverse_index(self):
        self.identical_char_index = defaultdict(set)
        self.same_pinyin_index = defaultdict(set)
        self.similar_pinyin_index = defaultdict(set)
        self.other_similar_pinyin_index = defaultdict(set)
        self.similar_shape_index = defaultdict(set)
        self.other_similar_shape_index = defaultdict(set)
        self.token_length = {}
        self.char_set = set()

        for k, idx in tqdm(self.vocab.items()):
            ori_token = k
            if self.is_bytes_level:
                try:
                    ori_token = k.decode("utf-8")
                except UnicodeDecodeError:
                    for i, byte in enumerate(ori_token):
                        self.identical_char_index[(i, byte)].add(idx)
                    # We assume that a Chinese char worth 3 bytes
                    self.token_length[idx] = len(ori_token) / 3
                    continue
            # Handle the byte fallback token: <0xFF>
            if (
                len(ori_token) == 6
                and ori_token.startswith("<0x")
                and ori_token.endswith(">")
            ):
                self.token_length[idx] = 1 / 3
            else:
                self.token_length[idx] = len(ori_token)
            
            token = ori_token
            
            # Build identical_char_index
            for i, char in enumerate(token):
                self.char_set.add(char)
                self.identical_char_index[(i, char)].add(idx)
                # Handle freq error chars: 做 <-> 作
                for equal_char in self.prone_to_confusion_dict.get(char, []):
                    self.identical_char_index[(i, equal_char)].add(idx)

            # Build same pinyin index
            token_pinyins = pinyin(token, style=Style.NORMAL, heteronym=True)
            if len(token_pinyins) == 1 and token_pinyins[0][0] == token:
                continue
            for i, ps in enumerate(token_pinyins):
                for p in ps:
                    self.same_pinyin_index[(i, p)].add(idx)
                    for similar_pinyin in self.similar_spell_dict.get(p, []):
                        self.other_similar_pinyin_index[(i, similar_pinyin)].add(idx)

            # Build similar pinyin index
            token_consonants = pinyin(token, style=Style.INITIALS, heteronym=True)
            token_vowels = pinyin(token, style=Style.FINALS, heteronym=True)
            for i, (consonant, vowel) in enumerate(zip(token_consonants, token_vowels)):
                similar_consonants = set(consonant)
                similar_vowels = set(vowel)
                for c in consonant:
                    similar_consonants.update(self.similar_consonant_dict.get(c, [c]))
                for v in vowel:
                    similar_vowels.update(self.similar_vowel_dict.get(v, [v]))
                for c in similar_consonants:
                    for v in similar_vowels:
                        self.similar_pinyin_index[(i, c + v)].add(idx)
            
            # Structure index
            for i, char in enumerate(token):
                if char in self.similar_shape_dict:
                    for similar_char in self.similar_shape_dict[char]:
                        self.similar_shape_index[(i, similar_char)].add(idx)
                # In case some shape similar char is missing
                if char in self.shape_confusion_dict:
                    for similar_char in self.shape_confusion_dict[char]:
                        self.other_similar_shape_index[(i, similar_char)].add(idx)

    def get_transformation_type(self, observed_sequence: str):
        token_transformation = defaultdict(dict)
        oov_transformation = None
        if self.is_bytes_level and len(observed_sequence) > 0 and observed_sequence[0] not in self.char_set:
            # Handle rare characters, that alway appear in bytes level
            oov_transformation = {}
            if isinstance(observed_sequence[0], int):
                assert isinstance(observed_sequence, bytes)
                token_bytes = observed_sequence
            else:
                token_bytes = observed_sequence[0].encode("utf-8")
            idx = None
            # Finding the longest bytes
            for i in range(len(token_bytes)):
                tmp_byte = token_bytes[: i + 1]
                if tmp_byte in self.vocab:
                    idx = self.vocab[tmp_byte]
            assert idx is not None, f"{observed_sequence} {token_bytes}"
            oov_transformation[idx] = ["IDT"]
        vaildated = True
        # Deal with sequence cause pinyin error
        try:
            token_pinyins = pinyin(
                observed_sequence,
                style=Style.NORMAL,
                heteronym=True,
                errors=lambda x: [char for char in x],
            )
            token_consonants = pinyin(
                observed_sequence,
                style=Style.INITIALS,
                heteronym=True,
                errors=lambda x: [char for char in x],
            )
            token_vowels = pinyin(
                observed_sequence,
                style=Style.FINALS,
                heteronym=True,
                errors=lambda x: [char for char in x],
            )
        except:
            vaildated = False
        for i in range(len(observed_sequence)):
            if (i, observed_sequence[i]) in self.identical_char_index:
                for idx in self.identical_char_index[(i, observed_sequence[i])]:
                    if i not in token_transformation[idx]:
                        token_transformation[idx][i] = "IDT"
            if observed_sequence[i] in PUNCT and observed_sequence[i] != "_":
                # Early break for time saving
                break
            if not vaildated:
                continue
            for p in token_pinyins[i]:
                if (i, p) in self.same_pinyin_index:
                    for idx in self.same_pinyin_index[(i, p)]:
                        if i not in token_transformation[idx]:
                            token_transformation[idx][i] = "SAP"

            if (i, tuple(token_consonants[i]), tuple(token_vowels[i])) in self.similar_pinyin_cache:
                pinyin_similar = self.similar_pinyin_cache[(i, tuple(token_consonants[i]), tuple(token_vowels[i]))]
            else:
                pinyin_similar = set()
                similar_consonants = set(token_consonants[i])
                similar_vowels = set(token_vowels[i])
                for c in token_consonants[i]:
                    similar_consonants.update(self.similar_consonant_dict.get(c, [c]))
                for v in token_vowels[i]:
                    similar_vowels.update(self.similar_vowel_dict.get(v, [v]))

                for c in similar_consonants:
                    for v in similar_vowels:
                        if c in token_consonants[i] or v in token_vowels[i]:
                            if (i, c + v) in self.similar_pinyin_index:
                                for idx in self.similar_pinyin_index[(i, c + v)]:
                                    pinyin_similar.add(idx)
                self.similar_pinyin_cache[(i, tuple(token_consonants[i]), tuple(token_vowels[i]))] = pinyin_similar
            for idx in pinyin_similar:
                if i not in token_transformation[idx]:
                    token_transformation[idx][i] = "SIP"

            if (i, observed_sequence[i]) in self.similar_shape_index:
                for idx in self.similar_shape_index[(i, observed_sequence[i])]:
                    if i not in token_transformation[idx]:
                        token_transformation[idx][i] = "SIS"

            # deal with other similar characters
            if (i, observed_sequence[i]) in self.other_similar_shape_index:
                for idx in self.other_similar_shape_index[(i, observed_sequence[i])]:
                    if i not in token_transformation[idx]:
                        token_transformation[idx][i] = "OTH"
            for p in token_pinyins[i]:
                if (i, p) in self.other_similar_pinyin_index:
                    for idx in self.other_similar_pinyin_index[(i, p)]:
                        if i not in token_transformation[idx]:
                            token_transformation[idx][i] = "OTH"

        new_transformation = {}
        # filter out partial matched tokens
        for idx, transformation in token_transformation.items():
            if len(transformation) == self.token_length[idx]:
                new_transformation[idx] = list(transformation.values())
            elif len(transformation) > 1 and (
                (self.token_length[idx] - len(transformation)) == 1
            ):
                new_transformation[idx] = list(transformation.values()) + ["UNR"]

        token_transformation = new_transformation
        if oov_transformation is not None:
            token_transformation.update(oov_transformation)

        # handle oov char again
        if len(token_transformation) == 0:
            for idx in self.identical_char_index[(0, OOV_CHAR)]:
                if self.token_length[idx] == 1:
                    token_transformation[idx] = ["IDT"]
        
        assert len(token_transformation) > 0, f"{observed_sequence} {token_transformation}"
        return token_transformation
