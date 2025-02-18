from collections import defaultdict, Counter
from math import ceil
import os
import pickle

from pypinyin import pinyin, Style
from pypinyin_dict.pinyin_data import ktghz2013
from pypinyin_dict.phrase_pinyin_data import large_pinyin
import json
from tqdm import tqdm

from lmcsc.common import PUNCT, OOV_CHAR, consonant_inits, reAlNUM


# load better pinyin data
ktghz2013.load()
# load better phrase pinyin data
large_pinyin.load()


import yaml

class TransformationType:
    r"""
    A class for handling various types of transformations on input sequences, particularly for Chinese text.

    This class provides functionality to identify and categorize different types of character transformations,
    such as similar shapes, similar pronunciations, and common confusions in Chinese characters.

    Args:
        vocab (`dict`):
            A dictionary mapping tokens to their indices in the vocabulary.
        is_bytes_level (`bool`):
            Flag indicating whether the input is at the byte level.
        distortion_type_prior_priority (`list`, *optional*):
            A list specifying the priority order of distortion types. If not provided, a default order is used.
        config_path (`str`, *optional*, defaults to 'configs/default_config.yaml'):
            Path to the configuration file containing paths to various dictionaries and resources.

    Attributes:
        similar_shape_dict (`dict`):
            Dictionary of characters with similar shapes.
        shape_confusion_dict (`dict`):
            Dictionary of characters prone to shape-based confusion.
        similar_consonant_dict (`dict`):
            Dictionary of similar consonants in pinyin.
        similar_vowel_dict (`dict`):
            Dictionary of similar vowels in pinyin.
        similar_spell_dict (`dict`):
            Dictionary of characters with similar spellings.
        near_spell_dict (`dict`):
            Dictionary of characters with near spellings.
        prone_to_confusion_dict (`dict`):
            Dictionary of characters prone to confusion.
        vocab (`dict`):
            The input vocabulary.
        is_bytes_level (`bool`):
            Flag indicating byte-level processing.
        distortion_type_priority_order (`list`):
            Ordered list of distortion type priorities.
        distortion_type_priority (`dict`):
            Dictionary mapping distortion types to their priorities.

    Note:
        This class relies on various external resources and dictionaries for Chinese language processing,
        which should be properly configured in the specified config file.
    """

    def __init__(self, vocab, is_bytes_level, distortion_type_prior_priority=None, config_path='configs/default_config.yaml'):
        r"""
        Initializes the TransformationType class.

        Args:
            vocab (dict): A dictionary mapping tokens to their indices in the vocabulary.
            is_bytes_level (bool): Flag indicating whether the input is at the byte level.
            distortion_type_prior_priority (list, optional): A list specifying the priority order of distortion types. If not provided, a default order is used.
            config_path (str, optional): Path to the configuration file containing paths to various dictionaries and resources. Defaults to 'configs/default_config.yaml'.
        """
        # Load configuration from YAML file
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            file_paths = config['transformation_type_paths']
            self.config = config

        # Load dictionary of characters with similar shapes
        self.similar_shape_dict = self.load_dict(file_paths['similar_shape_dict'])

        # Load set of characters that should not be missing
        if 'length_immutable_chars' in file_paths:
            self.length_immutable_chars = set(self.load_list(file_paths['length_immutable_chars']))
        else:
            self.length_immutable_chars = set()

        # Load dictionary of characters with similar strokes
        self.shape_confusion_dict = self.load_dict(file_paths['shape_confusion_dict'])

        # Load dictionaries of similar consonants and vowels
        self.similar_consonant_dict = self.load_dict(file_paths['similar_consonant_dict'])
        self.similar_vowel_dict = self.load_dict(file_paths['similar_vowel_dict'])

        # Load dictionaries of similar and near spellings
        self.similar_spell_dict, self.near_spell_dict = self.load_similar_spell_dict(
            file_paths['pinyin_distance_matrix']
        )

        # Load dictionary of characters prone to confusion
        self.prone_to_confusion_dict = self.load_dict(file_paths['prone_to_confusion_dict'])

        # Store vocabulary, byte-level flag, and shape similarity threshold
        self.vocab = vocab
        self.is_bytes_level = is_bytes_level

        self.allow_insert_punct = os.getenv("ALLOW_INSERT_PUNCT", "false").lower() == "true"
        print(f"allow_insert_punct: {self.allow_insert_punct}")

        # Build distortion type priority
        self.build_distortion_type_priority(distortion_type_prior_priority)

        # Build inverse index for efficient lookup
        self.build_inverse_index()

    def build_distortion_type_priority(self, distortion_type_prior_priority):
        r"""
        Builds the distortion type priority.

        Args:
            distortion_type_prior_priority (list, optional): A list specifying the priority order of distortion types. If not provided, a default order is used.
        """
        default_distortion_type_prior_priority_order = self.config['distortion_type_prior_priority_order']

        if distortion_type_prior_priority is None:
            self.distortion_type_priority_order = default_distortion_type_prior_priority_order
            distortion_type_prior_priority = self.config['distortion_type_prior_priority']
        else:
            self.distortion_type_priority_order = [
                    distortion_type 
                    for distortion_type in distortion_type_prior_priority
                    if distortion_type in default_distortion_type_prior_priority_order
                ]

        self.distortion_type_priority = {
            distortion_type: len(self.distortion_type_priority_order) - i
            for i, distortion_type in enumerate(distortion_type_prior_priority)
        }

    def load_dict(self, file_name):
        r"""
        Loads and returns a JSON dictionary from a file.

        Args:
            file_name (str or list): The name of the file to load the dictionary from.

        Returns:
            dict: The loaded dictionary.
        """
        final_dict = {}
        if isinstance(file_name, str):
            file_name = [file_name]
        for file in file_name:
            with open(file, "r", encoding="utf-8") as f:
                final_dict.update(json.load(f))
        return final_dict
    
    def load_list(self, file_name):
        r"""
        Loads and returns a list from a json file.
        """
        final_list = []
        if isinstance(file_name, str):
            file_name = [file_name]
        for file in file_name:
            with open(file, "r", encoding="utf-8") as f:
                final_list.extend(json.load(f))
        return final_list

    def load_similar_spell_dict(self, file_name):
        r"""
        Loads the spell distance matrix from a pickle file.

        Args:
            file_name (str): The name of the file to load the spell distance matrix from.

        Returns:
            tuple: A tuple containing two dictionaries: similar_spell_dict and near_spell_dict.
        """
        self.spell_distance_matrix = pickle.load(open(file_name, "rb"))
        similar_spell_dict = defaultdict(set)
        near_spell_dict = defaultdict(set)
        # Populate similar_spell_dict based on distance threshold
        for pair, distance in self.spell_distance_matrix.items():
            if distance <= 1.0:
                similar_spell_dict[pair[0]].add(pair[1])
                similar_spell_dict[pair[1]].add(pair[0])
        return similar_spell_dict, near_spell_dict
    
    def bag_of_chars_hash(self, token):
        # Create a hash of character counts in a token
        counter = Counter(token)
        item_seq = sorted(counter.items(), key=lambda x: x[0])
        # Convert to a string representation
        item_seq = "-".join([f"{k}:{v}" for k, v in item_seq])
        return item_seq
    
    def init_pinyin_of_token_hash(self, token_consonants):
        # Create a hash of initial pinyin of a token
        new_token_consonants = []
        for c in token_consonants:
            if len(c) > 0 and len(c[0]) > 0:
                if c[0][0] in consonant_inits:
                    new_token_consonants.append(c[0][0])
                else:
                    new_token_consonants.append('1')
            else:
                new_token_consonants.append('0')
        return "-".join(new_token_consonants)

    def build_inverse_index(self):
        r"""
        Builds inverse indices for efficient lookup.

        This method constructs multiple index dictionaries that map certain features of tokens (such as pinyin, character positions, etc.) to the indices of tokens in the vocabulary.
        These indices are used to efficiently perform lookups based on various transformation types, such as identical characters, similar pinyin, similar shapes, etc.

        It processes each token in the vocabulary and builds indices for:

        - **Identical characters at specific positions** (`identical_char_index`)
        - **Characters prone to confusion at specific positions** (`prone_to_confusion_char_index`)
        - **Tokens sharing the same pinyin at specific positions** (`same_pinyin_index`)
        - **Tokens with similar pinyin at specific positions** (`similar_pinyin_index`)
        - **Tokens with pinyin that are similar due to spelling errors at specific positions** (`other_similar_pinyin_index`)
        - **Tokens with characters of similar shapes at specific positions** (`similar_shape_index`)
        - **Tokens with characters of shapes that are confused at specific positions** (`other_similar_shape_index`)
        - **Identical tokens** (`identical_token_index`)

        Additionally, it keeps track of:

        - **Token lengths** (`token_length`)
        - **Mapping of indices back to tokens** (`idx_to_token`)
        - **Set of all unique characters in tokens** (`char_set`)

        These indices facilitate quick retrieval of tokens based on various linguistic and orthographic features.
        """
        # Initialize various index dictionaries for different types of transformations
        self.identical_char_index = defaultdict(set)          # Maps (position, character) -> set of token indices
        self.prone_to_confusion_char_index = defaultdict(set) # Maps (position, confusing character) -> set of token indices
        self.same_pinyin_index = defaultdict(set)             # Maps (position, pinyin) -> set of token indices
        self.similar_pinyin_index = defaultdict(set)          # Maps (position, similar pinyin) -> set of token indices
        self.init_pinyin_index = defaultdict(set)
        self.other_similar_pinyin_index = defaultdict(set)    # Maps (position, other similar pinyin) -> set of token indices
        self.similar_shape_index = defaultdict(set)           # Maps (position, character with similar shape) -> set of token indices
        self.other_similar_shape_index = defaultdict(set)     # Maps (position, character with shape confusion) -> set of token indices
        self.reorder_index = defaultdict(set)
        self.missing_char_index = defaultdict(set)
        self.identical_token_index = defaultdict(set)         # Maps token -> set of indices
        self.idx_to_token = {}                                # Maps token index -> token
        self.token_length = {}                                # Maps token index -> token length
        self.char_set = set()                                 # Set of all unique characters in tokens
        self.is_chinese_token = {}                            # Maps token index -> is_chinese_token

        # Iterate through all vocabulary items
        for k, idx in tqdm(self.vocab.items()):
            ori_token = k  # Original token (could be bytes or string)

            # If tokens are at byte level, attempt to decode them
            if self.is_bytes_level:
                try:
                    # Try to decode byte-level token to UTF-8 string
                    ori_token = k.decode("utf-8")
                except UnicodeDecodeError:
                    # If decoding fails, handle each byte separately
                    for i, byte in enumerate(ori_token):
                        # Map the byte at position i to the token index
                        self.identical_char_index[(i, byte)].add(idx)
                    # Approximate the length of the token (assuming Chinese characters are 3 bytes)
                    self.token_length[idx] = len(ori_token) / 3
                    continue  # Skip to the next token

            # Handle special tokens that represent bytes (e.g., '<0x00>')
            if (
                len(ori_token) == 6
                and ori_token.startswith("<0x")
                and ori_token.endswith(">")
            ):
                # Assign a fractional length to represent a single byte character
                self.token_length[idx] = 1 / 3
            else:
                # For regular tokens, use the length of the token string
                self.token_length[idx] = len(ori_token)
            
            token = ori_token  # Use the potentially decoded token

            # Build identical token indices
            self.identical_token_index[token].add(idx)
            self.idx_to_token[idx] = token  # Map index back to token
            
            # Build reorder index
            self.reorder_index[self.bag_of_chars_hash(token)].add(idx)

            # Build character-level indices
            for i, char in enumerate(token):
                self.char_set.add(char)  # Keep track of unique characters
                # Map (position, character) to token index
                self.identical_char_index[(i, char)].add(idx)
                # Handle characters that are frequently confused with others
                for equal_char in self.prone_to_confusion_dict.get(char, []):
                    # Map (position, confusing character) to token index
                    self.prone_to_confusion_char_index[(i, equal_char)].add(idx)
                # Build missing character index
                if char not in self.length_immutable_chars and char not in PUNCT:
                    key = token[:i] + token[i + 1 :]
                    self.missing_char_index[key].add(idx)

            # Build pinyin-related indices
            # Get list of possible pinyins for each character in the token
            token_pinyins = pinyin(token, style=Style.NORMAL, heteronym=True)
            # If the token is not converted (e.g., not Chinese characters), skip pinyin indices
            if len(token_pinyins) == 1 and token_pinyins[0][0] == token:
                # TODO: this skip may cause the short circuit of the shape confusion
                if self.allow_insert_punct and (token in PUNCT and len(token) == 1):
                    self.is_chinese_token[idx] = True
                else:
                    self.is_chinese_token[idx] = False
                continue  # Proceed to the next token
            else:
                self.is_chinese_token[idx] = True

            # For each character position and its possible pinyins
            for i, ps in enumerate(token_pinyins):
                for p in ps:
                    # Map (position, pinyin) to token index (exact pinyin match)
                    self.same_pinyin_index[(i, p)].add(idx)
                    # For pinyins that are similar due to spelling errors
                    for similar_pinyin in self.similar_spell_dict.get(p, []):
                        # Map (position, similar pinyin) to token index
                        self.other_similar_pinyin_index[(i, similar_pinyin)].add(idx)

            # Build similar pinyin indices based on consonants and vowels
            token_consonants = pinyin(token, style=Style.INITIALS, heteronym=True)  # Possible consonants
            token_vowels = pinyin(token, style=Style.FINALS, heteronym=True)        # Possible vowels

            for i, (consonant_variants, vowel_variants) in enumerate(zip(token_consonants, token_vowels)):
                # Initialize sets for similar consonants and vowels
                similar_consonants = set(consonant_variants)
                similar_vowels = set(vowel_variants)
                # Expand similar consonants based on predefined mappings
                for c in consonant_variants:
                    similar_consonants.update(self.similar_consonant_dict.get(c, [c]))
                # Expand similar vowels based on predefined mappings
                for v in vowel_variants:
                    similar_vowels.update(self.similar_vowel_dict.get(v, [v]))
                # Combine similar consonants and vowels to generate fuzzy pinyins
                for c in similar_consonants:
                    for v in similar_vowels:
                        # Concatenate consonant and vowel to form pinyin
                        fuzzy_pinyin = c + v
                        # Map (position, fuzzy pinyin) to token index
                        self.similar_pinyin_index[(i, fuzzy_pinyin)].add(idx)

            # # Build initial pinyin index
            if len(token) > 1:
                token_consonants = pinyin(token, style=Style.NORMAL, heteronym=False)
                key = self.init_pinyin_of_token_hash(token_consonants)
                self.init_pinyin_index[key].add(idx)

            # Build shape-related indices
            for i, char in enumerate(token):
                # If character has similar-shaped characters
                if char in self.similar_shape_dict:
                    for similar_char in self.similar_shape_dict[char]:
                        # Map (position, similar-shaped character) to token index
                        self.similar_shape_index[(i, similar_char)].add(idx)
                # If character is existing in shape confusion dict
                if char in self.shape_confusion_dict:
                    for similar_char in self.shape_confusion_dict[char]:
                        # Map (position, shape-confused character) to token index
                        self.other_similar_shape_index[(i, similar_char)].add(idx)

    def handle_oov_characters(self, observed_sequence):
        r"""
        Handles out-of-vocabulary (OOV) characters.

        Args:
            observed_sequence (str or bytes): The observed sequence containing OOV characters.

        Returns:
            dict: A dictionary mapping token indices to their corresponding transformation types.
        """
        oov_transformation = {}
        if isinstance(observed_sequence[0], int):
            assert isinstance(observed_sequence, bytes)
            token_bytes = observed_sequence
        else:
            token_bytes = observed_sequence[0].encode("utf-8")
        idx = None
        for i in range(len(token_bytes)):
            tmp_byte = token_bytes[: i + 1]
            if tmp_byte in self.vocab:
                idx = self.vocab[tmp_byte]
        assert idx is not None, f"{observed_sequence} {token_bytes}"
        oov_transformation[idx] = ("IDT", )
        return oov_transformation

    def get_pinyin_data(self, observed_sequence):
        r"""
        Gets pinyin data for the observed sequence.

        Args:
            observed_sequence (str): The observed sequence.

        Returns:
            tuple: A tuple containing two elements:
                - list: A list of pinyin representations for the observed sequence.
                - list: A list of consonant representations for the observed sequence.
        """
        try:
            token_pinyins = pinyin(observed_sequence, style=Style.NORMAL, heteronym=True, errors=lambda x: [char for char in x])
            token_consonants = pinyin(observed_sequence, style=Style.NORMAL, heteronym=False, errors=lambda x: [char for char in x])
        except:
            return None, None
        return token_pinyins, token_consonants

    def handle_identical_characters(self, i, char, token_transformation):
        r"""
        Handles identical characters.

        Args:
            i (int): The index of the character in the observed sequence.
            char (str): The character in the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        for idx in self.identical_char_index.get((i, char), []):
            token_transformation[idx].setdefault(i, "IDT")

    def handle_prone_to_confusion(self, i, char, token_transformation):
        r"""
        Handles characters prone to confusion.

        Args:
            i (int): The index of the character in the observed sequence.
            char (str): The character in the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        for idx in self.prone_to_confusion_char_index.get((i, char), []):
            token_transformation[idx].setdefault(i, "PTC")

    def is_punctuation_or_space(self, char):
        r"""
        Checks if a character is a punctuation or space.

        Args:
            char (str): The character to check.

        Returns:
            bool: True if the character is a punctuation or space, False otherwise.
        """
        # return char in PUNCT and char != "_"
        return char in PUNCT
    
    def handle_continuous_punctuation_or_space(self, i, observed_sequence, token_transformation):
        r"""
        Handles continuous punctuation or space.
        """
        if not isinstance(observed_sequence, str):
            return
        if i == 0:
            for l in range(2, len(observed_sequence) + 1):
                key = observed_sequence[:l]
                for idx in self.identical_token_index.get(key, []):
                    token_transformation[idx] = {k: "IDT" for k in range(len(key))}

    def handle_redundant_before_punctuation_or_space(self, i, char, observed_sequence, token_transformation, original_token_length):
        r"""
        Handles redundant characters before punctuation or space.
        """
        if i >= 1 and char in PUNCT:
            removed_char = observed_sequence[:i]
            for l in range(1, len(observed_sequence) - i):
                key = observed_sequence[i:i+l]
                for idx in self.identical_token_index.get(key, []):
                    token_transformation[idx] = {k: "RED" for k in range(len(removed_char))}
                    original_token_length[idx] = len(key.encode("utf-8")) if self.is_bytes_level else len(key)

    def handle_same_pinyin(self, i, token_pinyins, token_transformation):
        r"""
        Handles characters with the same pinyin.

        Args:
            i (int): The index of the character in the observed sequence.
            token_pinyins (list): A list of pinyin representations for the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        for p in token_pinyins[i]:
            for idx in self.same_pinyin_index.get((i, p), []):
                token_transformation[idx].setdefault(i, "SAP")

    def handle_reorder_tokens(self, i, part_observed_sequence, token_transformation):
        r"""
        Handles reordered tokens.

        Args:
            i (int): The index of the character in the observed sequence.
            part_observed_sequence (str): The observed sequence without the character at index i.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        if i > 0:
            current_hash = self.bag_of_chars_hash(part_observed_sequence)
            for idx in self.reorder_index.get(current_hash, []):
                if (idx not in token_transformation 
                        or len(token_transformation[idx]) < self.token_length[idx]
                        or not set(token_transformation[idx].values()).issubset({"IDT", })):
                    token_transformation[idx] = {k: "ROR" for k in range(int(self.token_length[idx]))}

    def handle_initial_pinyin_match(self, i, part_consonants, token_transformation):
        r"""
        Handles initial pinyin match.
        For example, "jq" -> "机器", "精确", ...

        Args:
            i (int): The index of the character in the observed sequence.
            part_consonants (list): A list of consonant representations for the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        if i > 0:
            current_hash = self.init_pinyin_of_token_hash(part_consonants)
            imp_type_priority = self.distortion_type_priority["IMP"]
            for idx in self.init_pinyin_index.get(current_hash, []):
                if idx not in token_transformation:
                    token_transformation[idx] = {k: "IMP" for k in range(int(self.token_length[idx]))}
                elif len(token_transformation[idx]) < self.token_length[idx]:
                    token_length = int(self.token_length[idx])
                    token_trans = token_transformation.get(idx, {})
                    for range_idx in range(token_length):
                        if range_idx not in token_trans or self.distortion_type_priority[token_trans[range_idx]] < imp_type_priority:
                            token_trans[range_idx] = "IMP"
                    token_transformation[idx] = token_trans

    def handle_similar_pinyin(self, i, token_pinyins, token_transformation):
        r"""
        Handles characters with similar pinyin.

        Args:
            i (int): The index of the character in the observed sequence.
            token_pinyins (list): A list of pinyin representations for the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        for p in token_pinyins[i]:
            for idx in self.similar_pinyin_index.get((i, p), []):
                token_transformation[idx].setdefault(i, "SIP")

    def handle_similar_shape(self, i, char, token_transformation):
        r"""
        Handles characters with similar shapes.

        Args:
            i (int): The index of the character in the observed sequence.
            char (str): The character in the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        for idx in self.similar_shape_index.get((i, char), []):
            token_transformation[idx].setdefault(i, "SIS")

    def handle_other_pinyin_error(self, i, token_pinyins, token_transformation):
        r"""
        Handles other pinyin errors.

        Args:
            i (int): The index of the character in the observed sequence.
            token_pinyins (list): A list of pinyin representations for the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        for p in token_pinyins[i]:
            for idx in self.other_similar_pinyin_index.get((i, p), []):
                token_transformation[idx].setdefault(i, "OTP")

    def handle_redundant_character_inside_token(self, i, part_observed_sequence, token_transformation, original_token_length):
        r"""
        Handles redundant characters inside the token.

        Args:
            i (int): The index of the character in the observed sequence.
            part_observed_sequence (str): The observed sequence without the character at index i.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
            original_token_length (dict): A dictionary mapping token indices to their original lengths.
        """
        if i > 0:
            for j in range(1, i):
                key = part_observed_sequence[:j] + part_observed_sequence[j+1:]
                removed_char = part_observed_sequence[j]
                if removed_char in PUNCT or reAlNUM.match(removed_char):
                    # do not remove punctuation or number and english letters
                    continue
                if j == 0 and removed_char in self.length_immutable_chars:
                    continue
                for idx in self.identical_token_index.get(key, []):
                    if (idx not in token_transformation 
                            or len(token_transformation[idx]) < self.token_length[idx]
                            or not set(token_transformation[idx].values()).issubset({"IDT", })):
                        this_token_transformation = {k: "IDT" for k in range(int(self.token_length[idx]))}
                        this_token_transformation[j] = "RED"
                        token_transformation[idx] = this_token_transformation
                        original_token_length[idx] = len(part_observed_sequence.encode("utf-8")) if self.is_bytes_level else len(part_observed_sequence)

    def handle_redundant_characters(self, observed_sequence, token_transformation, original_token_length):
        # Case 2: Redundant sequences
        # The original code is too strict
        if not isinstance(observed_sequence, str):
            return
        for i in range(1, 5):
            for j in range(1, len(observed_sequence) - i):
                key = observed_sequence[i:i+j]
                if len(key) == 0:
                    continue
                removed_chars = observed_sequence[:i]
                if any([reAlNUM.match(char) for char in removed_chars]) or any([char in PUNCT for char in removed_chars]):
                    # do not remove number and english letters
                    continue
                for idx in self.identical_token_index.get(key, []):
                    if idx not in token_transformation:
                        token_transformation[idx] = {k: "RED" for k in range(len(removed_chars))}
                        replaced_chars = removed_chars + key
                        original_token_length[idx] = len(replaced_chars.encode("utf-8")) if self.is_bytes_level else len(replaced_chars)

    def handle_other_similar_shape(self, i, char, token_transformation):
        r"""
        Handles characters with other similar shapes.

        Args:
            i (int): The index of the character in the observed sequence.
            char (str): The character in the observed sequence.
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.
        """
        for idx in self.other_similar_shape_index.get((i, char), []):
            token_transformation[idx].setdefault(i, "OTS")

    def handle_missing_characters(self, observed_sequence, broken_token_transformation, original_token_length_for_broken):
        r"""
        Handles missing characters.

        Args:
            observed_sequence (str): The observed sequence.
            broken_token_transformation (set): A set of token indices with missing characters.
            original_token_length_for_broken (dict): A dictionary mapping token indices to their original lengths.
        """
        for i in range(len(observed_sequence)):
            part_observed_sequence = observed_sequence[:i+1]
            for idx in self.missing_char_index.get(part_observed_sequence, []):
                broken_token_transformation.add(idx)
                original_token_length_for_broken[idx] = len(part_observed_sequence.encode("utf-8")) if self.is_bytes_level else len(part_observed_sequence)

    def filter_and_finalize_transformations(self, token_transformation, broken_token_transformation, original_token_length, original_token_length_for_broken):
        r"""
        Filters and finalizes the transformations.

        Args:
            token_transformation (dict): A dictionary mapping token indices to their corresponding transformation types.

        Returns:
            dict: A dictionary mapping token indices to their finalized transformation types.
        """
        new_transformation = {}
        potential_transformation = {}
        for idx, transformation in token_transformation.items():
            if transformation.get(0) == "ROR":
                new_transformation[idx] = ("ROR", ) + ("IDT",) * (self.token_length[idx] - 1)
            elif len(transformation) == self.token_length[idx] or "RED" in set(transformation.values()):
                new_transformation[idx] = tuple(transformation.values())
            elif self.token_length[idx] - len(transformation) <= 2:
                potential_transformation[idx] = transformation

        for idx in broken_token_transformation:
            if idx not in new_transformation:
                new_transformation[idx] = ("MIS",) + ("IDT",) * (self.token_length[idx] - 1)
                original_token_length[idx] = original_token_length_for_broken[idx]

        for idx, transformation in potential_transformation.items():
            if idx not in new_transformation:
                if len(transformation) >= 1:
                    new_transformation[idx] = tuple(transformation.values()) + ("UNR",) * ceil(self.token_length[idx] - len(transformation))

        return new_transformation

    def handle_final_oov(self, observed_sequence):
        r"""
        Handles out-of-vocabulary (OOV) characters as a final step.

        Args:
            observed_sequence (str): The observed sequence containing OOV characters.

        Returns:
            dict: A dictionary mapping token indices to their corresponding transformation types.
        """
        new_transformation = {}
        for idx in self.identical_char_index[(0, OOV_CHAR)]:
            if self.token_length[idx] == 1:
                new_transformation[idx] = ("IDT", )  # IDT: Identical character (OOV)
        return new_transformation

    def get_transformation_type(self, observed_sequence: str):
        r"""
        Determine the transformation types for all tokens in the vocabulary to the observed sequence.

        This method analyzes the input sequence and identifies various types of character transformations
        that may have occurred, such as character substitutions, pinyin-based errors, or shape-based confusions.
        It returns a mapping of token indices to their corresponding transformation types.

        Args:
            observed_sequence (str):
                The input sequence of characters to be analyzed for transformations.

        Returns:
            Tuple[Dict[int, Tuple[str]], Dict[int, int]]:
                A tuple containing two elements:
                    - A dictionary mapping token indices to a tuple of their corresponding transformation types.
                    - A dictionary of the original token lengths (currently empty in this implementation).

        Transformation Types:
            - **IDT**: Identical character (no transformation).
            - **PTC**: Prone to confusion (commonly confused characters).
            - **SAP**: Same pinyin (characters that share the same pinyin).
            - **SIP**: Similar pinyin (characters with similar pinyin).
            - **SIS**: Similar shape (characters with similar visual appearance).
            - **OTP**: Other pinyin error (pinyin-related errors not covered by SAP or SIP).
            - **OTS**: Other similar shape (shape-related errors not covered by SIS).
            - **MIS**: Missing characters (characters that are missing from the observed sequence).
            - **RED**: Redundant characters (characters that are not needed in the observed sequence).
            - **UNR**: Unrecognized transformation (no known transformation type).

        Example:
            >>> transformer = TransformationType(vocab, is_bytes_level=False)
            >>> transformations, _ = transformer.get_transformation_type("你好")
            >>> print(transformations)
            {36371: ('IDT', 'OTP'), 8225: ('IDT', 'UNR'), ...}
        Note:
            - This method relies on prior methods such as `get_pinyin_data`, `handle_identical_characters`, 
              and various handlers for specific distortion types.
            - The `distortion_type_priority_order` attribute determines the order in which distortion handlers are applied.
        """
        # Initialize a default dictionary to hold transformations for each token index
        token_transformation = defaultdict(dict)
        # Initialize an empty dictionary for original token lengths (unused in current implementation)
        original_token_length = dict()
        # record the token indices with missing characters
        broken_token_transformation = set()
        # record the original token lengths for the tokens with missing characters
        original_token_length_for_broken = dict()

        # Initialize a variable to hold transformations for Out-Of-Vocabulary (OOV) characters
        oov_transformation = None

        # Handle Out-Of-Vocabulary characters when operating at the byte level
        if (
            self.is_bytes_level
            and len(observed_sequence) > 0
            and observed_sequence[0] not in self.char_set
        ):
            # Get OOV transformations for the observed sequence
            oov_transformation = self.handle_oov_characters(observed_sequence)

        # Retrieve pinyin data for the observed sequence
        token_pinyins, token_consonants = self.get_pinyin_data(observed_sequence)
        # Check if pinyin data retrieval was successful
        validated = token_pinyins is not None

        # Iterate over each character in the observed sequence
        for i in range(len(observed_sequence)):
            # Get the current character
            part_observed_sequence = observed_sequence[: i + 1]
            char_i = observed_sequence[i]
            
            # Handle identical characters (no transformation needed)
            self.handle_identical_characters(i, char_i, token_transformation)

            # Skip further processing if the character is punctuation or whitespace
            if self.is_punctuation_or_space(char_i):
                if 'RED' in self.distortion_type_priority_order:
                    self.handle_redundant_before_punctuation_or_space(i, char_i, observed_sequence, token_transformation, original_token_length)
                self.handle_continuous_punctuation_or_space(i, observed_sequence, token_transformation)
                break

            # If pinyin data is not valid, skip further processing for this character
            if not validated:
                continue

            # Define a mapping of distortion types to their corresponding handler methods
            distortion_handlers = {
                "PTC": lambda: self.handle_prone_to_confusion(i, char_i, token_transformation),
                "SAP": lambda: self.handle_same_pinyin(i, token_pinyins, token_transformation),
                "ROR": lambda: self.handle_reorder_tokens(i, part_observed_sequence, token_transformation),
                "SIP": lambda: self.handle_similar_pinyin(i, token_pinyins, token_transformation),
                "SIS": lambda: self.handle_similar_shape(i, char_i, token_transformation),
                "IMP": lambda: self.handle_initial_pinyin_match(i, token_consonants, token_transformation),
                "OTP": lambda: self.handle_other_pinyin_error(i, token_pinyins, token_transformation),
                "RED": lambda: self.handle_redundant_character_inside_token(i, part_observed_sequence, token_transformation, original_token_length),
                "OTS": lambda: self.handle_other_similar_shape(i, char_i, token_transformation),
            }

            # Iterate over distortion types based on their priority order
            for distortion_type in self.distortion_type_priority_order:
                if distortion_type in distortion_handlers:
                    # Invoke the handler function for the current distortion type
                    distortion_handlers[distortion_type]()

        # Handle missing characters
        if 'MIS' in self.distortion_type_priority_order:
            self.handle_missing_characters(observed_sequence, broken_token_transformation, original_token_length_for_broken)

        # Handle redundant characters
        if 'RED' in self.distortion_type_priority_order:
            self.handle_redundant_characters(observed_sequence, token_transformation, original_token_length)

        # Filter the transformations to finalize the transformation types for each token
        new_transformation = self.filter_and_finalize_transformations(token_transformation, broken_token_transformation, original_token_length, original_token_length_for_broken)

        # Incorporate OOV transformations into the final transformation mapping, if any
        if oov_transformation is not None:
            new_transformation.update(oov_transformation)

        # If no transformations were found, handle the final OOV case
        if len(new_transformation) == 0:
            new_transformation = self.handle_final_oov(observed_sequence)

        # Ensure that the transformation mapping is not empty to avoid assertion errors
        assert len(new_transformation) > 0, f"No transformations found for sequence: '{observed_sequence}'"

        return new_transformation, original_token_length
