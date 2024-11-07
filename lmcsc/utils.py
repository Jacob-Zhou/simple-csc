import gc
import os
import torch

def clean_sentences(sentences):
    """
    Cleans a list of sentences by converting full-width characters to half-width characters.

    Args:
        sentences (list of str): List of sentences to be cleaned.

    Returns:
        tuple: A tuple containing:
            - new_sentences (list of str): List of cleaned sentences.
            - changes (list of list of tuple): List of changes made to each sentence. Each change is represented as a tuple
              (start_index, end_index, new_char, original_char).
    """
    new_sentences = []
    changes = []
    for sentence in sentences:
        new_sentence, change = clean_sentence(sentence)
        new_sentences.append(new_sentence)
        changes.append(change)
    return new_sentences, changes


def clean_sentence(sentence):
    """
    Cleans a single sentence by converting full-width characters to half-width characters.

    Args:
        sentence (str): The sentence to be cleaned.

    Returns:
        tuple: A tuple containing:
            - new_sentence (str): The cleaned sentence.
            - change (list of tuple): List of changes made to the sentence. Each change is represented as a tuple
              (start_index, end_index, new_char, original_char).
    """
    chars = []
    index = 0
    change = []
    for char in sentence:
        # Check if the character is a full-width character and not in the excluded set
        if ("\uff01" <= char <= "\uff5e") and char not in {
            "。",
            "，",
            "；",
            "：",
            "（",
            "）",
            "！",
            "？",
        }:
            new_char = chr(ord(char) - 0xFEE0)  # Convert to half-width character
            change.append((index, index + len(new_char), new_char, char))
            char = new_char
        index += len(char)
        chars.append(char)
    new_sentence = "".join(chars)
    assert index == len(new_sentence)  # Ensure the index matches the length of the new sentence
    return new_sentence, change


def rebuild_sentences(sentences, changes):
    """
    Rebuilds a list of sentences by reverting changes made by the model.

    Args:
        sentences (list of str): List of sentences to be rebuilt.
        changes (list of list of tuple): List of changes made to each sentence. Each change is represented as a tuple
              (start_index, end_index, new_char, original_char).

    Returns:
        list of str: List of rebuilt sentences.
    """
    new_sentences = []
    for sentence, change in zip(sentences, changes):
        new_sentence = rebuild_sentence(sentence, change)
        new_sentences.append(new_sentence)
    return new_sentences


def rebuild_sentence(sentence, change):
    """
    Rebuilds a single sentence by reverting changes made by the model.

    Args:
        sentence (str): The sentence to be rebuilt.
        change (list of tuple): List of changes made to the sentence. Each change is represented as a tuple
              (start_index, end_index, new_char, original_char).

    Returns:
        str: The rebuilt sentence.
    """
    for i_start, i_end, new_char, char in reversed(change):
        # If the character has been changed by the model, skip
        if sentence[i_start:i_end] == new_char:
            sentence = sentence[:i_start] + char + sentence[i_end:]
    return sentence


# Modified from https://github.com/huggingface/transformers/blob/8e9a2207b3e77eb586b2488aced2a748dde865ff/src/transformers/models/gpt2/tokenization_gpt2.py#L38
def get_vocab_decoder(vocab):
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    decoder = dict(zip(cs, bs))

    bytes_decoder = {}
    new_vocab = {}
    for k, idx in vocab.items():
        b = bytes(bytearray([decoder[u] for u in k]))
        bytes_decoder[idx] = b
        new_vocab[b] = idx
    return new_vocab, bytes_decoder


def qwen1_5_convert_ids_to_tokens(ids, decoder):
    return [decoder[id] for id in ids]


def is_chinese_word(text):
    return all([is_chinese_char(ord(char)) for char in text])


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False

def use_modelscope() -> bool:
    return bool(int(os.environ.get("USE_MODELSCOPE_HUB", "0")))

def try_download_model_from_ms(model_name_or_path, model_revision=None, cache_dir=None) -> None:
    if not use_modelscope() or os.path.exists(model_name_or_path):
        return

    try:
        from modelscope import snapshot_download

        revision = "master" if model_revision == "main" else model_revision
        model_name_or_path = snapshot_download(
            model_name_or_path, revision=revision, cache_dir=cache_dir
        )
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")

def is_nearby_pinyin(p1: str, p2: str) -> bool:
    """
    Determines whether two pinyin strings are nearby on a QWERTY keyboard.

    This function checks if the given pinyin strings are either identical or
    represent keys that are adjacent on a standard QWERTY keyboard layout.
    It also handles special cases for 'zh', 'ch', and 'sh' pinyin initials.

    Args:
        p1 (str): The first pinyin string to compare.
        p2 (str): The second pinyin string to compare.

    Returns:
        bool: True if the pinyin strings are nearby on the keyboard, False otherwise.

    Examples:
        >>> is_nearby_pinyin('qi', 'qi')
        True
        >>> is_nearby_pinyin('qi', 'si')
        True
        >>> is_nearby_pinyin('zhen', 'zen')
        True
        >>> is_nearby_pinyin('fu', 'pu')
        False
    """
    # Check if the pinyin strings are identical
    if p1 == p2:
        return True

    # Handle special cases for 'zh', 'ch', 'sh'
    if len(p1) != len(p2):
        p1, p2 = sorted([p1, p2], key=lambda x: len(x))
        if p2[0] in {'z', 'c', 'x'} and p2[1] == 'h':
            p1 = p1[:1]
            if p1 == p2:
                return True
            elif len(p1) != len(p2):
                return False
        else:
            return False

    # Define the QWERTY keyboard layout
    t26_keyboard = {
        'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6),
        'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
        'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4), 'h': (1, 5), 'j': (1, 6),
        'k': (1, 7), 'l': (1, 8),
        'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4), 'n': (2, 5), 'm': (2, 6)
    }

    # Check if all characters in both pinyin strings are valid keyboard keys
    if any(c not in t26_keyboard for c in p1 + p2):
        return False

    # Find the first differing character pair
    diff_c = [(c1, c2) for c1, c2 in zip(p1, p2) if c1 != c2]
    if len(diff_c) == 0:
        return True  # All characters are the same
    elif len(diff_c) > 1:
        return False

    c1, c2 = diff_c[0]

    # Calculate the Manhattan distance between the differing keys
    d_x = abs(t26_keyboard[c1][0] - t26_keyboard[c2][0])
    d_y = abs(t26_keyboard[c1][1] - t26_keyboard[c2][1])

    # Return True if the keys are adjacent (including diagonally)
    return max(d_x, d_y) <= 1

class Alignment:
    # Alignment adapted from: https://github.com/chrisjbryant/errant/blob/main/errant/alignment.py
    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A flag for standard Levenshtein alignment
    def __init__(self, orig, cor):
        # Set orig and cor
        self.orig = orig
        self.cor = cor
        # Align orig and cor and get the cost and op matrices
        self.cost_matrix, self.op_matrix = self.align()
        # Get the cheapest align sequence from the op matrix
        self.align_seq = self.get_cheapest_align_seq()

    # Input: A flag for standard Levenshtein alignment
    # Output: The cost matrix and the operation matrix of the alignment
    def align(self):
        # Sentence lengths
        o_len = len(self.orig)
        c_len = len(self.cor)
        # Lower case token IDs (for transpositions)
        o_low = [o.lower() for o in self.orig]
        c_low = [c.lower() for c in self.cor]
        # Create the cost_matrix and the op_matrix
        cost_matrix = [[0.0 for j in range(c_len + 1)]
                       for i in range(o_len + 1)]
        op_matrix = [["O" for j in range(c_len + 1)] for i in range(o_len + 1)]
        # Fill in the edges
        for i in range(1, o_len + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            op_matrix[i][0] = "D"
        for j in range(1, c_len + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            op_matrix[0][j] = "I"

        # Loop through the cost_matrix
        for i in range(o_len):
            for j in range(c_len):
                # Matches
                if self.orig[i] == self.cor[j]:
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                    op_matrix[i + 1][j + 1] = "M"
                # Non-matches
                else:
                    del_cost = cost_matrix[i][j + 1] + 1
                    ins_cost = cost_matrix[i + 1][j] + 1
                    trans_cost = float("inf")
                    # Standard Levenshtein (S = 1)
                    sub_cost = cost_matrix[i][j] + 1

                    # Costs
                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    # Get the index of the cheapest (first cheapest if tied)
                    l = costs.index(min(costs))
                    # Save the cost and the op in the matrices
                    cost_matrix[i + 1][j + 1] = costs[l]
                    if l == 0: op_matrix[i + 1][j + 1] = "T" + str(k + 1)
                    elif l == 1: op_matrix[i + 1][j + 1] = "S"
                    elif l == 2: op_matrix[i + 1][j + 1] = "I"
                    else: op_matrix[i + 1][j + 1] = "D"
        # Return the matrices
        return cost_matrix, op_matrix

    # Get the cheapest alignment sequence and indices from the op matrix
    # align_seq = [(op, o_start, o_end, c_start, c_end), ...]
    def get_cheapest_align_seq(self):
        i = len(self.op_matrix) - 1
        j = len(self.op_matrix[0]) - 1
        align_seq = []
        # Work backwards from bottom right until we hit top left
        while i + j != 0:
            # Get the edit operation in the current cell
            op = self.op_matrix[i][j]
            # Matches and substitutions
            if op in {"M", "S"}:
                align_seq.append((op, i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            # Deletions
            elif op == "D":
                align_seq.append((op, i - 1, i, j, j))
                i -= 1
            # Insertions
            elif op == "I":
                align_seq.append((op, i, i, j - 1, j))
                j -= 1
            # Transpositions
            else:
                # Get the size of the transposition
                k = int(op[1:])
                align_seq.append((op, i - k, i, j - k, j))
                i -= k
                j -= k
        # Reverse the list to go from left to right and return
        align_seq.reverse()
        return align_seq

class measure_cuda_memory:
    def __init__(self, device=None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Returns the current memory usage in bytes for the current device
        mem = torch.cuda.max_memory_allocated(self.device)
        return mem
    
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self.initial_memory = self.current_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # For garbage collection
        for _ in range(10):
            torch.cuda.empty_cache()
        gc.collect()