import os

def clean_sentences(sentences):
    new_sentences = []
    changes = []
    for sentence in sentences:
        new_sentence, change = clean_sentence(sentence)
        new_sentences.append(new_sentence)
        changes.append(change)
    return new_sentences, changes


def clean_sentence(sentence):
    chars = []
    index = 0
    change = []
    for char in sentence:
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
            new_char = chr(ord(char) - 0xFEE0)
            change.append((index, index + len(new_char), new_char, char))
            char = new_char
        index += len(char)
        chars.append(char)
    new_sentence = "".join(chars)
    assert index == len(new_sentence)
    return new_sentence, change


def rebuild_sentences(sentences, changes):
    new_sentences = []
    for sentence, change in zip(sentences, changes):
        new_sentence = rebuild_sentence(sentence, change)
        new_sentences.append(new_sentence)
    return new_sentences


def rebuild_sentence(sentence, change):
    for i_start, i_end, new_char, char in reversed(change):
        # if has been changed by model, skip
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