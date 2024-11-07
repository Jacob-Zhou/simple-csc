from collections import defaultdict
import json
import pickle

import pypinyin
from pypinyin_dict.pinyin_data import ktghz2013
from pypinyin_dict.phrase_pinyin_data import large_pinyin

from tqdm import tqdm

# load better pinyin data
ktghz2013.load()
# load better phrase pinyin data
large_pinyin.load()

RADICAL_INDEX = 0
COMPONENT_INDEX = 1
shape_similar_threshold = 0.45

def load_dict(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)

def load_similar_spell_dict(file_name):
    spell_distance_matrix = pickle.load(open(file_name, "rb"))
    similar_spell_dict = defaultdict(set)
    for pair, distance in spell_distance_matrix.items():
        if distance <= 1.0:
            similar_spell_dict[pair[0]].add(pair[1])
            similar_spell_dict[pair[1]].add(pair[0])
    return spell_distance_matrix, similar_spell_dict
    
radical_dict = load_dict("data/radical_dict.json")
four_corner_dict = load_dict("data/four_corner_dict.json")
similar_shape_dict = load_dict("data/similar_shape_dict.json")
shape_confusion_dict = load_dict("data/shape_confusion_dict.json")
similar_consonant_dict = load_dict("data/similar_consonant_dict.json")
similar_vowel_dict = load_dict("data/similar_vowel_dict.json")
spell_distance_matrix, similar_spell_dict = load_similar_spell_dict(
    "data/pinyin_distance_matrix.pkl"
)

char_set = set(radical_dict.keys()) | set(four_corner_dict.keys()) | set(map(chr, pypinyin.pinyin_dict.pinyin_dict.keys()))

four_corner_index = defaultdict(set)
radical_index = defaultdict(set)
part_of_radical_index = defaultdict(set)

for char in tqdm(char_set):
    # Structure index
    if char in four_corner_dict:
        fc = four_corner_dict[char]
        assert len(fc) == 5, f"{char} {fc}"
        for j, c in enumerate(fc[:-1]):
            four_corner_index[(j, c)].add(char)
    if char in radical_dict:
        for radical, components in radical_dict[char]:
            radical_index[(RADICAL_INDEX, 0, radical)].add(char)
            for j, component in enumerate(components):
                radical_index[(COMPONENT_INDEX, j, component)].add(char)
            # Deal with pair like 扯 <-> 止
            if len(components) == 1:
                part_of_radical_index[(components[0])].add(char)

similar_shape = defaultdict(set)

for char in tqdm(char_set):
    structure_similarity = defaultdict(int)
    if char in four_corner_dict:
        four_corner_similarity = defaultdict(int)
        fc = four_corner_dict[char]
        assert len(fc) == 5, f"{char} {fc}"
        for j, c in enumerate(fc[:-1]):
            if (j, c) in four_corner_index:
                for similar_char in four_corner_index[(j, c)]:
                    four_corner_similarity[similar_char] += 1
        for similar_char, count in four_corner_similarity.items():
            structure_similarity[similar_char] += 0.5 * (count / 4)
    if char in radical_dict:
        radical_similarity = defaultdict(int)
        for radical, components in radical_dict[char]:
            local_similarity = defaultdict(int)
            if (RADICAL_INDEX, 0, radical) in radical_index:
                for similar_char in radical_index[(RADICAL_INDEX, 0, radical)]:
                    local_similarity[similar_char] += 0.5
            for j, component in enumerate(components):
                if (COMPONENT_INDEX, j, component) in radical_index:
                    for similar_char in radical_index[
                        (COMPONENT_INDEX, j, component)
                    ]:
                        local_similarity[similar_char] += 0.5 / len(components)
            for similar_char, score in local_similarity.items():
                if score > radical_similarity[similar_char]:
                    radical_similarity[similar_char] = score
        if char in part_of_radical_index:
            for similar_char in part_of_radical_index[char]:
                radical_similarity[similar_char] = 0.75
        for similar_char, score in radical_similarity.items():
            structure_similarity[similar_char] += 0.5 * score
    similar_shape_set = {
        similar_char for similar_char, score in structure_similarity.items() if score > shape_similar_threshold
    }
    for similar_char in similar_shape_set:
        similar_shape[similar_char].add(char)

similar_shape = {k: list(v) for k, v in similar_shape.items()}
with open("data/similar_shape_dict.json", "w", encoding="utf-8") as f:
    json.dump(similar_shape, f, ensure_ascii=False, indent=2)