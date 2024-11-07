#!/usr/bin/env python
# encoding: utf-8
# Adapted from: https://github.com/nghuyong/cscd-ime/blob/master/evaluation/util.py

import unicodedata

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


def compute_p_r_f1(true_predict, all_predict, all_error):
    """
    @param true_predict:
    @param all_predict:
    @param all_error:
    @return:
    """
    if all_predict == 0:
        p = 0.0
    else:
        p = round(true_predict / all_predict * 100, 3)
    if all_error == 0:
        r = 0.0
    else:
        r = round(true_predict / all_error * 100, 3)
    f1 = round(2 * p * r / (p + r + 1e-10), 3)
    return {'p': p, 'r': r, 'f1': f1}


def write_report(output_file, metric, output_errors):
    """
    generate report
    @param output_file:
    @param metric:
    @param output_errors:
    @return:
    """
    with open(output_file, 'wt', encoding='utf-8') as f:
        f.write('overview:\n')
        for key in sorted(metric.keys()):
            f.write(f'{key}:\t{metric[key]}\n')
        f.write('\nbad cases:\n')
        for output_error in output_errors:
            f.write("\n".join(output_error))
            f.write("\n\n")


def input_check_and_process(src_sentences, tgt_sentences, pred_sentences):
    """
    check the input is valid
    @param src_sentences:
    @param tgt_sentences:
    @param pred_sentences:
    @return:
    """
    assert len(src_sentences) == len(tgt_sentences) == len(pred_sentences)
    src_char_list = [list(s) for s in src_sentences]
    tgt_char_list = [list(s) for s in tgt_sentences]
    pred_char_list = [list(s) for s in pred_sentences]
    return src_char_list, tgt_char_list, pred_char_list

def to_halfwidth_char(char):
    if u"\uff01" <= char <= u"\uff5e":
        return chr(ord(char) - 0xfee0)
    else:
        return char

def to_halfwidth(sentence):
    return "".join([to_halfwidth_char(char) for char in sentence])

def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)