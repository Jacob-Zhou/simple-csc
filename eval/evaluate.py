#!/usr/bin/env python
# encoding: utf-8
# Adapted from: https://github.com/nghuyong/cscd-ime/blob/master/evaluation/evaluate.py

try:
    from .utils import input_check_and_process, compute_p_r_f1, write_report, Alignment, to_halfwidth, clean_text
except ImportError:
    from utils import input_check_and_process, compute_p_r_f1, write_report, Alignment, to_halfwidth, clean_text

import argparse
from string import punctuation
from tqdm import tqdm
from opencc import OpenCC

def calculate_metric_wang(src_sentences, tgt_sentences, pred_sentences, report_file=None, ignore_chars="", strict=True, epsilon=1e-8):
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for src, tgt, predict in zip(src_sentences, tgt_sentences, pred_sentences):
        for c in ignore_chars:
            src = src.replace(c, "□")
            tgt = tgt.replace(c, "□")
            predict = predict.replace(c, "□")
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0

    TP = 0
    FP = 0
    FN = 0

    unions = list(zip(src_sentences, tgt_sentences, pred_sentences))

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(unions[i][2][j])
                if unions[i][1][j] == unions[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if unions[i][1][j]  in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall) if (correction_precision + correction_recall) > 0 else 0

    result = {
        "char detection p": round(detection_precision * 100, 3),
        "char detection r": round(detection_recall * 100, 3),
        "char detection f1": round(detection_f1 * 100, 3),
        "char correction p": round(correction_precision * 100, 3),
        "char correction r": round(correction_recall * 100, 3),
        "char correction f1": round(correction_f1 * 100, 3),
    }

    if report_file:
        write_report(report_file, result, [])
    return result

def calculate_metric_conventional(src_sentences, tgt_sentences, pred_sentences, report_file=None, ignore_chars="", strict=True, epsilon=1e-8):
    corrected_char = 0                      # 预测结果中修改了的字符数量
    wrong_char = 0                          # 原本就需要纠正的字符数量
    corrected_sent = 0                      # 预测结果中修改了的句子数量
    wrong_sent = 0                          # 原本就需要纠正的句子数量
    true_corrected_char = 0                 # 预测结果中修改对的字符数量
    true_corrected_sent = 0                 # 预测结果中修改对的句子数量
    true_detected_char = 0                  # 预测结果中检查对的字符数量
    true_detected_sent = 0                  # 预测结果中检查对的句子数量
    accurate_detected_sent, accurate_corrected_sent = 0, 0
    all_char, all_sent = 0, 0

    for src, tgt, pre in zip(src_sentences, tgt_sentences, pred_sentences):
        for c in ignore_chars:
            src = src.replace(c, "□")
            tgt = tgt.replace(c, "□")
            pre = pre.replace(c, "□")
        src = "".join(src.split())
        tgt = "".join(tgt.split())
        pre = "".join(pre.split())
        all_sent += 1

        wrong_num = 0
        corrected_num = 0
        original_wrong_num = 0
        true_detected_char_in_sentence = 0

        for s, t, p in zip(src, tgt, pre):
            all_char += 1
            if t != p:
                wrong_num += 1
            if s != p:
                corrected_num += 1
                if t == p:
                    true_corrected_char += 1
                if s != t:
                    true_detected_char += 1
                    true_detected_char_in_sentence += 1
            if t != s:
                original_wrong_num += 1

        corrected_char += corrected_num
        wrong_char += original_wrong_num
        if original_wrong_num != 0:
            wrong_sent += 1
        if corrected_num != 0 and wrong_num == 0:
            true_corrected_sent += 1

        if corrected_num != 0:
            corrected_sent += 1

        if strict:
            true_detected_flag = (true_detected_char_in_sentence == original_wrong_num == corrected_num and original_wrong_num != 0)
        else:
            true_detected_flag = (
                corrected_num != 0 and original_wrong_num != 0)
        if true_detected_flag:
            true_detected_sent += 1

        if tgt == pre:
            accurate_corrected_sent += 1
        if tgt == pre or true_detected_flag:
            accurate_detected_sent += 1

    det_char_pre = true_detected_char / (corrected_char + epsilon)
    det_char_rec = true_detected_char / (wrong_char + epsilon)
    det_char_f1 = 2 * det_char_pre * det_char_rec / \
        (det_char_pre + det_char_rec + epsilon)
    cor_char_pre = true_corrected_char / (corrected_char + epsilon)
    cor_char_rec = true_corrected_char / (wrong_char + epsilon)
    cor_char_f1 = 2 * cor_char_pre * cor_char_rec / \
        (cor_char_pre + cor_char_rec + epsilon)

    det_sent_acc = accurate_detected_sent / (all_sent + epsilon)
    det_sent_pre = true_detected_sent / (corrected_sent + epsilon)
    det_sent_rec = true_detected_sent / (wrong_sent + epsilon)
    det_sent_f1 = 2 * det_sent_pre * det_sent_rec / \
        (det_sent_pre + det_sent_rec + epsilon)
    cor_sent_acc = accurate_corrected_sent / (all_sent + epsilon)
    cor_sent_pre = true_corrected_sent / (corrected_sent + epsilon)
    cor_sent_rec = true_corrected_sent / (wrong_sent + epsilon)
    cor_sent_f1 = 2 * cor_sent_pre * cor_sent_rec / \
        (cor_sent_pre + cor_sent_rec + epsilon)

    result = {
        "sentence detection p": round(det_sent_pre * 100, 3),
        "sentence detection r": round(det_sent_rec * 100, 3),
        "sentence detection f1": round(det_sent_f1 * 100, 3),
        "sentence detection acc": round(det_sent_acc * 100, 3),
        "sentence correction p": round(cor_sent_pre * 100, 3),
        "sentence correction r": round(cor_sent_rec * 100, 3),
        "sentence correction f1": round(cor_sent_f1 * 100, 3),
        "sentence correction acc": round(cor_sent_acc * 100, 3),
        "char detection p": round(det_char_pre * 100, 3),
        "char detection r": round(det_char_rec * 100, 3),
        "char detection f1": round(det_char_f1 * 100, 3),
        "char correction p": round(cor_char_pre * 100, 3),
        "char correction r": round(cor_char_rec * 100, 3),
        "char correction f1": round(cor_char_f1 * 100, 3),
    }
    
    if report_file:
        write_report(report_file, result, [])
    return result

def calculate_metric_official(src_sentences, tgt_sentences, pred_sentences, report_file=None, ignore_chars=""):
    def compute_fpr_acc_pre_rec_f1(tp, fp, tn, fn, epsilon=1e-8):
        fpr = fp / (fp + tn + epsilon)
        acc = (tp + tn) / (tp + fp + tn + fn + epsilon)
        pre = tp / (tp + fp + epsilon)
        rec = tp / (tp + fn + epsilon)
        f1 = 2 * pre * rec / (pre + rec + epsilon)
        return fpr, acc, pre, rec, f1
    truth = []
    for index, (src, tgt, pre) in enumerate(zip(src_sentences, tgt_sentences, pred_sentences)):
        for c in ignore_chars:
            src = src.replace(c, "□")
            tgt = tgt.replace(c, "□")
            pre = pre.replace(c, "□")
        src = "".join(src.split())
        tgt = "".join(tgt.split())
        pre = "".join(pre.split())
        char_index = 0
        sent_gold = []
        sent_correct_flag = 0
        temp = []
        for s, t, p in zip(src, tgt, pre):
            char_index += 1
            if s != t:
                sent_correct_flag = 1
                temp.append(str(char_index) + ", " + t)
            else:
                continue
        sent_gold.extend(", ".join(temp))
        if sent_correct_flag == 0:
            sent_gold.extend("0")
        truth.append(str(index) + ", " + "".join(sent_gold).strip())

    predict = []
    for index, (src, tgt, pre) in enumerate(zip(src_sentences, tgt_sentences, pred_sentences)):
        char_index = 0
        sent_pre = []
        sent_correct_flag = 0
        temp = []
        for s, t, p in zip(src, tgt, pre):
            char_index += 1
            if s != p:
                sent_correct_flag = 1
                temp.append(str(char_index) + ", " + p)
            else:
                continue
        sent_pre.extend(", ".join(temp))
        if sent_correct_flag == 0:
            sent_pre.extend("0")
        predict.append(str(index) + ", " + "".join(sent_pre).strip())

    truth_dict = {}
    for sent in truth:
        sent_gold = sent.split(", ")
        if len(sent_gold) == 2:
            truth_dict[sent_gold[0]] = "0"
        else:
            content = {}
            truth_dict[sent_gold[0]] = content
            sent_gold = sent_gold[1:]
            for i in range(0, len(sent_gold), 2):
                content[sent_gold[i]] = sent_gold[i + 1]

    predict_dict = {}
    for sent in predict:
        sent_pre = sent.split(", ")
        if len(sent_pre) == 2:
            predict_dict[sent_pre[0]] = "0"
        else:
            content = {}
            predict_dict[sent_pre[0]] = content
            sent_pre = sent_pre[1:]
            for i in range(0, len(sent_pre), 2):
                content[sent_pre[i]] = sent_pre[i + 1]

    dtp, dfp, dtn, dfn = 0, 0, 0, 0
    ctp, cfp, ctn, cfn = 0, 0, 0, 0

    assert len(truth_dict) == len(predict_dict)

    for i in range(len(truth_dict)):
        gold = truth_dict[str(i)]
        pre = predict_dict[str(i)]
        if gold == "0":
            if pre == "0":
                dtn += 1
                ctn += 1
            else:
                dfp += 1
                cfp += 1
        elif pre == "0":
            dfn += 1
            cfn += 1
        elif len(gold) == len(pre) and gold.keys() == pre.keys():
            dtp += 1
            if list(gold.values()) == list(pre.values()):
                ctp += 1
            else:
                cfn += 1
        else:
            dfn += 1
            cfn += 1

    dfpr, dacc, dpre, drec, df1 = compute_fpr_acc_pre_rec_f1(dtp, dfp, dtn, dfn)
    cfpr, cacc, cpre, crec, cf1 = compute_fpr_acc_pre_rec_f1(ctp, cfp, ctn, cfn)

    result = {
        "sentence detection fpr": round(dfpr * 100, 3),
        "sentence detection acc": round(dacc * 100, 3),
        "sentence detection p": round(dpre * 100, 3),
        "sentence detection r": round(drec * 100, 3),
        "sentence detection f1": round(df1 * 100, 3),
        "sentence correction fpr": round(cfpr * 100, 3),
        "sentence correction acc": round(cacc * 100, 3),
        "sentence correction p": round(cpre * 100, 3),
        "sentence correction r": round(crec * 100, 3),
        "sentence correction f1": round(cf1 * 100, 3),
    }
    
    if report_file:
        write_report(report_file, result, [])
    return result

def calculate_metric(src_sentences, tgt_sentences, pred_sentences, report_file=None, ignore_chars=""):
    """
    :param src_sentences: list of origin sentences
    :param tgt_sentences: list of target sentences
    :param pred_sentences: list of predict sentences
    :param report_file: report file path
    :param ignore_chars: chars that is not evaluated
    :return:
    """
    src_char_list, tgt_char_list, pred_char_list = input_check_and_process(src_sentences, tgt_sentences, pred_sentences)
    sentence_detection, sentence_correction, char_detection, char_correction = [
        {'all_error': 0, 'true_predict': 0, 'all_predict': 0} for _ in range(4)]
    n_not_error = 0
    n_false_pred = 0
    output_errors = []
    for src_chars, tgt_chars, pred_chars in tqdm(zip(src_char_list, tgt_char_list, pred_char_list), total=len(pred_char_list)):
        true_error_indexes = set()
        true_error_edits = set()
        detect_indexes = set()
        detect_edits = set()

        gold_edits = Alignment(src_chars, tgt_chars).align_seq
        pred_edits = Alignment(src_chars, pred_chars).align_seq

        for gold_edit in gold_edits:
            edit_type, b_ori, e_ori, b_prd, e_prd = gold_edit
            if edit_type != 'M':
                src_char = ''.join(src_chars[b_ori:e_ori])
                if len(src_char) > 0 and src_char in ignore_chars:
                    continue
                char_detection['all_error'] += 1
                char_correction['all_error'] += 1
                true_error_indexes.add((b_ori, e_ori))
                true_error_edits.add((b_ori, e_ori, tuple(tgt_chars[b_prd:e_prd])))

        for pred_edit in pred_edits:
            edit_type, b_ori, e_ori, b_prd, e_prd = pred_edit
            if edit_type != 'M':
                src_char = ''.join(src_chars[b_ori:e_ori])
                if len(src_char) > 0 and src_char in ignore_chars:
                    continue
                char_detection['all_predict'] += 1
                char_correction['all_predict'] += 1
                detect_indexes.add((b_ori, e_ori))
                detect_edits.add((b_ori, e_ori, tuple(pred_chars[b_prd:e_prd])))
                if (b_ori, e_ori) in true_error_indexes:
                    char_detection['true_predict'] += 1
                if (b_ori, e_ori, tuple(pred_chars[b_prd:e_prd])) in true_error_edits:
                    char_correction['true_predict'] += 1

        if true_error_indexes:
            sentence_detection['all_error'] += 1
            sentence_correction['all_error'] += 1
        else:
            n_not_error += 1
            if detect_indexes:
                n_false_pred += 1
        if detect_indexes:
            sentence_detection['all_predict'] += 1
            sentence_correction['all_predict'] += 1
            if tuple(true_error_indexes) == tuple(detect_indexes):
                sentence_detection['true_predict'] += 1
            if tuple(true_error_edits) == tuple(detect_edits):
                sentence_correction['true_predict'] += 1

        origin_s = "".join(src_chars)
        target_s = "".join(tgt_chars)
        predict_s = "".join(pred_chars)
        if target_s == predict_s:
            error_type = "正确"
        elif origin_s == target_s and origin_s != predict_s:
            error_type = "过纠"
        elif origin_s != target_s and origin_s == predict_s:
            error_type = "漏纠"
        else:
            error_type = '综合'
        output_errors.append(
            [
                "原始: " + "".join(src_chars),
                "正确: " + "".join(["".join(tgt_chars[t_b:t_e]) if (s_b, s_e) not in true_error_indexes else f"【{''.join(tgt_chars[t_b:t_e])}】" for _, s_b, s_e, t_b, t_e in gold_edits]),
                "预测: " + "".join(["".join(pred_chars[t_b:t_e]) if (s_b, s_e) not in detect_indexes else f"【{''.join(pred_chars[t_b:t_e])}】" for _, s_b, s_e, t_b, t_e in pred_edits]),
                "错误类型: " + error_type,
            ]
        )

    result = dict()
    prefix_names = [
        "sentence detection ",
        "sentence correction ",
        "char detection ",
        "char correction ",
    ]
    for prefix_name, sub_metric in zip(prefix_names,
                                       [sentence_detection, sentence_correction, char_detection, char_correction]):
        sub_r = compute_p_r_f1(sub_metric['true_predict'], sub_metric['all_predict'], sub_metric['all_error']).items()
        for k, v in sub_r:
            result[prefix_name + k] = v
    
    result["sentence fpr"] = round(100 * n_false_pred / (n_not_error + 1e-10), 3)

    if report_file:
        write_report(report_file, result, output_errors)
    return result

def main(args):
    """
    evaluate
    :return:
    """
    src_sentences, tgt_sentences, pred_sentences = [], [], []
    ignored_indexes = set()
    cc = OpenCC("t2s")

    with open(args.gold, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if len(line) <= 0:
                continue
            items = line.strip().split('\t')
            if len(items) == 2:
                origin, corrected = line.strip().split('\t')
            elif len(items) == 3:
                _, origin, corrected = line.strip().split('\t')
            else:
                raise ValueError()
            origin = clean_text(origin).strip()
            corrected = clean_text(corrected).strip()
            if args.ignore_unmatch_length and len(origin) != len(corrected):
                ignored_indexes.add(idx)
            elif args.max_length > 0 and len(origin) > args.max_length:
                ignored_indexes.add(idx)
            else:
                if args.to_simplified:
                    origin = cc.convert(origin)
                    corrected = cc.convert(corrected)
                if args.to_halfwidth:
                    origin = to_halfwidth(origin)
                    corrected = to_halfwidth(corrected)
                if args.ignore_space:
                    origin = origin.replace(" ", '')
                    corrected = corrected.replace(" ", '')
                src_sentences.append(origin)
                tgt_sentences.append(corrected)
    with open(args.hypo, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in ignored_indexes:
                continue
            prediction = clean_text(line).strip()
            if args.to_simplified:
                prediction = cc.convert(prediction)
            if args.to_halfwidth:
                prediction = to_halfwidth(prediction)
            if args.ignore_space:
                prediction = prediction.replace(" ", '')
            pred_sentences.append(prediction)
    if len(ignored_indexes) > 0:
        print(f"Ingored {len(ignored_indexes)} instances")
    chars_to_ignore = set(args.ignore_chars)
    if args.ignore_punct:
        chinese_punct = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"
        english_punct = punctuation
        punct_set = set(chinese_punct + english_punct)
        chars_to_ignore = chars_to_ignore.union(punct_set)
    # if args.to_halfweight:
    #     pass
    if args.metric_algorithm == "levenshtein":
        metric = calculate_metric
    elif args.metric_algorithm == "conventional":
        metric = calculate_metric_conventional
    elif args.metric_algorithm == "official":
        metric = calculate_metric_official
    elif args.metric_algorithm == "wang":
        metric = calculate_metric_wang
    else:
        raise NotImplementedError
    result = metric(src_sentences, tgt_sentences, pred_sentences, args.output, chars_to_ignore)
    for key in sorted(result.keys()):
        print(f'{key}:\t{result[key]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str)
    parser.add_argument('--hypo', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--metric_algorithm', type=str,                       
        default="levenshtein",
        choices=[
            "levenshtein", "conventional", "official", "wang"
        ])
    parser.add_argument('--ignore_unmatch_length', action='store_true')
    parser.add_argument('--ignore_punct', action='store_true')
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--to_simplified', action='store_true')
    parser.add_argument('--to_halfwidth', action='store_true')
    parser.add_argument('--ignore_chars', type=str, default="")
    parser.add_argument('--ignore_space', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output = ".".join(args.hypo.split(".")[:-1]) + ".result"
    main(args)
