## CCTC datasets

## path: "datasets/cctc/cctc_test_wide.jsonl"
#    - "cctc/test.txt"
## format (jsonl):
# {"sentences": [["期货大讲堂"], ["大家好,今天的期货大讲堂,我们来说说期货与股票风险的不同。", "股票波动小,短期风险并不大,获利也相对平稳。", "但是一旦被套,长期风险相当大。", "期货波动快,表面上风险比股票大。", "但是期货当天可以开平。", "即使出现浮动亏损,也能随时止损出局甚至反手开仓。", "因此长期风险比股票小。"], ["大家明白了吗？"], ["明天的期货大讲堂,我们来说说期货与股票区别的最后一点,也就是交易品种的不同。", "敬请关注。", "希望明天过后大家都会对期货有一定的认识。"]], "corrections": [[[]], [[], [], [], [], [], [], []], [[]], [[], [], []]], "doc_id": "4213"}
#}

import json
import argparse
import os

def extract_cctc(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    output_file = open(output_file, 'w', encoding='utf-8')

    index = 1
    for item in data:
        paragraphs = item['sentences']
        paragraph_corrections = item['corrections']
        for paragraph, paragraph_correction in zip(paragraphs, paragraph_corrections):
            for sentence, sentence_correction in zip(paragraph, paragraph_correction):
                # reorder sentence_correction from end to start
                sentence_correction = sorted(sentence_correction, key=lambda x: x[0], reverse=True)
                source = sentence
                target = sentence
                for edit in sentence_correction:
                    position, error_type, erroneous_tokens, correct_tokens = edit
                    target = target[:position-1] + correct_tokens + target[position-1 + len(erroneous_tokens):]
                output_file.write(f"{source}\t{target}\n")
                index += 1

    output_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    extract_cctc(args.input_file, args.output_file)
