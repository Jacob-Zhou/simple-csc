## AISHELL-1 datasets

## path: "datasets/aishell1"
#    - "aishell1/test.txt"
## format:
# {
#     "utts": {
#         "BAC009S0764W0121": {
#             "output": [
#                 {
#                     "name": "target1[1]",
#                     "rec_text": "甚至出现交易几乎停滞的情况<eos>",
#                     "rec_token": "甚 至 出 现 交 易 几 乎 停 滞 的 情 况 <eos>",
#                     "rec_tokenid": "2474 3116 331 2408 82 1684 321 47 235 2199 2553 1319 307 4232",
#                     "score": -7.902437210083008,
#                     "shape": [
#                         13,
#                         4233
#                     ],
#                     "text": "甚至出现交易几乎停滞的情况",
#                     "token": "甚 至 出 现 交 易 几 乎 停 滞 的 情 况",
#                     "tokenid": "2474 3116 331 2408 82 1684 321 47 235 2199 2553 1319 307"
#                 }
#             ],
#            "utt2spk": "S0764"
#        },
#    }
#    ...
#}

import json
import argparse
import os

def extract_aishell1(input_file, output_file, output_with_contents):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    previous_srcs = []

    output_file = open(output_file, 'w', encoding='utf-8')
    output_with_contents = open(output_with_contents, 'w', encoding='utf-8')

    for utt in data['utts']:
        item = data['utts'][utt]
        output = item['output'][0]
        src = output['rec_text']
        tgt = output['text']
        assert src.endswith('<eos>')
        src = src[:-5]
        output_file.write(f"{src}\t{tgt}\n")
        this_previous = " ".join(previous_srcs[-3:])
        output_with_contents.write(f"{this_previous}\t{src}\t{tgt}\n")

        previous_srcs.append(src)

    output_file.close()
    output_with_contents.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="datasets/aishell1/test.json")
    parser.add_argument("--output_dir", type=str, default="datasets/aishell1")
    args = parser.parse_args()

    output_file = os.path.join(args.output_dir, "test.txt")
    output_with_contents = os.path.join(args.output_dir, "test_with_contents.txt")
    extract_aishell1(args.input_file, output_file, output_with_contents)
