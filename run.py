import argparse
import pickle
import time
import datetime

import os

from lmcsc.model import LMCSCModel
from lmcsc.common import OOV_CHAR
from lmcsc.utils import clean_sentences, rebuild_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    # decoding parameters
    parser.add_argument(
        "--batch-size", type=int, default=200, help="Number of characters in each batch"
    )
    parser.add_argument(
        "--max-sentences-per-batch", type=int, default=128, help="Number of sentences in each batch"
    )
    parser.add_argument(
        "--n-beam", type=int, default=8, help="Number of beams in beam search"
    )
    parser.add_argument(
        "--n-beam-hyps-to-keep",
        type=int,
        default=1,
        help="Number of beams to keep in beam search",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum length of the corrected sentence",
    )
    parser.add_argument(
        "--deocode-prefix",
        type=str,
        default="",
        help="Prefix to add to the input sentence",
    )
    # noise distortion model parameters
    parser.add_argument(
        "--n-observed-chars",
        type=int,
        default=8,
        help="How many next characters to observe",
    )
    parser.add_argument(
        "--shape-similar-threshold",
        type=float,
        default=0.45,
        help="Threshold for shape similarity",
    )
    parser.add_argument(
        "--distortion-model-smoothing",
        type=float,
        default=-15.0,
        help="Smoothing for distortion model",
    )
    parser.add_argument(
        "--alpha", type=float, default=2.5, help="Hyperparameter for the length reward"
    )
    parser.add_argument(
        "--use-faithfulness-reward",
        action="store_true",
        help="Whether to use faithfulness reward",
    )

    args = parser.parse_args()

    dataset = "_".join(args.input_file.split("/")[1:])
    dataset = ".".join(dataset.split(".")[:-1])
    print(f"Dataset: {dataset}")
    args.output_file = f"{args.path}/prediction.txt"
    os.makedirs(args.path, exist_ok=True)

    sources = []
    for line in open(args.input_file, "r"):
        source, _ = line.split("\t")
        sources.append(source.strip())

    # baichuan-inc/Baichuan2-7B-Base, Baichuan2 is the model_family
    model_family = args.model_name.split("/")[-1].split("-")[0]
    lmcsc_model = LMCSCModel(
        args.model_name,
        n_beam=args.n_beam,
        n_beam_hyps_to_keep=args.n_beam_hyps_to_keep,
        alpha=args.alpha,
        n_observed_chars=args.n_observed_chars,
        shape_similar_threshold=args.shape_similar_threshold,
        distortion_model_smoothing=args.distortion_model_smoothing,
        use_faithfulness_reward=args.use_faithfulness_reward,
    )

    # reorder sources by length, from longest to shortest
    src_index, reordered_sources = zip(
        *sorted(enumerate(sources), key=lambda x: len(x[1]), reverse=True)
    )
    reorder_index, _ = zip(*sorted(enumerate(src_index), key=lambda x: x[1]))

    hypos = []

    batch_size = args.batch_size
    i = 0
    batch = []
    ori_batch = []
    cur_batch_size = 0
    start = time.time()
    while i < len(sources):
        batch_start = time.time()
        # Build batch
        while True:
            if len(batch) == 0 or (
                (cur_batch_size + len(reordered_sources[i])) < batch_size
            ):
                ori_batch.append(reordered_sources[i])
                if "uer" in args.model_name:
                    batch.append("".join(reordered_sources[i][: args.max_length].split()))
                else:
                    batch.append(reordered_sources[i][: args.max_length])
                cur_batch_size += min(len(reordered_sources[i]), args.max_length)
                i += 1
                if i >= len(reordered_sources):
                    break
                if len(batch) > args.max_sentences_per_batch:
                    break
            else:
                break
        print(ori_batch[0])
        batch, changes = clean_sentences(batch)
        output = lmcsc_model(batch, [args.deocode_prefix] * len(batch))
        output = rebuild_sentences(output, changes)
        output = [
            (o + s[len(o) :]) if (len(o) < len(s)) else o
            for o, s in zip(output, ori_batch)
        ]

        # Handle oov characters
        output = [
            "".join(
                [
                    o_char if o_char != OOV_CHAR else s_char
                    for o_char, s_char in zip(o, s)
                ]
            )
            for o, s in zip(output, ori_batch)
        ]

        print(output[0])
        hypos.extend(output)
        speed = len(batch) / (time.time() - batch_start)
        print(
            f"Processed: {i}, Speed: {speed:.2f} sentences/sec, Time to go: {datetime.timedelta(seconds=(len(sources) - i) / speed)}"
        )
        print()

        # reset batch
        ori_batch = []
        batch = []
        cur_batch_size = 0

    hypos = [hypos[i].strip().replace("\n", " ") for i in reorder_index]
    output_file = open(args.output_file, "w", encoding="utf-8")
    output_file.write("\n".join(hypos))
    output_file.close()
    print(f"Total time: {datetime.timedelta(seconds=(time.time() - start))}")
