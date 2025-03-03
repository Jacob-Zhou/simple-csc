import argparse
import time
from typing import List

import os
import torch
from lmcsc import LMCorrector


from lmcsc.obversation_generator import NextObversationGenerator
from lmcsc.utils import clean_sentences, measure_cuda_memory

def run_model(model, 
              src: List[str], 
              contexts: List[str] = None,
              prompt_split: str = "\n"):
    # (
    #     model_kwargs,
    #     context_input_ids,
    #     context_attention_mask,
    #     beam_scorer,
    #     observed_sequence_generator,
    # ) = model.preprocess(src, contexts)

    # start = time.time()
    # with torch.no_grad():
    #     outputs = model.model.distortion_guided_beam_search(
    #         observed_sequence_generator,
    #         input_ids=context_input_ids,
    #         attention_mask=context_attention_mask,
    #         beam_scorer=beam_scorer,
    #         **model_kwargs,
    #     )
    # end = time.time()

    # return outputs, end - start

    n_beam = model.n_beam
    n_beam_hyps_to_keep = model.n_beam_hyps_to_keep

    if isinstance(src, str):
        src = [src]
    if contexts is not None:
        if isinstance(contexts, str):
            contexts = [contexts]
        assert len(src) == len(contexts), f"src and contexts must have the same length, got {len(src)} and {len(contexts)}"

    # Preprocess the source texts
    processed_src, changes = model.preprocess(src, contexts)

    # Prepare inputs for beam search generation
    (
        model_kwargs,
        context_input_ids,
        context_attention_mask,
        beam_scorer,
    ) = model.lm_model.prepare_beam_search_inputs(processed_src, contexts, prompt_split, n_beam, n_beam_hyps_to_keep)


    # Initialize the observed sequence generator
    observed_sequence_generator = NextObversationGenerator(
        processed_src,
        model.n_beam,
        model.n_observed_chars,
        model.is_byte_level_tokenize,
        verbose=False,
    )

    # Run the beam search generation
    with torch.no_grad():
        start = time.time()
        outputs = model.model.distortion_guided_beam_search(
            observed_sequence_generator,
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            beam_scorer=beam_scorer,
            **model_kwargs,
        )
        end = time.time()

    return outputs, end - start

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config-path", type=str, default="configs/default_config.yaml")
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
        "--decode-prefix",
        type=str,
        default="",
        help="Prefix to add to the input sentence",
    )
    parser.add_argument(
        "--prefix-split",
        type=str,
        default="\n",
        help="Separator used between prefixes in a batch",
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

    print(f"Deocode Prefix: {repr(args.decode_prefix)}")
    print(f"Prefix Split:   {repr(args.prefix_split)}")
    args.output_file = f"{args.path}/speed_test.log"
    os.makedirs(args.path, exist_ok=True)

    short_sentences = [] # < 32
    mid_sentences = [] # 32 < len < 64
    long_sentences = [] # > 64
    max_sent = 50
    for input_file in args.input_file.split(","):
        for line in open(input_file, "r"):
            source, *_ = line.split("\t")
            if len(source.strip()) < 32:
                if len(short_sentences) < max_sent:
                    short_sentences.append(source.strip())
            elif len(source.strip()) < 64:
                if len(mid_sentences) < max_sent:
                    mid_sentences.append(source.strip())
            else:
                if len(long_sentences) < max_sent:
                    long_sentences.append(source.strip())

    # baichuan-inc/Baichuan2-7B-Base, Baichuan2 is the model_family
    model_family = args.model_name.split("/")[-1].split("-")[0]
    with measure_cuda_memory() as mem:
        lm_corrector = LMCorrector(
            args.model_name,
            config_path=args.config_path,
            n_beam=args.n_beam,
            n_beam_hyps_to_keep=args.n_beam_hyps_to_keep,
            max_length=args.max_length,
            alpha=args.alpha,
            n_observed_chars=args.n_observed_chars,
            shape_similar_threshold=args.shape_similar_threshold,
            distortion_model_smoothing=args.distortion_model_smoothing,
            use_faithfulness_reward=args.use_faithfulness_reward,
        )
    mem = mem.final_memory
    print(f"{mem / float(2**20):.2f} MB")

    hypos = []

    i = 0
    batch = []
    cur_batch_size = 0
    batch_size = 800

    print("Start testing")
    sentences = {
        "short": short_sentences,
        "mid": mid_sentences,
        "long": long_sentences
    }
    for key, sub_sentences in sentences.items():
        runtime_list = []
        mem_list = []
        for source in sub_sentences:
            if "uer" in args.model_name:
                batch = ["".join(source.split())]
            else:
                batch = [source]
            batch, changes = clean_sentences(batch)
            with measure_cuda_memory() as mem:
                output, time_spend = run_model(lm_corrector, batch)
            mem = mem.final_memory
            print((len(batch[0]), time_spend, f"{mem / float(2**20):.2f} MB"), key)
            runtime_list.append((len(batch[0]), time_spend))
            mem_list.append(mem)
        
        print(key)
        print(f"Average time (ms): {sum([t for _, t in runtime_list]) / len(runtime_list) * 1000}")
        print(f"Avg. time per char (ms): {sum([t for _, t in runtime_list]) / sum([c for c, _ in runtime_list]) * 1000}")
        print(f"Average memory (MB): {sum(mem_list) / len(mem_list) / float(2**20):.2f}")
        print("-" * 100)