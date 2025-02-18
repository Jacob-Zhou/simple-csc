import argparse
from tqdm import tqdm

import torch
from lmcsc.common import HALF_MIN, MIN
from lmcsc.obversation_generator import NextObversationGenerator
from lmcsc.generation import distortion_probs_to_cuda
from lmcsc.corrector import LMCorrector


def get_score(src, pred, model, log_probs, enthopies, labels, vocab_size):
    lm_model = model.lm_model
    hf_model = model.model
    hf_model.cache = {}
    hf_model.cached_observed_sequences = []
    hf_model.max_cached_observed_sequences = 1

    model_kwargs = lm_model.get_model_kwargs()
    eos_token_id = model_kwargs["eos_token_id"]
    if len(pred) == 0:
        return MIN
    observed_sequence_generator = NextObversationGenerator(
        [src],
        1,
        8,
        is_bytes_level=model.is_byte_level_tokenize,
        verbose=False,
    )

    # template for the distortion model
    template_weight = hf_model.probs_template * hf_model.token_length * hf_model.distortion_probs["MIS"]
    template_weight[hf_model.token_length < 1] = hf_model.distortion_model_smoothing
    if template_weight.dtype == torch.float16:
        template_weight[hf_model.token_length < 0] = HALF_MIN
        template_weight[hf_model.is_chinese_token == False] = HALF_MIN
    else:
        template_weight[hf_model.token_length < 0] = MIN
        template_weight[hf_model.is_chinese_token == False] = MIN

    final_score = 0.0

    # simulate the decoding process
    for log_prob, entropy, label in zip(log_probs, enthopies, labels.tolist()):
        observed_sequences = observed_sequence_generator.get_observed_sequences()
        _batch_indices, _beam_indices, _token_indices, _distortion_probs, all_original_token_lengths, force_eos = (
            hf_model.get_distortion_probs(observed_sequences, eos_token_id)
        )
        related_token_indices = set(zip(_batch_indices, _beam_indices, _token_indices))
        force_eos = torch.tensor(force_eos, device=log_probs.device, dtype=torch.bool)

        distortion_probs = distortion_probs_to_cuda(
            template_weight,
            force_eos,
            1,
            1,
            1,
            vocab_size,
            _batch_indices,
            _beam_indices,
            _token_indices,
            torch.tensor(
                _distortion_probs, device=template_weight.device, dtype=template_weight.dtype
            )
        )
        selected_distortion_prob = distortion_probs[0, label]

        if hf_model.alpha != 0:
            length_reward = hf_model.alpha * (hf_model.token_length[None] - 1).clamp(min=0.0)
            selected_length_reward = length_reward[0, label]
        else:
            length_reward = 0.0
            selected_length_reward = 0.0

        faithfulness_coefficient = 1.0
        if hf_model.use_faithfulness_reward:
            entropy = entropy / hf_model.max_entropy
            faithfulness_coefficient = 1.0 + entropy

        final_score += log_prob + faithfulness_coefficient * (
            selected_distortion_prob + selected_length_reward
        )

        token = hf_model.convert_ids_to_tokens([label])[0]
        _original_token_lengths = [
            all_original_token_lengths[0][0].get(label, len(token) if (0, 0, label) in related_token_indices else 0)
        ]
        observed_sequence_generator.step([[token]], [_original_token_lengths])
        # observed_sequence_generator.show_steps()
        # print()

    return final_score.item()

def batch_get_lm_score(preds, model, bos_token_id):
    hf_model = model.model
    infos = model.tokenizer(preds, padding=True, padding_side="right", return_tensors="pt").to(hf_model.device)
    input_ids = infos["input_ids"]
    batch_size, _ = input_ids.size()
    input_ids = torch.cat([torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=hf_model.device), input_ids], dim=1)
    labels = input_ids[:, 1:]
    mask = infos["attention_mask"]
    input_len = mask.sum(dim=-1)

    with torch.no_grad():
        output = hf_model(input_ids)
    # token-wise probability
    prob_dist = torch.nn.functional.log_softmax(output.logits, dim=-1)[:, :-1]
    log_probs = prob_dist.gather(2, labels[..., None])

    vocab_size = output.logits.size(-1)

    # token-wise log prob and entropy
    log_probs = log_probs.view(batch_size, -1)
    enthopies = -(prob_dist.exp() * prob_dist).sum(dim=-1)
    log_probs = [log_probs[i, :input_len[i]] for i in range(batch_size)]
    enthopies = [enthopies[i, :input_len[i]] for i in range(batch_size)]
    labels = [labels[i, :input_len[i]] for i in range(batch_size)]

    return log_probs, enthopies, labels, vocab_size

def main(args):
    model = LMCorrector(args.model_name, config_path=args.config_path)
    if "qwen" in args.model_name.lower():
        bos_token_id = model.tokenizer.encode("\n")[0]
    elif "llama" in args.model_name.lower():
        bos_token_id = model.tokenizer.encode("\n")[-1]
    else:
        if "uer" in args.model_name:
            stop_token = "[CLS]"
        else:
            stop_token = "\n"
        bos_token_id = model.tokenizer.get_vocab()[stop_token]

    output_writer = open(args.output_file, "w")
    
    
    with open(args.input_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing lines"):
            src, *preds = line.strip().split("\t")
            scores = []
            log_probs, enthopies, labels, vocab_size = batch_get_lm_score(preds, model, bos_token_id)
            for pred, log_prob, entropy, label in tqdm(zip(preds, log_probs, enthopies, labels), desc="Scoring predictions", leave=False):
                score = get_score(src, pred, model, log_prob, entropy, label, vocab_size)
                scores.append(score)
            # sort by score, highest first
            scores, preds = zip(*sorted(zip(scores, preds), key=lambda x: x[0], reverse=True))
            output_writer.write("\t".join([src] + list(preds)) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="name of model to use")
    parser.add_argument("--config-path", help="path of config file")
    parser.add_argument("--input-file", help="path of input datasets")
    parser.add_argument("--output-file", help="path of output file")
    args = parser.parse_args()
    main(args)
