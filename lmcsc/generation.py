import os
import torch
import torch.distributed as dist
from typing import Tuple, Union, List, Optional
from torch import nn

import warnings

from transformers import (
    BeamScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import (
    GenerateBeamOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput,
)

from lmcsc.common import HALF_MIN, MIN
from transformers import AutoModelForCausalLM
from lmcsc.obversation_generator import BaseObversationGenerator

def token_transformation_to_probs(self, observed_sequence: str) -> Tuple[List[int], List[float], dict]:
    """
    Transforms an observed sequence into token indices and their corresponding probabilities.

    Args:
        observed_sequence (str): The input sequence to be transformed.

    Returns:
        Tuple[List[int], List[float], dict]: A tuple containing:
            - List of token indices.
            - List of corresponding probabilities.
            - Dictionary of original token lengths.
    """
    # Get the token transformation and original token length for the observed sequence
    token_transformation, original_token_length = self.transformation_type.get_transformation_type(observed_sequence)
    cache = self.transformation_type_cache
    indices = list(token_transformation.keys())

    def get_weight(trans):
        # Retrieve the weight from cache if available, otherwise compute and cache it
        if trans in cache:
            return cache[trans]
        else:
            w = sum(self.distortion_probs[t] for t in trans)
            cache[trans] = w
            return w

    # Compute weights for each token transformation
    weight = [get_weight(trans) for trans in token_transformation.values()]

    return indices, weight, original_token_length


def get_distortion_probs(
    self, batch_observed_sequences: List[List[str]], eos_token_id: int
) -> Tuple[List[int], List[int], List[int], List[float], List[List[dict]], List[bool]]:
    """
    Computes distortion probabilities for a batch of observed sequences.

    Args:
        batch_observed_sequences (List[List[str]]): A batch of observed sequences.
        eos_token_id (int): The end-of-sequence token ID.

    Returns:
        Tuple[List[int], List[int], List[int], List[float], List[List[dict]], List[bool]]: A tuple containing:
            - List of batch indices.
            - List of beam indices.
            - List of token indices.
            - List of distortion probabilities.
            - List of original token lengths for each beam.
            - List of boolean values indicating if EOS is forced.
    """
    cache = self.cache
    batch_indices, beam_indices, token_indices, distortion_probs = [], [], [], []
    force_eos = []
    original_token_lengths = []

    for batch_index, observed_sequences in enumerate(batch_observed_sequences):
        beam_original_token_lengths = []
        for beam_index, observed_sequence in enumerate(observed_sequences):
            if observed_sequence in cache:
                indices, weight, original_token_length = cache[observed_sequence]
            else:
                if observed_sequence:
                    indices, weight, original_token_length = self.token_transformation_to_probs(observed_sequence)
                else:
                    indices = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
                    weight = [0.0] * len(indices)
                    original_token_length = {}
                cache[observed_sequence] = (indices, weight, original_token_length)
            force_eos.append(len(observed_sequence) == 0)
            batch_indices.extend([batch_index] * len(indices))
            beam_indices.extend([beam_index] * len(indices))
            token_indices.extend(indices)
            distortion_probs.extend(weight)
            beam_original_token_lengths.append(original_token_length)
        
        original_token_lengths.append(beam_original_token_lengths)

    return batch_indices, beam_indices, token_indices, distortion_probs, original_token_lengths, force_eos

@torch.jit.script
def distortion_probs_to_cuda_jit(
    template_tensor: torch.Tensor, 
    force_eos: torch.Tensor,
    batch_size: int, 
    num_beams: int, 
    batch_beam_size: int, 
    vocab_size: int, 
    _batch_indices: List[int], 
    _beam_indices: List[int], 
    _token_indices: List[int], 
    _distortion_probs: torch.Tensor) -> torch.Tensor:
    """
    Transfers distortion probabilities to a CUDA tensor.

    Args:
        template_tensor (torch.Tensor): The template tensor to be used.
        force_eos (torch.Tensor): Tensor indicating where to force end-of-sequence.
        batch_size (int): The size of the batch.
        num_beams (int): The number of beams.
        batch_beam_size (int): The size of the batch beam.
        vocab_size (int): The size of the vocabulary.
        _batch_indices (List[int]): List of batch indices.
        _beam_indices (List[int]): List of beam indices.
        _token_indices (List[int]): List of token indices.
        _distortion_probs (List[float]): List of distortion probabilities.

    Returns:
        torch.Tensor: The resulting tensor with distortion probabilities.
    """
    # Initialize distortion probabilities tensor and mask positions where EOS is forced
    if template_tensor.dtype == torch.float16:
        MIN = -1e4
    else:
        MIN = -1e32
    distortion_probs = template_tensor.masked_fill(force_eos[:, None], MIN).view(batch_size, num_beams, vocab_size)
    
    # Update distortion probabilities with the provided values
    distortion_probs[_batch_indices, _beam_indices, _token_indices] = _distortion_probs

    return distortion_probs.view(batch_beam_size, vocab_size)

def distortion_guided_beam_search(
    self,
    observed_sequence_generator: BaseObversationGenerator,
    beam_scorer: BeamScorer,
    input_ids: torch.LongTensor = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    A modified beam search function for CSC.

    Notes:
        This code is based on the `beam_search` function in the `transformers` library.
        We make 5 modifications to the original code:
            0. Initialization.
            1. Intervention of decoding process.
            2. Update the observed sequences.
            3. Remove stopping_criteria.
            4. Put the generated results into Streamer.
        You can search `## Modification X.*` in the code to find the corresponding part.
        
    Parameters:
        observed_sequence_generator (`BaseObversationGenerator`):
            An instance of [`BaseObversationGenerator`] that defines how observed sequences are generated.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Returns:
        [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """

    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    ## Modification 0:
    ## Initialization

    if input_ids is None:
        # In this case, we don't provide prompt or context, so we need to generate the first token
        input_ids = torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
        input_ids = input_ids * self.config.decoder_start_token_id

    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids, expand_size=num_beams, **model_kwargs
    )

    try:
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    except:
        pass

    vocab_size = self.vocab_size

    # template for the distortion model
    template_weight = self.probs_template * self.distortion_model_smoothing
    if template_weight.dtype == torch.float16:
        template_weight[self.token_length > 1] = HALF_MIN
    else:
        template_weight[self.token_length > 1] = MIN

    # clear the cache
    self.cache = {}
    self.cached_observed_sequences = []
    self.max_cached_observed_sequences = num_beams * batch_size

    ## END of modification

    batch_beam_size, cur_len = input_ids.shape
    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        ## Modification 1.0:
        observed_sequences = observed_sequence_generator.get_observed_sequences()
        _batch_indices, _beam_indices, _token_indices, _distortion_probs, all_original_token_lengths, force_eos = (
            self.get_distortion_probs(observed_sequences, eos_token_id)
        )
        ## END of modification

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        ## Modification 1.1:
        ## Intervention of decoding process
        # get the observed sequences and calculate the distortion probs
        force_eos = torch.tensor(force_eos, device=input_ids.device, dtype=torch.bool)

        distortion_probs = distortion_probs_to_cuda_jit(
            template_weight,
            force_eos,
            batch_size,
            num_beams,
            batch_beam_size,
            vocab_size,
            _batch_indices,
            _beam_indices,
            _token_indices,
            torch.tensor(
                _distortion_probs, device=template_weight.device, dtype=template_weight.dtype
            )
        )

        # calculate the length reward
        if self.alpha != 0:
            length_reward = self.alpha * (self.token_length[None] - 1).clamp(min=0.0)
        else:
            length_reward = 0.0

        # faithfulness reward
        faithfulness_coefficient = 1.0
        if self.use_faithfulness_reward:
            entropy = -torch.sum(
                next_token_scores * torch.exp(next_token_scores), dim=-1, keepdim=True
            )
            entropy = entropy / self.max_entropy
            faithfulness_coefficient = 1.0 + entropy

        # adjust the next token scores
        next_token_scores = next_token_scores + faithfulness_coefficient * (
            distortion_probs + length_reward
        )

        ## END of modification

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams * 5,
            dim=1,
            largest=True,
            sorted=True,
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # Remove redundant predictions
        _, n_candidate = next_tokens.shape
        for batch_i in range(batch_size):
            predicted_sequences_set = set()
            for candidate_i in range(n_candidate):
                i = next_indices[batch_i][candidate_i]
                t = next_tokens[batch_i][candidate_i]
                batch_beam_idx = batch_i * num_beams + i
                this_input_ids = input_ids[batch_beam_idx].tolist() + [t.item()]
                tokens = self.convert_ids_to_tokens(this_input_ids)
                if len(tokens) > 0:
                    if isinstance(tokens[0], bytes):
                        this_text = b"".join(tokens)
                    else:
                        this_text = "".join(tokens)
                else:
                    this_text = ""
                if this_text in predicted_sequences_set:
                    if next_token_scores.dtype == torch.float16:
                        next_token_scores[batch_i][candidate_i] = HALF_MIN
                    else:
                        next_token_scores[batch_i][candidate_i] = MIN
                else:
                    predicted_sequences_set.add(this_text)
                if len(predicted_sequences_set) > (max(2, 1 + n_eos_tokens) * num_beams):
                    break

        next_token_scores, candidate_index = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )
        next_tokens = next_tokens.gather(-1, candidate_index)
        next_indices = next_indices.gather(-1, candidate_index)

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        ## Modification 2:
        ## Update the observed sequences
        local_beam_idx = (beam_idx % num_beams).view(batch_size, num_beams)
        beam_next_tokens = beam_next_tokens.view(batch_size, num_beams)
        observed_sequence_generator.reorder(local_beam_idx)
        predicted_tokens = []
        original_token_lengths = []

        for batch_idx, (beams, token_ids) in enumerate(zip(local_beam_idx, beam_next_tokens)):
            _predicted_tokens = self.convert_ids_to_tokens(token_ids.tolist())
            _original_token_lengths = [
                all_original_token_lengths[batch_idx][beam].get(token_id, len(token))
                for beam, token, token_id in zip(beams, _predicted_tokens, token_ids.tolist())
            ]
            predicted_tokens.append(_predicted_tokens)
            original_token_lengths.append(_original_token_lengths)
        observed_sequence_generator.step(predicted_tokens, original_token_lengths)
        if streamer is not None:
            streamer.put((beam_scorer, input_ids.cpu()))
        # If you want to what's happening in the decoding process, you can uncomment the following line:
        # observed_sequence_generator.show_steps()
        # print()
        ## END of modification

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        # increase cur_len
        cur_len = cur_len + 1

        ## Modification 3:
        ## Remove stopping_criteria
        if beam_scorer.is_done:
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
        ## END of modification

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    ## Modification 4:
    if streamer is not None:
        streamer.put((beam_scorer, input_ids.cpu()))
        streamer.end()
    ## END of modification

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]

def process_reward_beam_search(
    self,
    observed_sequence_generator: BaseObversationGenerator,
    prompted_model: AutoModelForCausalLM,
    beam_scorer: BeamScorer,
    input_ids: torch.LongTensor = None,
    prompted_input_ids: torch.LongTensor = None,
    prompted_model_kwargs: dict = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    A modified beam search function for CSC.

    Notes:
        This code is based on the `beam_search` function in the `transformers` library.
        We make 5 modifications to the original code:
            0. Initialization.
            1. Intervention of decoding process.
            2. Update the observed sequences.
            3. Remove stopping_criteria.
            4. Put the generated results into Streamer.
        You can search `## Modification X.*` in the code to find the corresponding part.
        
    Parameters:
        observed_sequence_generator (`BaseObversationGenerator`):
            An instance of [`BaseObversationGenerator`] that defines how observed sequences are generated.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Returns:
        [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """

    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    ## Modification 0:
    ## Initialization

    if input_ids is None:
        # In this case, we don't provide prompt or context, so we need to generate the first token
        input_ids = torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
        input_ids = input_ids * self.config.decoder_start_token_id

    if prompted_model is not None:
        prompted_model_kwargs["attention_mask"] = model_kwargs.pop("prompted_attention_mask")

    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids, expand_size=num_beams, **model_kwargs
    )

    try:
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    except:
        pass

    if prompted_model is not None:
        assert prompted_input_ids is not None, "prompted_input_ids is required for realigned beam search"

        prompted_input_ids, prompted_model_kwargs = self._expand_inputs_for_generation(
            input_ids=prompted_input_ids, expand_size=num_beams, **prompted_model_kwargs
        )

        try:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
            prompted_model_kwargs = self._get_initial_cache_position(prompted_input_ids, prompted_model_kwargs)
        except:
            pass

    vocab_size = self.vocab_size

    # template for the distortion model
    if "MIS" in self.distortion_probs:
        alway_replace = False
        # if "MIS" is not in the distortion_probs, mark unrelevant tokens to be inserted
        template_weight = self.probs_template * self.token_length * self.distortion_probs["MIS"]
        template_weight[self.token_length < 1] = self.distortion_model_smoothing
        if template_weight.dtype == torch.float16:
            template_weight[self.token_length < 0] = HALF_MIN
            template_weight[self.is_chinese_token == False] = HALF_MIN
        else:
            template_weight[self.token_length < 0] = MIN
            template_weight[self.is_chinese_token == False] = MIN
    else:
        alway_replace = True
        template_weight = self.probs_template * self.distortion_model_smoothing
        if template_weight.dtype == torch.float16:
            template_weight[self.token_length > 1] = HALF_MIN
        else:
            template_weight[self.token_length > 1] = MIN

    # clear the cache
    self.cache = {}
    self.cached_observed_sequences = []
    self.max_cached_observed_sequences = num_beams * batch_size

    ## END of modification

    batch_beam_size, cur_len = input_ids.shape
    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if prompted_model is not None:
            prompted_model_inputs = self.prepare_inputs_for_generation(prompted_input_ids, **prompted_model_kwargs)
            prompted_outputs = prompted_model(
                **prompted_model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        ## Modification 1.0:
        observed_sequences = observed_sequence_generator.get_observed_sequences()
        # pdb.set_trace()
        _batch_indices, _beam_indices, _token_indices, _distortion_probs, all_original_token_lengths, force_eos = (
            self.get_distortion_probs(observed_sequences, eos_token_id)
        )
        related_token_indices = set(zip(_batch_indices, _beam_indices, _token_indices))
        ## END of modification

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        if prompted_model is not None:
            prompted_next_token_logits = prompted_outputs.logits[:, -1, :] / self.temperature
            prompted_next_token_scores = nn.functional.log_softmax(
                prompted_next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

        ## Modification 1.1:
        ## Intervention of decoding process
        # get the observed sequences and calculate the distortion probs
        force_eos = torch.tensor(force_eos, device=input_ids.device, dtype=torch.bool)

        distortion_probs = distortion_probs_to_cuda_jit(
            template_weight,
            force_eos,
            batch_size,
            num_beams,
            batch_beam_size,
            vocab_size,
            _batch_indices,
            _beam_indices,
            _token_indices,
            torch.tensor(
                _distortion_probs, device=template_weight.device, dtype=template_weight.dtype
            )
        )

        # calculate the length reward
        if self.alpha != 0:
            length_reward = self.alpha * (self.token_length[None] - 1).clamp(min=0.0)
        else:
            length_reward = 0.0

        # faithfulness reward
        faithfulness_coefficient = 1.0
        if self.use_faithfulness_reward:
            entropy = -torch.sum(
                next_token_scores * torch.exp(next_token_scores), dim=-1, keepdim=True
            )
            entropy = entropy / self.max_entropy
            faithfulness_coefficient = 1.0 + entropy

        # adjust the next token scores
        # if os.getenv("NO_PROMPT_LLM", "false").lower() == "true":
        if prompted_model is None:
            prompted_next_token_scores = 0.0

        if os.getenv("NO_PURE_LM", "false").lower() == "true":
            next_token_scores = 0.0
            faithfulness_coefficient = 1.0

        reward = (
            next_token_scores + faithfulness_coefficient * (
                distortion_probs + length_reward
            )
        )

        next_token_scores = prompted_next_token_scores + reward

        ## END of modification

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams * 5,
            dim=1,
            largest=True,
            sorted=True,
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # Remove redundant predictions
        _, n_candidate = next_tokens.shape
        for batch_i in range(batch_size):
            predicted_sequences_set = set()
            for candidate_i in range(n_candidate):
                i = next_indices[batch_i][candidate_i]
                t = next_tokens[batch_i][candidate_i]
                batch_beam_idx = batch_i * num_beams + i
                this_input_ids = input_ids[batch_beam_idx].tolist() + [t.item()]
                tokens = self.convert_ids_to_tokens(this_input_ids)
                if len(tokens) > 0:
                    if isinstance(tokens[0], bytes):
                        this_text = b"".join(tokens)
                    else:
                        this_text = "".join(tokens)
                else:
                    this_text = ""
                if this_text in predicted_sequences_set:
                    if next_token_scores.dtype == torch.float16:
                        next_token_scores[batch_i][candidate_i] = HALF_MIN
                    else:
                        next_token_scores[batch_i][candidate_i] = MIN
                else:
                    predicted_sequences_set.add(this_text)
                if len(predicted_sequences_set) > (max(2, 1 + n_eos_tokens) * num_beams):
                    break

        next_token_scores, candidate_index = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )
        next_tokens = next_tokens.gather(-1, candidate_index)
        next_indices = next_indices.gather(-1, candidate_index)

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        if prompted_model is not None:
            prompted_input_ids = torch.cat(
                [prompted_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

        ## Modification 2:
        ## Update the observed sequences
        local_beam_idx = (beam_idx % num_beams).view(batch_size, num_beams)
        beam_next_tokens = beam_next_tokens.view(batch_size, num_beams)
        observed_sequence_generator.reorder(local_beam_idx)
        predicted_tokens = []
        original_token_lengths = []

        for batch_idx, (beam_ids, token_ids) in enumerate(zip(local_beam_idx, beam_next_tokens)):
            _predicted_tokens = self.convert_ids_to_tokens(token_ids.tolist())
            if alway_replace:
                _original_token_lengths = [
                    all_original_token_lengths[batch_idx][beam_id].get(token_id, len(token))
                    for beam_id, token, token_id in zip(beam_ids.tolist(), _predicted_tokens, token_ids.tolist())
                ]
            else:
                _original_token_lengths = [
                    all_original_token_lengths[batch_idx][beam_id].get(token_id, len(token) if (batch_idx, beam_id, token_id) in related_token_indices else 0)
                    for beam_id, token, token_id in zip(beam_ids.tolist(), _predicted_tokens, token_ids.tolist())
                ]
            predicted_tokens.append(_predicted_tokens)
            original_token_lengths.append(_original_token_lengths)
        observed_sequence_generator.step(predicted_tokens, original_token_lengths)
        if streamer is not None:
            streamer.put((beam_scorer, input_ids.cpu()))
        # If you want to what's happening in the decoding process, you can uncomment the following line:
        # observed_sequence_generator.show_steps()
        # print()
        ## END of modification

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if prompted_model is not None:
            prompted_model_kwargs = self._update_model_kwargs_for_generation(
                prompted_outputs, prompted_model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if prompted_model_kwargs["past_key_values"] is not None:
                prompted_model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    prompted_model_kwargs["past_key_values"], beam_idx
                )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        # increase cur_len
        cur_len = cur_len + 1

        ## Modification 3:
        ## Remove stopping_criteria
        if beam_scorer.is_done:
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
        ## END of modification

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    ## Modification 4:
    if streamer is not None:
        streamer.put((beam_scorer, input_ids.cpu()))
        streamer.end()
    ## END of modification

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]
