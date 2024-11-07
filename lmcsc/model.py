from typing import List
from functools import partial

import torch
import importlib
import math

from lmcsc.generation import (
    token_transformation_to_probs,
    get_distortion_probs,
    distortion_guided_beam_search,
)
from lmcsc.obversation_generator import NextObversationGenerator
from lmcsc.transformation_type import TransformationType
from lmcsc.common import MIN
from lmcsc.utils import (
    get_vocab_decoder,
    qwen1_5_convert_ids_to_tokens,
    try_download_model_from_ms,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BeamSearchScorer, DynamicCache


class LMCSCModel:
    def __init__(
        self,
        model_name: str,
        n_observed_chars: int = 8,
        n_beam: int = 8,
        n_beam_hyps_to_keep: int = 1,
        alpha: float = 2.5,  # hyperparameter for the length reward
        shape_similar_threshold: float = 0.45,
        distortion_model_smoothing: float = -15.0,
        use_faithfulness_reward: bool = True,
    ):
        self.n_beam = n_beam
        self.n_beam_hyps_to_keep = n_beam_hyps_to_keep

        self.model_name = model_name
        self.n_observed_chars = n_observed_chars

        try_download_model_from_ms(self.model_name)

        if (
            "qwen" in self.model_name.lower()
            and importlib.util.find_spec("flash_attn") is not None
        ):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.vocab = self.tokenizer.get_vocab()
        self.alpha = alpha
        self.shape_similar_threshold = shape_similar_threshold
        self.distortion_model_smoothing = distortion_model_smoothing

        self.use_faithfulness_reward = use_faithfulness_reward

        self.decorate_model_instance()

    def decorate_model_instance(self) -> None:
        self.model.n_observed_chars = self.n_observed_chars

        self.is_byte_level_tokenize = isinstance(list(self.vocab.keys())[0], bytes)
        if "qwen" in self.model_name.lower():
            self.is_byte_level_tokenize = True
        elif "llama" in self.model_name.lower():
            self.is_byte_level_tokenize = True
        self.model.is_byte_level_tokenize = self.is_byte_level_tokenize

        if "qwen" in self.model_name.lower():
            self.model.config.decoder_start_token_id = self.tokenizer.encode("\n")[0]
        elif "llama" in self.model_name.lower():
            self.model.config.decoder_start_token_id = self.tokenizer.encode("\n")[-1]
        else:
            if "uer" in self.model_name:
                stop_token = "[CLS]"
            else:
                stop_token = b"\n" if self.is_byte_level_tokenize else "\n"
            self.model.config.decoder_start_token_id = self.vocab[stop_token]

        # Use vocab to get vocab_size is unreliable, need access to the model's output layer
        if "Baichuan2" in self.model_name:
            self.model.vocab_size = self.model.lm_head.weight.shape[0]
        elif "internlm2" in self.model_name:
            self.model.vocab_size = self.model.output.out_features
        else:
            self.model.vocab_size = self.model.lm_head.out_features

        # Qwen1.5 use titoken to tokenize the input
        if "qwen" in self.model_name.lower():
            self.vocab, bytes_decoder = get_vocab_decoder(self.vocab)
            self.model.convert_ids_to_tokens = partial(
                qwen1_5_convert_ids_to_tokens, decoder=bytes_decoder
            )
        elif "llama" in self.model_name.lower():
            self.vocab, bytes_decoder = get_vocab_decoder(self.vocab)
            self.model.convert_ids_to_tokens = partial(
                qwen1_5_convert_ids_to_tokens, decoder=bytes_decoder
            )
        else:
            self.model.convert_ids_to_tokens = self.tokenizer.convert_ids_to_tokens

        self.tokenizer.padding_side = "left"
        self.model.probs_template = torch.ones((self.model.vocab_size,)).to(
            self.model.device
        )
        self.model.transformation_type = TransformationType(
            self.vocab,
            is_bytes_level=self.is_byte_level_tokenize,
            shape_similar_threshold=self.shape_similar_threshold,
        )
        self.model.token_length = (
            torch.ones((self.model.vocab_size,)).to(self.model.device) * MIN
        )
        for idx, l in self.model.transformation_type.token_length.items():
            self.model.token_length[idx] = l
        self.model.transformation_type_cache = {}

        self.model.alpha = self.alpha

        self.model.distortion_probs = {
            "IDT": -0.04,
            "SAP": -3.75,
            "SIP": -4.85,
            "SIS": -5.40,
            "OTH": -8.91,
            "UNR": -14.99,
        }


        self.model.distortion_model_smoothing = self.distortion_model_smoothing
        self.model.use_faithfulness_reward = self.use_faithfulness_reward
        self.model.max_entropy = (
            torch.tensor(self.model.vocab_size).float().log().to(self.model.device)
        )

        # Logs
        self.print_params()

        # add function to the model instance
        self.model.token_transformation_to_probs = (
            token_transformation_to_probs.__get__(self.model, type(self.model))
        )
        self.model.get_distortion_probs = get_distortion_probs.__get__(
            self.model, type(self.model)
        )
        self.model.distortion_guided_beam_search = distortion_guided_beam_search.__get__(
            self.model, type(self.model)
        )

    def get_n_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        all_param = 0
        if self.model is None:
            return "N/A"
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params

        # convert to human readable format (i.e. 178B instead of 178000000000)
        def human_format(num):
            num = int(num)
            if num == 0:
                return "0"
            units = [
                "", "K", "M", "B", "T", "P", "E", "Z", "Y", "B", "C", "D", "N",
                "U"
            ]
            p = int(math.floor(math.log(num) / math.log(1000)))
            s = round(num / math.pow(1000, p), 2)
            return "%s%s" % (s, units[p])

        return human_format(all_param)

    def print_params(self):
        print(f"Model: {self.model_name}")
        print(f"Number of parameters: {self.get_n_parameters()}")
        print(f"Beam size: {self.n_beam}")
        print(f"Alpha: {self.alpha}")
        print(f"Use faithfulness reward: {self.use_faithfulness_reward}")
        print(f"Distortion model probs: {self.model.distortion_probs}")
        print(f"Max entropy: {self.model.max_entropy}")
        print(f"Shape similar threshold: {self.shape_similar_threshold}")
        print(f"Distortion model smoothing: {self.distortion_model_smoothing}")
        print(f"N observed chars: {self.n_observed_chars}")

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            if hasattr(self, key):
                setattr(self, key, value)

    def preprocess(
        self, src: List[str], contexts: List[str] = None, prompt_split: str = "\n"
    ):
        if "qwen" in self.model_name:
            eos_token_id = self.tokenizer.encode("<|endoftext|>")[0]
            pad_token_id = eos_token_id
        elif "llama" in self.model_name:
            eos_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
        elif "uer" in self.model_name:
            eos_token_id = 102
            pad_token_id = 0
        else:
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id

        model_kwargs = {
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "is_encoder_decoder": False,
        }

        if "Baichuan2" not in self.model_name:
            model_kwargs["past_key_values"] = DynamicCache()

        if contexts is None:
            contexts = [prompt_split for _ in src]
        else:
            contexts = [context + prompt_split for context in contexts]

        context_infos = self.tokenizer(contexts, return_tensors="pt", padding=True)
        context_infos.to(self.model.device)
        context_input_ids = context_infos["input_ids"]
        context_attention_mask = context_infos["attention_mask"]

        assert self.n_beam > 1, "Beam size must be greater than 1"

        beam_scorer = BeamSearchScorer(
            batch_size=len(src),
            num_beams=self.n_beam,
            num_beam_hyps_to_keep=self.n_beam_hyps_to_keep,
            max_length=self.model.config.max_length,
            device=self.model.device,
            length_penalty=0.0,
        )

        # Build the observed sequence generator
        observed_sequence_generator = NextObversationGenerator(
            src,
            self.n_beam,
            self.n_observed_chars,
            self.is_byte_level_tokenize,
            verbose=False,
        )

        return (
            model_kwargs,
            context_input_ids,
            context_attention_mask,
            beam_scorer,
            observed_sequence_generator,
        )

    def postprocess(self, outputs):
        preds = [
            pred.strip().split("\n")[-1]
            for pred in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        if "uer" in self.model_name:
            preds = ["".join(pred.split()) for pred in preds]

        if self.n_beam_hyps_to_keep > 1:
            preds = [
                preds[i : i + self.n_beam_hyps_to_keep]
                for i in range(0, len(preds), self.n_beam_hyps_to_keep)
            ]
        return preds

    def __call__(self, src: List[str], contexts: List[str] = None):
        (
            model_kwargs,
            context_input_ids,
            context_attention_mask,
            beam_scorer,
            observed_sequence_generator,
        ) = self.preprocess(src, contexts)

        with torch.no_grad():
            outputs = self.model.distortion_guided_beam_search(
                observed_sequence_generator,
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                beam_scorer=beam_scorer,
                **model_kwargs,
            )

        preds = self.postprocess(outputs)

        return preds
