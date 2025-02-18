from functools import partial
import importlib
import math
import os
from typing import List
import torch
from lmcsc.utils import (
    get_vocab_decoder,
    qwen1_5_convert_ids_to_tokens,
    try_download_model_from_ms,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BeamSearchScorer, DynamicCache

class LMModel:
    """
    A base class for language models.

    Args:
        model (str): The name or path of the pre-trained model.
        attn_implementation (str, optional): The attention implementation to use. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        model_name (str): The name of the model.
        model (AutoModelForCausalLM): The loaded language model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        vocab (dict): The vocabulary of the model.
        is_byte_level_tokenize (bool): Whether the tokenization is byte-level.

    """

    def __init__(
        self,
        model: str,
        attn_implementation: str = None,
        *args,
        **kwargs
    ):
        self.model_name = model
        try_download_model_from_ms(self.model_name)
        device_map = kwargs.pop("device_map", "auto")
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        attn_implementation = kwargs.pop("attn_implementation", attn_implementation)
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code
        )
        self.model.eval()

        self.vocab = self.tokenizer.get_vocab()
        self.is_byte_level_tokenize = isinstance(list(self.vocab.keys())[0], bytes)

        self.decorate_model_instance()

    def set_decoder_start_token_id(self):
        """
        Sets the decoder start token ID.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def set_vocab_size(self):
        """
        Sets the vocabulary size.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def set_convert_ids_to_tokens(self):
        """
        Sets the convert_ids_to_tokens function.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def decorate_model_instance(self):
        """
        Decorates the model instance with additional attributes and settings.
        """
        self.set_decoder_start_token_id()
        self.set_vocab_size()
        self.set_convert_ids_to_tokens()

        self.tokenizer.padding_side = "left"
        self.model.probs_template = torch.ones((self.model.vocab_size,), dtype=self.model.dtype).to(
            self.model.device
        )

    def get_model_kwargs(self):
        """
        Gets the model-specific keyword arguments.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def prepare_beam_search_inputs(
        self, src: List[str], contexts: List[str] = None, prompt_split: str = "\n", n_beam: int = 8, n_beam_hyps_to_keep: int = 1
    ):
        """
        Prepares inputs for beam search.

        Args:
            src (List[str]): The source sentences.
            contexts (List[str], optional): The context for each source sentence. Defaults to None.
            prompt_split (str, optional): The prompt split token. Defaults to "\\n".
            n_beam (int, optional): The number of beams. Defaults to 8.
            n_beam_hyps_to_keep (int, optional): The number of beam hypotheses to keep. Defaults to 1.

        Returns:
            tuple: A tuple containing model_kwargs, context_input_ids, context_attention_mask, and beam_scorer.
        """
        model_kwargs = self.get_model_kwargs()

        if contexts is None:
            contexts = [prompt_split for _ in src]
        else:
            contexts = [context + prompt_split for context in contexts]

        context_infos = self.tokenizer(contexts, return_tensors="pt", padding=True)
        context_infos.to(self.model.device)
        context_input_ids = context_infos["input_ids"]
        context_attention_mask = context_infos["attention_mask"]

        assert n_beam > 1, "Beam size must be greater than 1"

        beam_scorer = BeamSearchScorer(
            batch_size=len(src),
            num_beams=n_beam,
            num_beam_hyps_to_keep=n_beam_hyps_to_keep,
            max_length=self.model.config.max_length,
            device=self.model.device,
            length_penalty=0.0,
        )

        return (
            model_kwargs,
            context_input_ids,
            context_attention_mask,
            beam_scorer
        )

    def prepare_prompted_inputs(self, src: List[str]):
        """
        Prepares inputs for beam search.

        Args:
            src (List[str]): The source sentences.
            contexts (List[str], optional): The context for each source sentence. Defaults to None.
            prompt_split (str, optional): The prompt split token. Defaults to "\\n".
            n_beam (int, optional): The number of beams. Defaults to 8.
            n_beam_hyps_to_keep (int, optional): The number of beam hypotheses to keep. Defaults to 1.

        Returns:
            tuple: A tuple containing model_kwargs, context_input_ids, context_attention_mask, and beam_scorer.
        """
        model_kwargs = self.get_model_kwargs()
        if os.getenv("DETAILED_PROMPT", "false").lower() == "true":
            prompted_contexts = [f"你是一个优秀的中文纠错模型，中文纠错模型即更正用户输入句子中的错误。你需要识别并纠正用户输入的句子中可能的错别字、多字、漏字并输出正确的句子，在修改的同时尽可能减少对原句子的改动(不新增、删除和修改标点符号)。只输出没有错误的句子，不要添加任何其他解释或说明。如果句子没有错误，就直接输出和输入相同的句子。\n输入：{s}\n输出：" for s in src]
        else:
            prompted_contexts = [f"纠正下面句子中的错别字以及多字少字错误，并给出修改后的句子。\n输入：{s}\n输出：" for s in src]
        prompted_context_infos = self.tokenizer(prompted_contexts, return_tensors="pt", padding=True)
        prompted_context_infos.to(self.model.device)
        prompted_context_input_ids = prompted_context_infos["input_ids"]
        prompted_context_attention_mask = prompted_context_infos["attention_mask"]

        return (
            model_kwargs,
            prompted_context_input_ids,
            prompted_context_attention_mask,
        )

    def process_generated_outputs(self, outputs, contexts: List[str] = None, prompt_split: str = "\n", n_beam_hyps_to_keep: int = 1, need_decode: bool = True):
        """
        Processes the generated outputs.

        Args:
            outputs: The generated outputs.
            contexts (List[str], optional): The context for each output. Defaults to None.
            prompt_split (str, optional): The prompt split token. Defaults to "\\n".
            n_beam_hyps_to_keep (int, optional): The number of beam hypotheses to keep. Defaults to 1.
            need_decode (bool, optional): Whether to decode the outputs. Defaults to True.

        Returns:
            List[List[str]]: The processed predictions.
        """
        if need_decode:
            preds = [
                pred
                for pred in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ]
        else:
            preds = outputs

        if contexts is None:
            contexts = [prompt_split for _ in preds]
        else:
            contexts = [context + prompt_split for context in contexts]

        preds = [
            preds[i : i + n_beam_hyps_to_keep]
            for i in range(0, len(preds), n_beam_hyps_to_keep)
        ]

        preds = [
            [
                pred[len(context) :] if pred.startswith(context) else pred
                for pred in _preds
            ]
            for _preds, context in zip(preds, contexts)
        ]

        return preds

    def get_n_parameters(self):
        """
        Returns the number of parameters in the model in a human-readable format.

        Returns:
            str: The number of parameters in a human-readable format.
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

class ChatLMModel(LMModel):
    def prepare_prompted_inputs(self, src: List[str]):
        """
        Prepares inputs for beam search.

        Args:
            src (List[str]): The source sentences.
            contexts (List[str], optional): The context for each source sentence. Defaults to None.
            prompt_split (str, optional): The prompt split token. Defaults to "\\n".
            n_beam (int, optional): The number of beams. Defaults to 8.
            n_beam_hyps_to_keep (int, optional): The number of beam hypotheses to keep. Defaults to 1.

        Returns:
            tuple: A tuple containing model_kwargs, context_input_ids, context_attention_mask, and beam_scorer.
        """
        model_kwargs = self.get_model_kwargs()

        prompted_contexts = []
        for s in src:
            if os.getenv("DETAILED_PROMPT", "false").lower() == "true":
                context = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "你是一个优秀的中文纠错模型，中文纠错模型即更正用户输入句子中的错误。"},
                        {"role": "user", "content": "你需要识别并纠正用户输入的句子中可能的错别字、多字、漏字并输出正确的句子，在修改的同时尽可能减少对原句子的改动(不新增、删除和修改标点符号)。只输出没有错误的句子，不要添加任何其他解释或说明。如果句子没有错误，就直接输出和输入相同的句子。"},
                        {"role": "user", "content": s}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                context = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": f"纠正下面句子中的错别字以及多字少字错误，并给出修改后的句子。\n输入：{s}"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                ) + "输出："
            prompted_contexts.append(context)
        prompted_context_infos = self.tokenizer(prompted_contexts, return_tensors="pt", padding=True)
        prompted_context_infos.to(self.model.device)
        prompted_context_input_ids = prompted_context_infos["input_ids"]
        prompted_context_attention_mask = prompted_context_infos["attention_mask"]

        return (
            model_kwargs,
            prompted_context_input_ids,
            prompted_context_attention_mask,
        )


class QwenModel(LMModel):
    """
    A class for Qwen language models.

    Args:
        model (str): The name or path of the pre-trained Qwen model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, model, *args, **kwargs):
        try:
            super().__init__(model, attn_implementation="flash_attention_2", *args, **kwargs)
        except ImportError:
            print("FlashAttention2 is not available, using default attention implementation")
            super().__init__(model, *args, **kwargs)
        self.is_byte_level_tokenize = True

    def set_decoder_start_token_id(self):
        """
        Sets the decoder start token ID for Qwen models.
        """
        self.model.config.decoder_start_token_id = self.tokenizer.encode("\n")[0]

    def set_vocab_size(self):
        """
        Sets the vocabulary size for Qwen models.
        """
        self.model.vocab_size = self.model.lm_head.out_features

    def set_convert_ids_to_tokens(self):
        """
        Sets the convert_ids_to_tokens function for Qwen models.
        """
        self.vocab, bytes_decoder = get_vocab_decoder(self.vocab)
        self.model.convert_ids_to_tokens = partial(
            qwen1_5_convert_ids_to_tokens, decoder=bytes_decoder
        )

    def get_model_kwargs(self):
        """
        Gets the model-specific keyword arguments for Qwen models.
        Different from other models, Qwen uses <|endoftext|> as both eos_token and pad_token.
        Qwen uses DynamicCache for past_key_values.

        Returns:
            dict: A dictionary of keyword arguments.
        """
        eos_token_id = self.tokenizer.encode("<|endoftext|>")[0]
        pad_token_id = eos_token_id
        return {
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "is_encoder_decoder": False,
            "past_key_values": DynamicCache()
        }

class ChatQwenModel(ChatLMModel, QwenModel):
    pass

class LlamaModel(LMModel):
    """
    A class for Llama language models.

    Args:
        model (str): The name or path of the pre-trained Llama model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.is_byte_level_tokenize = True

    def set_decoder_start_token_id(self):
        """
        Sets the decoder start token ID for Llama models.
        """
        self.model.config.decoder_start_token_id = self.tokenizer.encode("\n")[-1]

    def set_vocab_size(self):
        """
        Sets the vocabulary size for Llama models.
        """
        self.model.vocab_size = self.model.lm_head.out_features

    def set_convert_ids_to_tokens(self):
        """
        Sets the convert_ids_to_tokens function for Llama models.
        """
        self.vocab, bytes_decoder = get_vocab_decoder(self.vocab)
        self.model.convert_ids_to_tokens = partial(
            qwen1_5_convert_ids_to_tokens, decoder=bytes_decoder
        )

    def prepare_beam_search_inputs(self, src: List[str], contexts: List[str] = None, prompt_split: str = "\n", n_beam: int = 8, n_beam_hyps_to_keep: int = 1):
        """
        Prepares inputs for beam search for Llama models.

        Args:
            src (List[str]): The source sentences.
            contexts (List[str], optional): The context for each source sentence. Defaults to None.
            prompt_split (str, optional): The prompt split token. Defaults to "\\n".
            n_beam (int, optional): The number of beams. Defaults to 8.
            n_beam_hyps_to_keep (int, optional): The number of beam hypotheses to keep. Defaults to 1.

        Returns:
            tuple: A tuple containing model_kwargs, context_input_ids, context_attention_mask, and beam_scorer.
        """
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        return super().prepare_beam_search_inputs(src, contexts, prompt_split, n_beam, n_beam_hyps_to_keep)

    def get_model_kwargs(self):
        """
        Gets the model-specific keyword arguments for Llama models.

        Returns:
            dict: A dictionary of keyword arguments.
        """
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        return {
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "is_encoder_decoder": False,
            "past_key_values": DynamicCache()
        }

class ChatLlamaModel(ChatLMModel, LlamaModel):
    pass

class BaichuanModel(LMModel):
    """
    A class for Baichuan language models.

    Args:
        model (str): The name or path of the pre-trained Baichuan model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def set_decoder_start_token_id(self):
        """
        Sets the decoder start token ID for Baichuan models.
        """
        stop_token = b"\n" if self.is_byte_level_tokenize else "\n"
        self.model.config.decoder_start_token_id = self.vocab[stop_token]

    def set_vocab_size(self):
        """
        Sets the vocabulary size for Baichuan models.
        """
        self.model.vocab_size = self.model.lm_head.weight.shape[0]

    def set_convert_ids_to_tokens(self):
        """
        Sets the convert_ids_to_tokens function for Baichuan models.
        """
        self.model.convert_ids_to_tokens = self.tokenizer.convert_ids_to_tokens

    def get_model_kwargs(self):
        """
        Gets the model-specific keyword arguments for Baichuan models.

        Returns:
            dict: A dictionary of keyword arguments.
        """
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        return {
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "is_encoder_decoder": False,
        }

class ChatBaichuanModel(ChatLMModel, BaichuanModel):
    pass

class InternLM2Model(LMModel):
    """
    A class for InternLM2 language models.

    Args:
        model (str): The name or path of the pre-trained InternLM2 model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def set_decoder_start_token_id(self):
        """
        Sets the decoder start token ID for InternLM2 models.
        """
        stop_token = b"\n" if self.is_byte_level_tokenize else "\n"
        self.model.config.decoder_start_token_id = self.vocab[stop_token]

    def set_vocab_size(self):
        """
        Sets the vocabulary size for InternLM2 models.
        """
        self.model.vocab_size = self.model.output.out_features

    def set_convert_ids_to_tokens(self):
        """
        Sets the convert_ids_to_tokens function for InternLM2 models.
        """
        self.model.convert_ids_to_tokens = self.tokenizer.convert_ids_to_tokens

    def get_model_kwargs(self):
        """
        Gets the model-specific keyword arguments for InternLM2 models.

        Returns:
            dict: A dictionary of keyword arguments.
        """
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        return {
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "is_encoder_decoder": False,
        }

class ChatInternLM2Model(ChatLMModel, InternLM2Model):
    pass

class UerModel(LMModel):
    """
    A class for UER language models.

    Args:
        model (str): The name or path of the pre-trained UER model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
    
    def set_decoder_start_token_id(self):
        """
        Sets the decoder start token ID for UER models.
        """
        stop_token = "[CLS]"
        self.model.config.decoder_start_token_id = self.vocab[stop_token]

    def set_vocab_size(self):
        """
        Sets the vocabulary size for UER models.
        """
        self.model.vocab_size = self.model.lm_head.out_features

    def set_convert_ids_to_tokens(self):
        """
        Sets the convert_ids_to_tokens function for UER models.
        """
        self.model.convert_ids_to_tokens = self.tokenizer.convert_ids_to_tokens

    def get_model_kwargs(self):
        """
        Gets the model-specific keyword arguments for UER models.

        Returns:
            dict: A dictionary of keyword arguments.
        """
        eos_token_id = 102
        pad_token_id = 0
        return {
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "is_encoder_decoder": False
        }

    def process_generated_outputs(self, outputs, contexts: List[str] = None, prompt_split: str = "\n", n_beam_hyps_to_keep: int = 1):
        """
        Processes the generated outputs for UER models.

        Args:
            outputs: The generated outputs.
            contexts (List[str], optional): The context for each output. Defaults to None.
            prompt_split (str, optional): The prompt split token. Defaults to "\\n".
            n_beam_hyps_to_keep (int, optional): The number of beam hypotheses to keep. Defaults to 1.

        Returns:
            List[List[str]]: The processed predictions.
        """
        preds = super().process_generated_outputs(outputs, contexts, prompt_split, n_beam_hyps_to_keep)
        return [
            [
                "".join(pred.split()) for pred in _preds
            ]
            for _preds in preds
        ]

class ChatUerModel(ChatLMModel, UerModel):
    pass

class AutoLMModel:
    """
    A factory class for automatically selecting and instantiating the appropriate language model.

    This class provides a static method to create instances of specific language model classes
    based on the model name or path provided.
    """

    @staticmethod
    def from_pretrained(model: str, use_chat_prompted_model: bool = False, *args, **kwargs):
        """
        Creates and returns an instance of the appropriate language model class based on the model name.

        Args:
            model (str): The name or path of the pre-trained model.
            *args: Variable length argument list to be passed to the model constructor.
            **kwargs: Arbitrary keyword arguments to be passed to the model constructor.

        Returns:
            LMModel: An instance of the appropriate language model class.

        Raises:
            ValueError: If an unsupported model type is specified.
        """
        if use_chat_prompted_model:
            if "qwen" in model.lower():
                return ChatQwenModel(model, *args, **kwargs)
            elif "llama" in model.lower():
                return ChatLlamaModel(model, *args, **kwargs)
            elif "Baichuan2" in model:
                return ChatBaichuanModel(model, *args, **kwargs)
            elif "internlm2" in model.lower():
                return ChatInternLM2Model(model, *args, **kwargs)
            elif "uer" in model.lower():
                return ChatUerModel(model, *args, **kwargs)
            else:
                raise ChatLMModel(model, *args, **kwargs)
        else:
            if "qwen" in model.lower():
                return QwenModel(model, *args, **kwargs)
            elif "llama" in model.lower():
                return LlamaModel(model, *args, **kwargs)
            elif "Baichuan2" in model:
                return BaichuanModel(model, *args, **kwargs)
            elif "internlm2" in model.lower():
                return InternLM2Model(model, *args, **kwargs)
            elif "uer" in model.lower():
                return UerModel(model, *args, **kwargs)
            else:
                return LMModel(model, *args, **kwargs)
