from threading import Thread
from typing import List, Tuple, Union
from queue import Queue
import torch

import yaml

from lmcsc.generation import (
    process_reward_beam_search,
    token_transformation_to_probs,
    get_distortion_probs,
    distortion_guided_beam_search,
)
from lmcsc.model import AutoLMModel, LMModel
from lmcsc.obversation_generator import NextObversationGenerator
from lmcsc.streamer import BeamStreamer
from lmcsc.transformation_type import TransformationType
from lmcsc.common import MIN, OOV_CHAR
from lmcsc.utils import Alignment, clean_sentences, rebuild_sentences

class LMCorrector:
    """
    A language model-based corrector that utilizes beam search with distortion probabilities to correct text input.
    The corrector can be used to fix errors in text based on a pretrained language model.

    Args:
        model (Union[str, LMModel]): The pretrained language model or a string identifier of the model.
        config_path (str, optional): Path to the configuration file. Defaults to 'configs/default_config.yaml'.
        n_observed_chars (int, optional): Number of observed characters for the input. Defaults to None.
        n_beam (int, optional): Number of beams for beam search. Defaults to None.
        n_beam_hyps_to_keep (int, optional): Number of beam hypotheses to keep. Defaults to None.
        alpha (float, optional): Hyperparameter for the length reward during beam search. Defaults to None.
        temperature (float, optional): Temperature for the prompt-based LLM. Defaults to None.
        distortion_model_smoothing (float, optional): Smoothing factor for distortion model probabilities. Defaults to None.
        use_faithfulness_reward (bool, optional): Whether to use faithfulness reward in beam search. Defaults to None.
        customized_distortion_probs (dict, optional): Custom distortion probabilities for different transformation types. Defaults to None.
        max_length (int, optional): Maximum allowed length for the input. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Note:
        Default `None` means using the default value from the configuration file.

    """

    def __init__(
        self,
        model: Union[str, LMModel],
        prompted_model: Union[str, LMModel] = None,
        config_path: str = 'configs/default_config.yaml',
        n_observed_chars: int = None,
        n_beam: int = None,
        n_beam_hyps_to_keep: int = None,
        alpha: float = None,  # hyperparameter for the length reward
        temperature: float = None,
        distortion_model_smoothing: float = None,
        use_faithfulness_reward: bool = None,
        customized_distortion_probs: dict = None,
        max_length: int = None,
        use_chat_prompted_model: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes the LMCorrector with the given parameters and loads the configuration.

        """
        self.config_path = config_path

        # Load configuration from YAML file
        with open(self.config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        # Set parameters, using either the provided ones or those from the configuration, use `or` is very unsafe
        self.n_beam = n_beam if n_beam is not None else self.config['n_beam']
        self.n_beam_hyps_to_keep = n_beam_hyps_to_keep if n_beam_hyps_to_keep is not None else self.config['n_beam_hyps_to_keep']
        self.n_observed_chars = n_observed_chars if n_observed_chars is not None else self.config['n_observed_chars']
        self.alpha = alpha if alpha is not None else self.config['alpha']
        self.temperature = temperature if temperature is not None else self.config['temperature']
        self.distortion_model_smoothing = distortion_model_smoothing if distortion_model_smoothing is not None else self.config['distortion_model_smoothing']
        self.use_faithfulness_reward = use_faithfulness_reward if use_faithfulness_reward is not None else self.config['use_faithfulness_reward']
        self.distortion_probs = customized_distortion_probs if customized_distortion_probs is not None else self.config['distortion_probs']
        self.max_length = max_length if max_length is not None else self.config['max_length']

        # Load the language model
        if isinstance(model, str):
            self.lm_model = AutoLMModel.from_pretrained(model, *args, **kwargs)
        else:
            self.lm_model = model

        if prompted_model is None:
            self.prompted_model = None
        else:
            if isinstance(prompted_model, str):
                if not use_chat_prompted_model and prompted_model == model:
                    # If the prompted model is the same as the language model, use the same model
                    self.prompted_model = self.lm_model
                else:
                    self.prompted_model = AutoLMModel.from_pretrained(prompted_model, use_chat_prompted_model=use_chat_prompted_model, *args, **kwargs)
                    if prompted_model == model:
                        # the same model, release the memory
                        self.prompted_model.model.to("cpu")
                        del self.prompted_model.model
                        self.prompted_model.model = self.lm_model.model
            else:
                self.prompted_model = prompted_model

        self.model = self.lm_model.model
        self.tokenizer = self.lm_model.tokenizer
        self.vocab = self.lm_model.vocab
        self.is_byte_level_tokenize = self.lm_model.is_byte_level_tokenize

        # Decorate the model instance with necessary attributes and methods
        self.decorate_model_instance()

    def decorate_model_instance(self) -> None:
        """
        Decorates the model instance by setting necessary attributes,
        adding methods, and configuring distortion probabilities and transformation types.
        """
        # Set attributes for the model
        self.model.n_observed_chars = self.n_observed_chars
        self.model.is_byte_level_tokenize = self.is_byte_level_tokenize

        # Set distortion probabilities
        self.model.distortion_probs = self.distortion_probs

        # Set default distortion type priority from configuration
        default_distortion_type_prior_priority = self.config['distortion_type_prior_priority']
        self.default_distortion_type_prior_priority = {
            dt: len(default_distortion_type_prior_priority) - i for i, dt in enumerate(default_distortion_type_prior_priority)
        }

        # Dynamically adjust distortion type priority based on distortion probabilities
        self.distortion_type_prior_priority = list(sorted(
            self.default_distortion_type_prior_priority,
            key=lambda x: (self.model.distortion_probs[x], self.default_distortion_type_prior_priority[x]),
            reverse=True
        ))

        # Initialize transformation types with priority
        self.model.transformation_type = TransformationType(
            self.vocab,
            is_bytes_level=self.is_byte_level_tokenize,
            distortion_type_prior_priority=self.distortion_type_prior_priority,
            config_path=self.config_path
        )

        # Initialize token lengths
        self.model.token_length = (
            torch.ones((self.model.vocab_size,)).to(self.model.device) * MIN
        )
        for idx, l in self.model.transformation_type.token_length.items():
            if idx < self.model.vocab_size:
                self.model.token_length[idx] = l

        self.model.is_chinese_token = torch.zeros((self.model.vocab_size,), dtype=torch.bool).to(self.model.device)
        for idx, is_chinese in self.model.transformation_type.is_chinese_token.items():
            if idx < self.model.vocab_size:
                self.model.is_chinese_token[idx] = is_chinese

        self.model.transformation_type_cache = {}

        # Set additional model parameters
        self.model.alpha = self.alpha
        self.model.temperature = self.temperature
        self.model.distortion_model_smoothing = self.distortion_model_smoothing
        self.model.use_faithfulness_reward = self.use_faithfulness_reward
        self.model.max_entropy = (
            torch.tensor(self.model.vocab_size).float().log().to(self.model.device)
        )

        # Log the parameters
        self.print_params()

        # Add functions to the model instance
        self.model.token_transformation_to_probs = token_transformation_to_probs.__get__(self.model, type(self.model))
        self.model.get_distortion_probs = get_distortion_probs.__get__(self.model, type(self.model))
        self.model.distortion_guided_beam_search = distortion_guided_beam_search.__get__(self.model, type(self.model))
        self.model.process_reward_beam_search = process_reward_beam_search.__get__(self.model, type(self.model))

    def print_params(self):
        """
        Logs the current parameters of the model and the corrector.
        """
        print(f"Model: {self.lm_model.model_name}")
        print(f"Number of parameters: {self.lm_model.get_n_parameters()}")
        print(f"Beam size: {self.n_beam}")
        print(f"Alpha: {self.alpha}")
        print(f"Temperature: {self.temperature}")
        print(f"Use faithfulness reward: {self.use_faithfulness_reward}")
        print(f"Distortion model probs: {self.model.distortion_probs}")
        print(f"Max entropy: {self.model.max_entropy}")
        print(f"Distortion model smoothing: {self.distortion_model_smoothing}")
        print(f"N observed chars: {self.n_observed_chars}")
        print(f"Distortion type priority: {self.distortion_type_prior_priority}")

    def update_params(self, **kwargs):
        """
        Updates the parameters of the model and the corrector.

        Args:
            **kwargs: Arbitrary keyword arguments corresponding to the parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            if hasattr(self, key):
                setattr(self, key, value)

        # Update distortion type priority based on the new distortion probabilities
        self.distortion_type_prior_priority = list(sorted(
            self.default_distortion_type_prior_priority,
            key=lambda x: (self.model.distortion_probs[x], self.default_distortion_type_prior_priority[x]),
            reverse=True
        ))
        self.model.transformation_type.build_distortion_type_priority(self.distortion_type_prior_priority)

    def preprocess(self, src: Union[List[str], str], contexts: Union[List[str], str] = None):
        """
        Preprocesses the source text by cleaning and truncating.

        Args:
            src (Union[List[str], str]): Source text or list of source texts.
            contexts (Union[List[str], str], optional): Additional context texts. Defaults to None.

        Returns:
            Tuple[List[str], List[List[Tuple[int, int, str, str]]]]: Cleaned source texts and lists of changes made during cleaning.
        """
        # Truncate the source text to the maximum length
        src_to_predict = [s[:self.max_length] for s in src]
        # Clean the sentences and get the changes
        src, changes = clean_sentences(src_to_predict)
        return src, changes

    def postprocess(
            self,
            preds: List[str],
            ori_srcs: List[str],
            changes: List[List[Tuple[int, int, str, str]]],
            append_src_left_over: bool = True
    ):
        """
        Postprocesses the predictions to rebuild the sentences and handle out-of-vocabulary characters.

        Args:
            preds (List[str]): List of predicted texts.
            ori_srcs (List[str]): List of original source texts.
            changes (List[List[Tuple[int, int, str, str]]]): Lists of changes made during preprocessing.
            append_src_left_over (bool, optional): Whether to append leftover source text after the max length. Defaults to True.

        Returns:
            List[List[str]]: Postprocessed predictions.
        """
        processed_preds = []
        for output in zip(*preds):
            # Rebuild sentences using the changes
            output = rebuild_sentences(output, changes)
            if append_src_left_over is not None:
                # Append the left-over source text beyond max_length
                output = [
                    (o + t[self.max_length:])
                    for o, t in zip(output, ori_srcs)
                ]

            # Handle out-of-vocabulary characters by aligning predictions with source text
            new_output = []
            for p, s in zip(output, ori_srcs):
                if not append_src_left_over:
                    s = s[:self.max_length]
                pred_chars = list(p)
                src_chars = list(s)
                new_text = ""
                pred_edits = Alignment(src_chars, pred_chars).align_seq
                for _, s_b, s_e, p_b, p_e in pred_edits:
                    if p[p_b:p_e] == OOV_CHAR:
                        # Replace OOV_CHAR in prediction with the original source characters
                        new_text += s[s_b:s_e]
                    else:
                        new_text += p[p_b:p_e]
                new_output.append(new_text)
            output = new_output

            processed_preds.append(output)
        
        # Transpose the list to get predictions per example
        processed_preds = list(zip(*processed_preds))
        return processed_preds

    def _stream_run(self, 
                    src: List[str],
                    changes: List[List[Tuple[int, int, str, str]]],
                    contexts: List[str],
                    prompt_split: str,
                    generation_kwargs: dict,
                    n_beam_hyps_to_keep: int = None
                    ):
        """
        Internal method to run the correction in streaming mode.

        Args:
            src (List[str]): List of source texts.
            changes (List[List[Tuple[int, int, str, str]]]): Lists of changes made during preprocessing.
            contexts (List[str]): List of context texts.
            prompt_split (str): The prompt split string.
            generation_kwargs (dict): Keyword arguments for generation.
            n_beam_hyps_to_keep (int, optional): Number of beam hypotheses to keep. Defaults to None.

        Yields:
            Generator: Yields intermediate predictions or errors during streaming.
        """
        if n_beam_hyps_to_keep is None:
            n_beam_hyps_to_keep = self.n_beam_hyps_to_keep

        # Initialize the streamer
        streamer = BeamStreamer(self.tokenizer, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        error_queue = Queue()

        def thread_target():
            try:
                self.model.process_reward_beam_search(**generation_kwargs)
            except torch.cuda.OutOfMemoryError as e:
                error_queue.put(e)
                streamer.end()
            except Exception as e:
                error_queue.put(e)

        # Start the generation in a separate thread
        thread = Thread(target=thread_target)
        thread.start()

        while True:
            if not error_queue.empty():
                # If there is an error, yield the error and stop
                yield error_queue.get()
                thread.join()
            try:
                # Get the next output from the streamer
                output = next(streamer)
            except StopIteration:
                break
            # Process the output predictions
            preds = self.lm_model.process_generated_outputs([output], contexts, prompt_split, n_beam_hyps_to_keep, need_decode=False)
            preds = self.postprocess(preds, src, changes, append_src_left_over=False)
            yield preds

        thread.join()

    def __call__(
            self,
            src: Union[List[str], str],
            contexts: Union[List[str], str] = None,
            prompt_split: str = "\n",
            observed_sequence_generator_cls: type = NextObversationGenerator,
            n_beam: int = None,
            n_beam_hyps_to_keep: int = None,
            stream: bool = False
    ):
        """
        Runs the corrector on the given source text(s) with optional contexts.

        Args:
            src (Union[List[str], str]): Source text(s) to correct.
            contexts (Union[List[str], str], optional): Additional context text(s) for each source. Defaults to None.
            prompt_split (str, optional): The prompt split string used in the tokenizer. Defaults to "\\n".
            observed_sequence_generator_cls (type, optional): The class to generate observed sequences. Defaults to NextObversationGenerator.
            n_beam (int, optional): Number of beams for beam search. Defaults to None.
            n_beam_hyps_to_keep (int, optional): Number of beam hypotheses to keep. Defaults to None.
            stream (bool, optional): Whether to run in streaming mode. Defaults to False.

        Returns:
            List[Tuple[str]] or Generator: Returns corrected text(s) or a generator if streaming.

        Note:
            If `stream` is True, the returned value is a generator that yields __intermediate predictions__.
            Otherwise, the returned value is a list of corrected texts.

        Examples:
            >>> corrector("完善农产品上行发展机智。")
            [('完善农产品上行发展机制。',)]

            >>> for output in corrector("完善农产品上行发展机智。", stream=True):
            ...     print(output)
            [('完善',)]
            [('完善农产品',)]
            [('完善农产品上',)]
            [('完善农产品上行',)]
            [('完善农产品上行发展',)]
            [('完善农产品上行发展机制',)]
            [('完善农产品上行发展模式。',)]
            [('完善农产品上行发展机制。',)]
            [('完善农产品上行发展机制。',)]
            [('完善农产品上行发展机制。',)]
            [('完善农产品上行发展机制。',)]
            [('完善农产品上行发展机制。',)]
            [('完善农产品上行发展机制。',)]

        """
        if n_beam is None:
            n_beam = self.n_beam
        if n_beam_hyps_to_keep is None:
            n_beam_hyps_to_keep = self.n_beam_hyps_to_keep

        if isinstance(src, str):
            src = [src]
        if contexts is not None:
            if isinstance(contexts, str):
                contexts = [contexts]
            assert len(src) == len(contexts), f"src and contexts must have the same length, got {len(src)} and {len(contexts)}"

        if stream:
            assert len(src) == 1, f"Stream only supports batch size 1, got {len(src)}"

        # Preprocess the source texts
        processed_src, changes = self.preprocess(src, contexts)

        # Prepare inputs for beam search generation
        (
            model_kwargs,
            context_input_ids,
            context_attention_mask,
            beam_scorer,
        ) = self.lm_model.prepare_beam_search_inputs(processed_src, contexts, prompt_split, n_beam, n_beam_hyps_to_keep)

        if self.prompted_model is not None:
            # Prepare inputs for prompted model
            (
                prompted_model_kwargs,
                prompted_context_input_ids,
                prompted_context_attention_mask,
            ) = self.prompted_model.prepare_prompted_inputs(processed_src)
            prompted_model = self.prompted_model.model
        else:
            prompted_model_kwargs = None
            prompted_context_input_ids = None
            prompted_context_attention_mask = None
            prompted_model = None

        # Initialize the observed sequence generator
        observed_sequence_generator = observed_sequence_generator_cls(
            processed_src,
            self.n_beam,
            self.n_observed_chars,
            self.is_byte_level_tokenize,
            verbose=False,
        )

        if stream:
            # Run in streaming mode
            generation_kwargs = dict(
                observed_sequence_generator=observed_sequence_generator,
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                prompted_model=prompted_model,
                prompted_input_ids=prompted_context_input_ids,
                prompted_attention_mask=prompted_context_attention_mask,
                prompted_model_kwargs=prompted_model_kwargs,
                beam_scorer=beam_scorer,
                **model_kwargs,
            )
            return self._stream_run(src, changes, contexts, prompt_split, generation_kwargs)
        else:
            # Run the beam search generation
            with torch.no_grad():
                outputs = self.model.process_reward_beam_search(
                    observed_sequence_generator,
                    input_ids=context_input_ids,
                    attention_mask=context_attention_mask,
                    prompted_model=prompted_model,
                    prompted_input_ids=prompted_context_input_ids,
                    prompted_attention_mask=prompted_context_attention_mask,
                    prompted_model_kwargs=prompted_model_kwargs,
                    beam_scorer=beam_scorer,
                    **model_kwargs,
                )

            # Process and postprocess the outputs
            preds = self.lm_model.process_generated_outputs(outputs, contexts, prompt_split, n_beam_hyps_to_keep)
            preds = self.postprocess(preds, src, changes, append_src_left_over=True)

            return preds

