from typing import Optional
from transformers import BeamSearchScorer
from transformers.generation.streamers import BaseStreamer
from queue import Queue


class BeamStreamer(BaseStreamer):
    """
    A streamer class that handles beam search output streaming during text generation.

    This class extends BaseStreamer to provide functionality for streaming beam search results,
    processing the beam hypotheses, and providing an iterator interface for accessing the generated text.

    Notes:
        This class only supports batch size 1.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens into text.
        timeout (`float`, *optional*, defaults to `None`):
            The timeout in seconds for queue operations. If None, queue operations block indefinitely.
        **decode_kwargs:
            Additional keyword arguments passed to the tokenizer's decode method.

    Attributes:
        tokenizer (`AutoTokenizer`):
            The tokenizer instance used for decoding.
        decode_kwargs (`dict`):
            Additional arguments for token decoding.
        print_len (`int`):
            Length of previously printed text.
        text_queue (`Queue`):
            Queue for storing generated text chunks.
        stop_signal:
            Signal used to indicate end of stream.
        timeout (`float`):
            Timeout value for queue operations.
        last_text (`str`):
            Most recently generated text.

    Examples:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from lmcsc.streamer import BeamStreamer
        >>> 
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> streamer = BeamStreamer(tokenizer)
        >>> 
        >>> # Stream generated text
        >>> for text in streamer:
        ...     print(text)
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        timeout: Optional[float] = None,
        **decode_kwargs
    ):
        self.tokenizer = tokenizer
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.print_len = 0

        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout
        self.last_text = ""

    def put(self, value: BeamSearchScorer):
        """
        Receives tokens, decodes them, and puts the decoded text into the queue.

        Args:
            value (tuple): A tuple containing (BeamSearchScorer, decoded_text).
                The BeamSearchScorer contains beam hypotheses and the decoded_text is a list of token IDs.

        Raises:
            ValueError: If batch size is greater than 1.
        """
        beam_scorer, decoded_text = value
        if (len(beam_scorer._beam_hyps) // beam_scorer.num_beam_groups) > 1:
            raise ValueError("BeamStreamer only supports batch size 1")

        # retrieve best hypotheses
        candidate_beams = [
            beam for beam_hyp in beam_scorer._beam_hyps for beam in beam_hyp.beams
        ]
        if len(candidate_beams) <= 0:
            best_hyp = decoded_text[0]
        else:
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            best_hyp_tuple = sorted_hyps.pop()
            best_hyp = best_hyp_tuple[1]

        text = self.tokenizer.decode(best_hyp, **self.decode_kwargs)

        text = text.replace("▁", " ")

        self.last_text = text

        self.on_finalized_text(text)

    def end(self):
        """Signals the end of the stream by putting the stop signal in the queue."""
        self.on_finalized_text(self.last_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        Puts finalized text into the queue and handles stream end signaling.

        Args:
            text (str): The text to put in the queue.
            stream_end (bool, optional): Whether this is the end of the stream. Defaults to False.
        """
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)
            self.last_text = ""

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
