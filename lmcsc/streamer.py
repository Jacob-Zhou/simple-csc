from typing import Optional
from transformers import BeamSearchScorer
from transformers.generation.streamers import BaseStreamer
from queue import Queue


class BeamStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
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

    def put(self, value: BeamSearchScorer):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        beam_scorer, decoded_text = value
        if (len(beam_scorer._beam_hyps) // beam_scorer.num_beam_groups) > 1:
            raise ValueError("BeamStreamer only supports batch size 1")
        decoded_text = decoded_text[0]

        # retrieve best hypotheses
        candidate_beams = [
            beam for beam_hyp in beam_scorer._beam_hyps for beam in beam_hyp.beams
        ]
        if len(candidate_beams) <= 0:
            text = self.tokenizer.decode(decoded_text, **self.decode_kwargs)
        else:
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            best_hyp_tuple = sorted_hyps.pop()
            best_hyp = best_hyp_tuple[1]
            text = self.tokenizer.decode(best_hyp, **self.decode_kwargs)

        text = text.replace("▁", " ")
        self.on_finalized_text(text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        self.on_finalized_text("", stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
