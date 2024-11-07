from typing import Union, List
from copy import deepcopy


class BaseObversationGenerator:
    def reorder(self, beam_idx: List[int]) -> None:
        raise NotImplementedError

    def step(self, token_lists: List[List[Union[str, bytes]]]) -> None:
        raise NotImplementedError

    def show_steps(self) -> None:
        raise NotImplementedError

    def get_observed_sequences(self) -> List[str]:
        raise NotImplementedError


class NextObversationGenerator(BaseObversationGenerator):
    '''
        This class is recording the progress of the beam search.
        That is, what has been generated so far, what are characters that not yet generated.
    '''
    def __init__(self, src, n_beam, n_observed_chars, is_bytes_level, verbose=False):
        self.src = src
        self.n_beam = n_beam
        self.n_observed_chars = n_observed_chars
        self.is_bytes_level = is_bytes_level
        self.verbose = verbose

        if is_bytes_level:
            self.src = [s.encode("utf-8") for s in src]
            self.batch_steps = [[b""] * n_beam for _ in range(len(src))]
        else:
            self.batch_steps = [[""] * n_beam for _ in range(len(src))]

        self.verbose = verbose
        if self.verbose:
            self.batch_verbose_steps = [
                [[] for _ in range(n_beam)] for _ in range(len(src))
            ]
        self.is_finished = [[False] * n_beam for _ in range(len(src))]

    def reorder(self, beam_idx: List[int]) -> None:
        self.batch_steps = [
            [self.batch_steps[i][b] for b in beam] for i, beam in enumerate(beam_idx)
        ]
        self.is_finished = [
            [self.is_finished[i][b] for b in beam] for i, beam in enumerate(beam_idx)
        ]
        if self.verbose:
            self.batch_verbose_steps = [
                [
                    deepcopy(self.batch_verbose_steps[i][b]) for b in beam
                ] for i, beam in enumerate(beam_idx)
            ]

    def step(self, token_lists: List[List[Union[str, bytes]]]) -> None:
        for i, tokens in enumerate(token_lists):
            for j, token in enumerate(tokens):
                if self.is_finished[i][j]:
                    continue
                if token not in {"<|endoftext|>", "</s>", "[SEP]"}:
                    self.batch_steps[i][j] += token
                    if self.verbose:
                        self.batch_verbose_steps[i][j].append(token)
                else:
                    self.is_finished[i][j] = True

    def show_steps(self) -> None:
        batch_steps = self.batch_verbose_steps if self.verbose else self.batch_steps
        for steps in batch_steps:
            for step in steps:
                try:
                    if self.is_bytes_level:
                        if self.verbose:
                            print([s.decode("utf-8") for s in step])
                        else:
                            print(step.decode("utf-8"))
                    else:
                        print(step)
                except:
                    print(step)
            print()

    def get_observed_sequences(self) -> List[str]:
        batch_observed_sequences = []
        n_observed_chars = self.n_observed_chars
        for batch_idx, steps in enumerate(self.batch_steps):
            observed_sequences = []
            src = self.src[batch_idx]
            for step in steps:
                step = len(step)
                # In fact, there is a bug here.
                # We assume that all Chinese characters are 3 bytes long.
                # However, some Chinese characters are 4 bytes long.
                # When a 4 bytes character correct to 3 bytes, it will introduce a garbled character.
                if self.is_bytes_level:
                    try:
                        token = src[step:].decode("utf-8")
                        observed_sequence = token[:n_observed_chars]
                    except:
                        observed_sequence = src[
                            step : step + (n_observed_chars * 3)
                        ]
                else:
                    observed_sequence = src[step : step + n_observed_chars]
                    observed_sequence = observed_sequence.replace(" ", "▁")
                observed_sequences.append(observed_sequence)
            batch_observed_sequences.append(observed_sequences)
        return batch_observed_sequences
