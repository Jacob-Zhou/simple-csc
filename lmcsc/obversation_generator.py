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
    r"""
    This class records the progress of the beam search, tracking what has been generated so far
    and what characters are yet to be generated.

    Parameters:
        src (`List[str]`):
            The source sequences.
        n_beam (`int`):
            The number of beams for beam search.
        n_observed_chars (`int`):
            The number of characters to observe.
        is_bytes_level (`bool`):
            Whether to operate at the byte level.
        verbose (`bool`, *optional*, defaults to `False`):
            Whether to enable verbose mode.

    Attributes:
        src (`List[Union[str, bytes]]`):
            The source sequences, potentially encoded to bytes.
        n_beam (`int`):
            The number of beams.
        n_observed_chars (`int`):
            The number of characters to observe.
        is_bytes_level (`bool`):
            Whether operating at byte level.
        verbose (`bool`):
            Verbose mode flag.
        batch_predicts (`List[List[Union[str, bytes]]]`):
            Predictions for each beam in each batch.
        batch_steps (`List[List[int]]`):
            Steps taken for each beam in each batch.
        batch_verbose_steps (`List[List[List[Union[str, bytes]]]]`):
            Verbose steps for each beam in each batch.
        is_finished (`List[List[bool]]`):
            Flags indicating if each beam in each batch is finished.
    """

    def __init__(self, src, n_beam, n_observed_chars, is_bytes_level, verbose=False):
        self.src = src
        self.n_beam = n_beam
        self.n_observed_chars = n_observed_chars
        self.is_bytes_level = is_bytes_level
        self.verbose = verbose

        if is_bytes_level:
            self.src = [s.encode("utf-8") for s in src]
            self.batch_predicts = [[b""] * n_beam for _ in range(len(src))]
            # TODO: handle bytes level
        else:
            self.batch_predicts = [[""] * n_beam for _ in range(len(src))]
        self.batch_steps = [[0] * n_beam for _ in range(len(src))]
        self.insert_counters = [[0] * n_beam for _ in range(len(src))]

        self.verbose = verbose
        if self.verbose:
            self.batch_verbose_steps = [
                [[] for _ in range(n_beam)] for _ in range(len(src))
            ]
        self.is_finished = [[False] * n_beam for _ in range(len(src))]

    def reorder(self, beam_idx: List[int]) -> None:
        """
        Reorders the beams based on the given indices.

        Args:
            beam_idx (List[int]): The indices to reorder the beams.
        """
        self.batch_predicts = [
            [self.batch_predicts[i][b] for b in beam] for i, beam in enumerate(beam_idx)
        ]
        self.batch_steps = [
            [self.batch_steps[i][b] for b in beam] for i, beam in enumerate(beam_idx)
        ]
        self.is_finished = [
            [self.is_finished[i][b] for b in beam] for i, beam in enumerate(beam_idx)
        ]
        self.insert_counters = [
            [self.insert_counters[i][b] for b in beam] for i, beam in enumerate(beam_idx)
        ]
        if self.verbose:
            self.batch_verbose_steps = [
                [
                    deepcopy(self.batch_verbose_steps[i][b]) for b in beam
                ] for i, beam in enumerate(beam_idx)
            ]

    def step(self, token_lists: List[List[Union[str, bytes]]], step_lists: List[List[int]]):
        """
        Performs a step in the beam search process.

        Args:
            token_lists (List[List[Union[str, bytes]]]): The tokens generated in this step.
            step_lists (List[List[int]]): The corresponding steps for each token.
        """
        for i, (tokens, steps) in enumerate(zip(token_lists, step_lists)):
            for j, (token, step) in enumerate(zip(tokens, steps)):
                if self.is_finished[i][j]:
                    continue
                if token not in {"<|endoftext|>", "</s>", "[SEP]"}:
                    self.batch_predicts[i][j] += token
                    if step == 0:
                        self.insert_counters[i][j] += 1
                    else:
                        # reset the insert counter
                        self.insert_counters[i][j] = 0
                    if self.insert_counters[i][j] > 1:
                        # force to move forward
                        step = 1
                        if self.is_bytes_level:
                            src = self.src[i]
                            while True:
                                try:
                                    src[self.batch_steps[i][j] + step:].decode("utf-8")
                                    break
                                except:
                                    step += 1
                        self.insert_counters[i][j] = 0
                    self.batch_steps[i][j] += step
                    if self.verbose:
                        self.batch_verbose_steps[i][j].append(token)
                else:
                    self.is_finished[i][j] = True

    def show_steps(self) -> None:
        """
        Displays the steps taken in the beam search process.
        """
        batch_predicts = self.batch_verbose_steps if self.verbose else self.batch_predicts
        for predicts in batch_predicts:
            for predict in predicts:
                try:
                    if self.is_bytes_level:
                        if self.verbose:
                            print([s.decode("utf-8") for s in predict])
                        else:
                            print(predict.decode("utf-8"))
                    else:
                        print(predict)
                except:
                    print(predict)
            print()

    def get_observed_sequences(self) -> List[str]:
        """
        Retrieves the observed sequences from the beam search process.

        Returns:
            List[str]: The observed sequences for each beam in each batch.
        """
        batch_observed_sequences = []
        n_observed_chars = self.n_observed_chars
        for batch_idx, steps in enumerate(self.batch_steps):
            observed_sequences = []
            src = self.src[batch_idx]
            for step in steps:
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
