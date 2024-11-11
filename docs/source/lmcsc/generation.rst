Generation
==========

The ``generation`` module provides functionality for text generation using language models with distortion-guided beam search. It extends the capabilities of standard beam search by incorporating distortion probabilities and observed sequences.

Key Components
--------------

- ``token_transformation_to_probs``: Transforms observed sequences into token indices and probabilities.
- ``get_distortion_probs``: Computes distortion probabilities for a batch of observed sequences.
- ``distortion_probs_to_cuda``: Transfers distortion probabilities to a CUDA tensor.
- ``distortion_guided_beam_search``: Implements the main beam search algorithm with distortion guidance.

API Documentation
-----------------
.. automodule:: lmcsc.generation
   :members:
   :undoc-members:
   :show-inheritance:
