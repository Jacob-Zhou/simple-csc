Streamer
========

The ``streamer`` module provides functionality for streaming the output of beam search during text generation. It includes the ``BeamStreamer`` class, which is designed to work with the LMCSC (Language Model-based Corrector with Semantic Constraints) system.

Key Components
--------------

- ``BeamStreamer``: A class that extends the ``BaseStreamer`` from the Transformers library to handle beam search output streaming.

BeamStreamer
------------

The ``BeamStreamer`` class is a specialized streamer for handling beam search output. It processes the beam search results and provides a streaming interface for accessing the generated text.

Key Features:
^^^^^^^^^^^^^

- Supports streaming of beam search results
- Handles decoding of tokens into text
- Provides an iterator interface for easy access to generated text
- Supports timeout for streaming operations
- Handles end-of-stream signaling

API Documentation
-----------------

.. automodule:: lmcsc.streamer
   :members:
   :undoc-members:
   :show-inheritance:
