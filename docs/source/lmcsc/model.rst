Model
=====

The ``model`` module provides classes for different language models used in the LMCSC (Language Model-based Corrector with Semantic Constraints) system. It includes a base class ``LMModel`` and several specific model implementations.

Key Components
--------------

- ``LMModel``: Base class for language models.
- ``QwenModel``: Class for Qwen language models.
- ``LlamaModel``: Class for Llama language models.
- ``BaichuanModel``: Class for Baichuan language models.
- ``InternLM2Model``: Class for InternLM2 language models.
- ``UerModel``: Class for UER language models.
- ``AutoLMModel``: Factory class for automatically selecting and instantiating the appropriate language model.

LMModel
-------

The ``LMModel`` class serves as the base class for all language models in the LMCSC system. It provides common functionality and interfaces for working with different types of language models.

Key Features:
^^^^^^^^^^^^^

- Initialization with pre-trained models
- Tokenization and vocabulary management
- Beam search preparation and output processing
- Model parameter counting

Model-Specific Classes
----------------------

The module provides specific implementations for various language models:

- ``QwenModel``: Optimized for Qwen models, with support for FlashAttention2.
- ``LlamaModel``: Tailored for Llama models, with specific tokenization and padding strategies.
- ``BaichuanModel``: Designed for Baichuan models, with custom token handling.
- ``InternLM2Model``: Specialized for InternLM2 models.
- ``UerModel``: Adapted for UER models, with specific output processing.

Each of these classes inherits from ``LMModel`` and overrides certain methods to accommodate the unique characteristics of their respective model architectures.

AutoLMModel
-----------

The ``AutoLMModel`` class provides a convenient way to instantiate the appropriate language model based on the model name or path. It automatically selects the correct model class and initializes it with the given parameters.

Example:

.. code-block:: python

   from lmcsc.model import AutoLMModel

   # Create a Qwen model
   qwen_model = AutoLMModel.from_pretrained("qwen-7b")

   # Create a Llama model
   llama_model = AutoLMModel.from_pretrained("llama-7b")

   # Create a Baichuan model
   baichuan_model = AutoLMModel.from_pretrained("Baichuan2-7B-Base")

This factory pattern allows for easy integration of new model types and simplifies the process of working with different language models within the LMCSC system.

API Documentation
-----------------

.. automodule:: lmcsc.model
   :members:
   :undoc-members:
   :show-inheritance: