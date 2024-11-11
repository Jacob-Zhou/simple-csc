LMCorrector
===========

The ``LMCorrector`` class is a language model-based corrector that utilizes beam search with distortion probabilities to correct text input. It can be used to fix errors in text based on a pretrained language model.

Key Features
------------

- Utilizes beam search with distortion probabilities
- Supports various pretrained language models
- Configurable parameters for fine-tuning correction behavior
- Supports both batch and streaming correction modes

Usage
-----

Here's a basic example of how to use the ``LMCorrector``:

.. code-block:: python

   from lmcsc.corrector import LMCorrector

   # Initialize the corrector with a pretrained model
   corrector = LMCorrector("gpt2")

   # Correct a single sentence
   result = corrector("完善农产品上行发展机智。")
   print(result)  # Output: [('完善农产品上行发展机制。',)]

   # Correct multiple sentences
   results = corrector(["完善农产品上行发展机智。", "这是一个测试句子。"])
   print(results)

   # Use streaming mode
   for output in corrector("完善农产品上行发展机智。", stream=True):
       print(output)

Configuration
-------------

The ``LMCorrector`` can be configured using a YAML configuration file. The default configuration file is located at ``configs/default_config.yaml``. You can specify a custom configuration file path when initializing the corrector:

.. code-block:: python

   corrector = LMCorrector("gpt2", config_path="path/to/custom_config.yaml")

Advanced Usage
--------------

The ``LMCorrector`` class provides several advanced features and customization options:

1. Custom distortion probabilities
2. Faithfulness reward
3. Beam search parameters adjustment
4. Context-aware correction

For more details on these advanced features, please refer to the class documentation.

API Documentation
-----------------
.. automodule:: lmcsc.corrector
   :members:
   :undoc-members:
   :show-inheritance:
