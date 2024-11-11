TransformationType
==================

The ``TransformationType`` class is responsible for handling various types of transformations on input sequences, particularly for Chinese text. It provides functionality to identify and categorize different types of character transformations, such as similar shapes, similar pronunciations, and common confusions in Chinese characters.

Key Features
------------

- Handles multiple types of character transformations
- Supports both character-level and byte-level processing
- Utilizes various dictionaries for efficient lookup of similar characters, pronunciations, and shapes
- Configurable priority order for different distortion types

Usage
-----

Here's a basic example of how to use the ``TransformationType`` class:

.. code-block:: python

   from lmcsc.transformation_type import TransformationType

   # Initialize the TransformationType with a vocabulary
   vocab = {...}  # Your vocabulary dictionary
   transformer = TransformationType(vocab, is_bytes_level=False)

   # Get transformation types for a sequence
   observed_sequence = "你好"
   transformations, _ = transformer.get_transformation_type(observed_sequence)
   print(transformations)

Configuration
-------------

The ``TransformationType`` class can be configured using a YAML configuration file. The default configuration file is located at ``configs/default_config.yaml``. You can specify a custom configuration file path when initializing the class:

.. code-block:: python

   transformer = TransformationType(vocab, is_bytes_level=False, config_path="path/to/custom_config.yaml")

Transformation Types
--------------------

The class identifies several types of transformations:

- **IDT**: Identical character (no transformation)
- **PTC**: Prone to confusion (commonly confused characters)
- **SAP**: Same pinyin (characters that share the same pinyin)
- **SIP**: Similar pinyin (characters with similar pinyin)
- **SIS**: Similar shape (characters with similar visual appearance)
- **OTP**: Other pinyin error (pinyin-related errors not covered by SAP or SIP)
- **OTS**: Other similar shape (shape-related errors not covered by SIS)
- **UNR**: Unrecognized transformation (no known transformation type)

Advanced Usage
--------------

The ``TransformationType`` class provides several advanced features:

1. Handling of Out-of-Vocabulary (OOV) characters
2. Building inverse indices for efficient lookup
3. Customizable distortion type priorities
4. Support for both character-level and byte-level processing

For more details on these advanced features, please refer to the class documentation.

API Documentation
-----------------

.. automodule:: lmcsc.transformation_type
   :members:
   :undoc-members:
   :show-inheritance:
