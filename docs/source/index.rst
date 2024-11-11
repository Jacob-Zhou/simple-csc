LMCSC Package
=============

The LMCSC (Language Model for Chinese Spelling Check) package provides a way to convert a Chinese Language Model into a Chinese Spelling Check Model without any extra training.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Modules

   lmcsc/corrector
   lmcsc/generation
   lmcsc/model
   lmcsc/obversation_generator
   lmcsc/streamer
   lmcsc/transformation_type

Corrector
---------

The :doc:`lmcsc/corrector` module contains the ``LMCorrector`` class, which is the main interface for text correction. It utilizes beam search with distortion probabilities to correct input text based on pretrained language models.

Generation Function
------------------

The :doc:`lmcsc/generation` module provides functionality for text generation using language models with distortion-guided beam search. It extends standard beam search by incorporating distortion probabilities and observed sequences.

Model
-----

The :doc:`lmcsc/model` module includes various language model implementations used in the LMCSC system. It provides a base ``LMModel`` class and specific implementations for different model architectures such as Qwen, Llama, Baichuan, InternLM2, and UER.

ObservationGenerator
--------------------

The :doc:`lmcsc/obversation_generator` module contains classes for generating and managing observations during the beam search process. It includes the ``BaseObversationGenerator`` abstract class and the ``NextObversationGenerator`` concrete implementation.

Streamer
--------

The :doc:`lmcsc/streamer` module provides functionality for streaming the output of beam search during text generation. It includes the ``BeamStreamer`` class, which handles beam search output streaming.

TransformationType
------------------

The :doc:`lmcsc/transformation_type` module contains the ``TransformationType`` class, which is responsible for handling various types of transformations on input sequences, particularly for Chinese text.

These modules work together to provide a powerful and flexible system for text correction and generation, with a focus on handling Chinese text and incorporating advanced language models.
