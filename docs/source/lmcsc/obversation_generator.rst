ObservationGenerator
====================

The ``obversation_generator`` module provides classes for generating and managing observations during the beam search process in the LMCSC (Language Model-based Corrector with Semantic Constraints) system.

Key Components
--------------

- ``BaseObversationGenerator``: Abstract base class for observation generators.
- ``NextObversationGenerator``: Concrete implementation of an observation generator.

BaseObversationGenerator
------------------------

The ``BaseObversationGenerator`` class serves as an abstract base class for observation generators. It defines the interface that all observation generators should implement.

Key Methods:
^^^^^^^^^^^^

- ``reorder``: Reorders the beams based on given indices.
- ``step``: Performs a step in the beam search process.
- ``show_steps``: Displays the steps taken in the beam search process.
- ``get_observed_sequences``: Retrieves the observed sequences from the beam search process.

NextObversationGenerator
------------------------

The ``NextObversationGenerator`` class is a concrete implementation of ``BaseObversationGenerator``. It records the progress of the beam search, tracking what has been generated so far and what characters are yet to be generated.

Key Features:
^^^^^^^^^^^^^

- Supports both string and byte-level operations
- Tracks predictions, steps, and completion status for each beam
- Provides verbose mode for detailed step tracking
- Handles reordering of beams during search
- Generates observed sequences based on the current state of the search

API Documentation
-----------------

.. automodule:: lmcsc.obversation_generator
   :members:
   :undoc-members:
   :show-inheritance:
