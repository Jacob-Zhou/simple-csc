#!/bin/bash

[ ! -d venv ] && virtualenv venv && . venv/bin/activate && pip install -r requirements.txt && scripts/download_datasets.sh
. venv/bin/activate
