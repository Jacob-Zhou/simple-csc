#!/bin/bash

[ ! -d venv ] && virtualenv venv && . venv/bin/activate && pip install -r requirements.txt
. venv/bin/activate
