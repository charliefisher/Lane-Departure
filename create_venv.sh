#!/usr/bin/env bash
set -e

VENV_NAME=".venv"

python -m venv ${VENV_NAME}
source ${VENV_NAME}/Scripts/activate
pip install -r requirements.txt
deactivate
