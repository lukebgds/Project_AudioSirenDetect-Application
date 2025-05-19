#!/bin/bash

VENV_PATH="virtual_environment/bin/activate"

if [[ ! -f "$VENV_PATH" ]]; then
    echo "Erro: Ambiente virtual não encontrado em $VENV_PATH"
    exit 1
fi

source "$VENV_PATH"

echo "Executando main.py"
python3 main.py

deactivate