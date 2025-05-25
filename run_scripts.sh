#!/bin/bash

echo "Ativando ambiente virtual"
source virtual_environment/bin/activate

echo "Executando main.py"
python3 src/main.py

deactivate
