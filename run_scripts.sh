#!/bin/bash

source virtual_environment/bin/activate

echo "Scripts disponíveis:"

select FILE in *.py; do
	if [[ -n "FILE" ]]; then
		echo "Executando $FILE"
		python3 "$FILE"
		break
	else 
		echo "Seleção Inválida!"
	fi
done
deactivate
