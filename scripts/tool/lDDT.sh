#!/bin/bash

name=$(echo $1 | cut -d'.' -f1)
lddt single_chain_pdb/$1 single_chain_pdb/$2 > ${name}_lddt.txt
cat ${name}_lddt.txt | awk -F'\t' '{print $5}' | sed -n '/./p' | sed '1d' > ${name}_score.txt
rm ${name}_lddt.txt
