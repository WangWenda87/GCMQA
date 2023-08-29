#!/bin/bash

name=$(echo $2 | rev | cut -d'/' -f1 | rev | cut -d'.' -f1)
cat $1 | sed -n '/^ATOM/p' | cut -c22 | sort | uniq | sed '/^$/d' > ${name}_chain.txt

for c in $(cat ${name}_chain.txt)
do
	./DockQ/DockQ.py $2 $1 -native_chain1 $c > ${name}_dq.txt
	cat ${name}_dq.txt | awk '$1=="DockQ"{print $2}' >> ${name}_value.txt
done
rm ${name}_chain.txt
rm ${name}_dq.txt
