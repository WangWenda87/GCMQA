#! /bin/bash
mkdir ./result
for pdb in $(ls ./examples)
do
	for i in {1..20}
	do
		./run_dq.sh $pdb $i
		mv RESULTS.txt ./result/${pdb}_$i.txt
	done
done
