#!/bin/bash

for year in {2013..2019}
do
	for lr in 1e-04 3e-04 1e-03 3e-03
	do	
		for lambda_r in 1e-04 3e-04 1e-03 3e-03
		do
			python3.8 -u train.py \
			--mode "$1" \
			--theta_c "$2" \
			--cuda "$3" \
			--epochs 1000 \
			--random_seed 1 \
			--year $year \
			--lr $lr \
			--lambda_r $lambda_r
		done
		wait
	done
done
