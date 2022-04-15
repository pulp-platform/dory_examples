#!/bin/bash
TEST_COUNT=100
K_IN_MAX=50
K_OUT_MAX=50
H_IN_MAX=20
W_IN_MAX=20
for i in {1..$TEST_COUNT}
do
	make K_IN=$((1 + $RANDOM % $K_IN_MAX)) K_OUT=$((1 + $RANDOM % $K_OUT_MAX)) H_IN=$((3 + $RANDOM % $H_IN_MAX)) W_IN=$((3 + $RANDOM % $W_IN_MAX)) FS=3
done
