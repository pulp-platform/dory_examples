K_IN ?= 20
K_OUT ?= 40
H_IN ?= 3
W_IN ?= 3
FS ?= 1
PAD ?= 0
DW ?= 0

all:
	python gen_mininet.py --Kin=$(K_IN) --Kout=$(K_OUT) --Hin=$(H_IN) --Win=$(W_IN) --Fs=$(FS) --Pad=$(PAD) --Dw=$(DW)
	python deploy_mininet.py
	cd application; make clean all run | fgrep -B 30 "Test target layer successful: no errors"

debug:
	APP_CFLAGS=-DGVSOC_LOGGING $(MAKE) -C application clean all run runner_args="--trace=ne16" > logs/ne16.log
	cat logs/ne16.log | grep -B 42 STREAMOUT | grep -A 32 accum > logs/output.log
	cat logs/ne16.log | grep -A 36 NORMQUANT_BIAS | grep -A 35 'nqb_iter=3' > logs/acc_before_clip.log
	cat logs/ne16.log | grep -A 34 'Exiting MATRIXVEC' | grep -B 33 'NORMQUANT_MULT' > logs/interm.log
	cat logs/ne16.log | grep -B 25 'x_array' > logs/inputs.log

COUNT_SUBTILES = $(shell echo '($(wc -l < logs/acc_before_norm_quant.log) + 1) / 35' | bc)

count-subtiles:
	echo $(COUNT_SUBTILES)

preprocess-inputs:
	sed '/--\|x_array/d' logs/inputs.log | sed 's/{//g' | sed 's/}//g' | sed 's/ //g' > logs/inputs_cleaned.log

preprocess-interm:
	sed '/--\|accum\|NORMQUANT_MULT/d' logs/interm.log | sed 's/{//g' | sed 's/}//g' | sed 's/ //g' > logs/interm_cleaned.log

preprocess-output:
	sed '/--\|accum/d' logs/output.log | sed 's/{//g' | sed 's/}//g' | sed 's/ //g' > logs/output_cleaned.log

preprocess: preprocess-inputs preprocess-output preprocess-interm

compare-inputs:
	python sort_inputs.py
	diff logs/input_sorted.log MiniNet/input.txt

compare-intermediates:
	python sort_intermediates.py
	diff -y MiniNet/interm_layer0.txt logs/intermediates_sorted.log > logs/interm_diff.log

build:
	$(MAKE) -C application all

run:
	$(MAKE) -C application run
