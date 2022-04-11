K_IN ?= 16
K_OUT ?= 32
H_IN ?= 3
W_IN ?= 3
FS ?= 1
PAD ?= 0
DW ?= 0

all:
	python gen_mininet.py --Kin=$(K_IN) --Kout=$(K_OUT) --Hin=$(H_IN) --Win=$(W_IN) --Fs=$(FS) --Pad=$(PAD) --Dw=$(DW)
	python deploy_mininet.py
	cd application; make clean all run | fgrep "Test target layer successful: no errors"
