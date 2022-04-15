#!/bin/bash

NB_CORES=1
#nice -n10 
./bwruntests.py --report-junit -t 300 --yaml -o tests.xml -p ${NB_CORES} gvsoc_basic_$1x$1.yml
if test $? -ne 0; then
    exit 1
fi
