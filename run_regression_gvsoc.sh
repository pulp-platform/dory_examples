#!/bin/bash

NB_CORES=1
#nice -n10 
./bwruntests.py --report-junit -t 300 --yaml -o tests.xml -p ${NB_CORES} $1
if test $? -ne 0; then
    exit 1
fi
