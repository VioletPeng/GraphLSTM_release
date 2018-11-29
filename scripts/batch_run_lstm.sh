#!/bin/bash

for i in {0..4}; do
    echo ${i}
    run_lstm.sh ${i} ${1}Relation gpu #WeightedGraphLSTM
    run_drug_var.sh ${i} ${1}Relation gpu  #WeightedGraphLSTM
done
