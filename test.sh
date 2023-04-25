#!/bin/bash

lexd test.lexd > test.att
./learn_fst.py --att test.att --alpha 0.1 --epochs 1000 --graph --model test.model
./learn_fst.py --model test.model --input 'n a n a y a'
./learn_fst.py --model test.model --input 'd a d a d a'

