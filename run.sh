#!/bin/sh

cd src
python3 main.py
python3 q_a.py > ../q_a_print.txt
python3 q_b.py > ../q_b_print.txt
python3 q_c.py > ../q_c_print.txt