#!/bin/bash

python3 experiments/collect_retargeting_data.py --num_levels=100 --intervention="normal" &

python3 experiments/collect_retargeting_data.py --num_levels=100 --intervention="cheese" &

python3 experiments/collect_retargeting_data.py --num_levels=100 --magnitude=5.5 --intervention=55 &

python3 experiments/collect_retargeting_data.py --num_levels=100 --magnitude=2.3 --intervention="effective" &

python3 experiments/collect_retargeting_data.py --num_levels=100 --magnitude=1.0 --intervention="all" &

