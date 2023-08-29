#!/bin/bash

NUM_EPISODES=1000
for i in {1..15}; do
	# Skip if we've already collected enough data
	PKL_COUNT=$(find "experiments/statistics/data/model_rand_region_$i/" -name "*.pkl" | wc -l)
	if [ "$PKL_COUNT" -lt "$NUM_EPISODES" ]; then 
		REMAINING=$((NUM_EPISODES - PKL_COUNT))
        python experiments/statistics/gather_data.py --model_file "trained_models/maze_I/model_rand_region_$i.pth" --num_episodes $REMAINING 
    fi
done

