#!/usr/bin/env bash
exp_id=$(date "+%Y%m%d-%H%M")
run_num=5
dataset="replica"
scene_list="${dataset}/office0 ${dataset}/office2 ${dataset}/office3 ${dataset}/office4 ${dataset}/room0 ${dataset}/room1 ${dataset}/room2 ${dataset}/hotel0"
planner_list="confidence"

for run in $(seq 1 $run_num); do
    for scene in $scene_list; do
        python data_generation.py scene=$scene
        for planner in $planner_list; do
            echo "experiment on $scene using $planner planner !!!!!!!"
            python main.py planner=$planner scene=$scene use_gui=False experiment.exp_id=$exp_id experiment.run_id=$run
            python mesh_generation.py planner=$planner scene=$scene experiment.exp_id=$exp_id experiment.run_id=$run
            python eval.py planner=$planner scene=$scene test_folder=dataset/$scene experiment.exp_id=$exp_id experiment.run_id=$run
        done
    done
done
