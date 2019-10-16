#!/bin/bash

declare -a max_assign_cnt_list=(2)
declare -a hidden_dim_list=(128 256)
declare -a n_layers_list=(1)
declare -a bidirectional_list=(False)

create_data=true
cnt=0
for max_assign_cnt in ${max_assign_cnt_list[@]}; do
    for hidden_dim in ${hidden_dim_list[@]}; do
        for n_layers in ${n_layers_list[@]}; do
            for bidirectional in ${bidirectional_list[@]}; do
                cnt=$(($cnt+1))
                echo The ${cnt}th experiment
                echo max assign cnt: $max_assign_cnt
                echo hidden dim: $hidden_dim
                echo n_layers: $n_layers
                echo bidirectional: $bidirectional
                echo create data: $create_data
                #if [ $cnt -lt 2 ]; then
                #    continue
                #fi

                if $create_data
                then
                    time python3 seq.py --train \
                                        --create_data \
                                        --max_assign_cnt=$max_assign_cnt \
                                        --hidden_dim=$hidden_dim \
                                        --n_layers=$n_layers \
                                        --bidirectional=$bidirectional \
                                        --model_path=model_assign${max_assign_cnt}_hidden${hidden_dim}_layer${n_layers}.pt
                else
                    create_data=false
                    time python3 seq.py --train \
                                        --max_assign_cnt=$max_assign_cnt \
                                        --hidden_dim=$hidden_dim \
                                        --n_layers=$n_layers \
                                        --bidirectional=$bidirectional \
                                        --model_path=model_assign${max_assign_cnt}_hidden${hidden_dim}_layer${n_layers}.pt
                fi
            done
        done
    done
    create_data=true
done
