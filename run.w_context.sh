#!/bin/bash

set -o nounset
set -o errexit
set -o pipefail

{
    . scripts/set_environment.sh
    args=$@
    for arg in $args; do
        eval "$arg"
    done

    echo "n_observed_chars:        ${n_observed_chars:=8}"
    echo "n_beam:                  ${n_beam:=8}"
    echo "batch_size:              ${batch_size:=200}"
    echo "alpha:                   ${alpha:=2.5}"
    echo "use_faithfulness_reward: ${use_faithfulness_reward:=true}"
    echo "prob_smooth:             ${prob_smooth:=-15.0}"
    echo "prefix_split:            ${prefix_split:=$'\n'}"
    echo "model:                   ${model:=baichuan-inc/Baichuan2-7B-Base}"
    echo "base_suite:              ${base_suite:=v1}"


    use_faithfulness_reward_flag=""
    if [ "$use_faithfulness_reward" = true ]; then
        echo "Setting --use-faithfulness-reward"
        use_faithfulness_reward_flag="--use-faithfulness-reward"
        suite="${base_suite}.alpha-${alpha}.prob_smooth-${prob_smooth}.n_beam-${n_beam}.n_observed_chars-${n_observed_chars}"
    else
        echo "Not setting --use-faithfulness-reward"
        suite="${base_suite}.no_faithfulness_reward.alpha-${alpha}.prob_smooth-${prob_smooth}.n_beam-${n_beam}.n_observed_chars-${n_observed_chars}"
    fi


    datasets=(
        "aishell1/test_with_contents.txt"
    )


    for dataset in "${datasets[@]}"; do
        dataset_name=$(echo "${dataset}" | cut -d. -f1)
        dataset_name=${dataset_name//\//_}
        mkdir -p "results/${model}/${suite}/${dataset_name}"
        python -u run.w_context.py \
            --input-file "datasets/${dataset}"  \
            --path "results/${model}/${suite}/${dataset_name}"  \
            --model-name "${model}"  \
            --n-observed-chars "${n_observed_chars}"  \
            --prefix-split "${prefix_split}"  \
            --n-beam "${n_beam}"  \
            --batch-size "${batch_size}"  \
            --max-length 256  \
            --max-sentences-per-batch 64  \
            --alpha "${alpha}" $use_faithfulness_reward_flag \
            --distortion-model-smoothing "${prob_smooth}" | tee "results/${model}/${suite}/${dataset_name}/prediction.log"
        if [[ "$dataset_name" =~ "aishell1" ]]; then
            python eval/evaluate.py \
                --gold "datasets/aishell1/test.txt"  \
                --hypo "results/${model}/${suite}/${dataset_name}/prediction.txt"  \
                --to_halfwidth  \
                --ignore_unmatch_length  \
                --ignore_space
            python eval/evaluate_wer.py --char=1 --v=1 "datasets/aishell1/test.txt" "results/${model}/${suite}/${dataset_name}/prediction.txt" | tee "results/${model}/${suite}/${dataset_name}/prediction.wer.result"
        else
            python eval/evaluate.py \
                --gold "datasets/${dataset}"  \
                --hypo "results/${model}/${suite}/${dataset_name}/prediction.txt"  \
                --to_halfwidth  \
                --ignore_unmatch_length  \
                --ignore_space
        fi
    done

}
