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
    echo "max_sentences_per_batch: ${max_sentences_per_batch:=24}"
    echo "alpha:                   ${alpha:=2.5}"
    echo "temperature:             ${temperature:=1.5}"
    echo "use_faithfulness_reward: ${use_faithfulness_reward:=true}"
    echo "use_chat_prompted_model: ${use_chat_prompted_model:=false}"
    echo "prob_smooth:             ${prob_smooth:=-15.0}"
    echo "prefix_split:            ${prefix_split:=$'\n'}"
    echo "model:                   ${model:=Qwen/Qwen2.5-7B}"
    echo "prompted_model:          ${prompted_model:=Qwen/Qwen2.5-7B}"
    echo "config_path:             ${config_path:=configs/default_config.yaml}"
    echo "base_suite:              ${base_suite:=v1}"

    prompted_model_name=${prompted_model//\//_}
    suite="${prompted_model_name}-${base_suite}"


    use_faithfulness_reward_flag=""
    if [ "$use_faithfulness_reward" = true ]; then
        echo "Setting --use-faithfulness-reward"
        use_faithfulness_reward_flag="--use-faithfulness-reward"
    else
        echo "Not setting --use-faithfulness-reward"
        suite="${suite}.no_faithfulness_reward"
    fi

    use_chat_prompted_model_flag=""
    if [ "$use_chat_prompted_model" = true ]; then
        echo "Setting --use-chat-prompted-model"
        use_chat_prompted_model_flag="--use-chat-prompted-model"
        suite="${suite}.use_chat_prompted_model"
    else
        echo "Not setting --use-chat-prompted-model"
    fi

    suite="${suite}.alpha-${alpha}.temperature-${temperature}.prob_smooth-${prob_smooth}.n_beam-${n_beam}.n_observed_chars-${n_observed_chars}"

    datasets=(
        # "ecspell/law_500.txt"
        # "ecspell/med_500.txt"
        # "ecspell/odw_500.txt"
        # "sighan_rev/sighan13.txt"
        # "sighan_rev/sighan14.txt"
        # "sighan_rev/sighan15.txt"
        "c2ec/dev.txt"
        "lemon_v2/cot.txt"
        "lemon_v2/car.txt"
        "lemon_v2/enc.txt"
        "lemon_v2/gam.txt"
        "lemon_v2/mec.txt"
        "lemon_v2/new.txt"
        "lemon_v2/nov.txt"
        # "mcsc/test.txt"
        "cscd_ime/test.txt"
        "cscd_ime/dev.txt"
        "c2ec/test.txt"
    )


    for dataset in "${datasets[@]}"; do
        dataset_name=$(echo "${dataset}" | cut -d. -f1)
        dataset_name=${dataset_name//\//_}
        mkdir -p "results/${model}/${suite}/${dataset_name}"
        python -u run.py \
            --input-file "datasets/${dataset}"  \
            --path "results/${model}/${suite}/${dataset_name}"  \
            --model-name "${model}"  \
            --prompted-model-name "${prompted_model}"  \
            --config-path "${config_path}"  \
            --n-observed-chars "${n_observed_chars}"  \
            --prefix-split "${prefix_split}"  \
            --n-beam "${n_beam}"  \
            --batch-size "${batch_size}"  \
            --max-length 256  \
            --max-sentences-per-batch "${max_sentences_per_batch}"  \
            --alpha "${alpha}"  \
            --temperature "${temperature}" $use_faithfulness_reward_flag $use_chat_prompted_model_flag \
            --distortion-model-smoothing "${prob_smooth}" | tee "results/${model}/${suite}/${dataset_name}/prediction.log"
        if [[ "$dataset_name" =~ "ecspell" || "$dataset_name" =~ "sighan_rev" || "$dataset_name" =~ "lemon_v2" || "$dataset_name" =~ "cscd_ime" || "$dataset_name" =~ "mcsc" ]]; then
            # Ignore unmatch length for CSC datasets
            python eval/evaluate.py \
                --gold "datasets/${dataset}"  \
                --hypo "results/${model}/${suite}/${dataset_name}/prediction.txt"  \
                --to_halfwidth  \
                --ignore_unmatch_length  \
                --ignore_space
        else
            python eval/evaluate.py \
                --gold "datasets/${dataset}"  \
                --hypo "results/${model}/${suite}/${dataset_name}/prediction.txt"  \
                --to_halfwidth  \
                --ignore_space
        fi
    done

}
