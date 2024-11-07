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

    echo "n_observed_chars:      ${n_observed_chars:=8}"
    echo "n_beam:                ${n_beam:=8}"
    echo "batch_size:            ${batch_size:=200}"
    echo "alpha:                 ${alpha:=2.5}"
    echo "model:                 ${model:=baichuan-inc/Baichuan2-7B-Base}"
    echo "base_suite:            ${base_suite:=v1}"

    suite="${base_suite}.alpha-${alpha}.n_beam-${n_beam}.n_observed_chars-${n_observed_chars}"

    datasets=(
        "ecspell/law_500.txt"
        "ecspell/med_500.txt"
        # "ecspell/odw_500.txt"
        "sighan_rev/sighan13.txt"
        "sighan_rev/sighan14.txt"
        # "sighan_rev/sighan15.txt"
        "lemon_v2/car.txt"
        "lemon_v2/cot.txt"
        "lemon_v2/enc.txt"
        "lemon_v2/gam.txt"
        "lemon_v2/mec.txt"
        "lemon_v2/new.txt"
        # "lemon_v2/nov.txt"
        "lemon_v2/nov_1000.txt"
        "cscd_ime/test.txt"
        "mcsc/test.txt"
    )


    for dataset in "${datasets[@]}"; do
        dataset_name=$(echo "${dataset}" | cut -d. -f1)
        dataset_name=${dataset_name//\//_}
        mkdir -p "results/${model}/${suite}/${dataset_name}"
        python -u run.py \
            --input-file "datasets/${dataset}"  \
            --path "results/${model}/${suite}/${dataset_name}"  \
            --model-name "${model}"  \
            --n-observed-chars "${n_observed_chars}"  \
            --n-beam "${n_beam}"  \
            --batch-size "${batch_size}"  \
            --alpha "${alpha}"  \
            --use-faithfulness-reward | tee "results/${model}/${suite}/${dataset_name}/prediction.log"
        python eval/evaluate.py \
            --gold "datasets/${dataset}"  \
            --hypo "results/${model}/${suite}/${dataset_name}/prediction.txt"  \
            --to_halfwidth  \
            --ignore_unmatch_length  \
            --ignore_space
    done

}
