#!/bin/bash

mkdir -p datasets

git_raw_url="https://raw.githubusercontent.com"

# Function to download and clean files
clean() {
    local path=$1 file=$2 clean_cmd=$3
    if [ "$(uname)" == "Darwin" ]; then
        sed -i '' "${clean_cmd}" "${path}"/"${file}"
    else
        sed -i "${clean_cmd}" "${path}"/"${file}"
    fi
}

# Check datasets wholeness
all_md5=(
    "cfc5ab4ca6221c2c0dfd2e008db9e5fa  datasets/cscd_ime/test.txt"
    "8ee33a3dda1d6dcd01194648ac64046c  datasets/ecspell/law_500.txt"
    "001247a587a537eae782c35ede44211e  datasets/ecspell/med_500.txt"
    "ddb9a56e9bc65315d445e36cc7cee180  datasets/ecspell/odw_500.txt"
    "a782e9ccb106af4f06f805bfd816322f  datasets/lemon_v2/car.txt"
    "efcc84f12b93a7bbcde61cd2833f7a36  datasets/lemon_v2/cot.txt"
    "e69adf772610b720d2bdca0e544935a4  datasets/lemon_v2/enc.txt"
    "1aa09d22b09e83cbecde7b5d68b7b754  datasets/lemon_v2/gam.txt"
    "5d0f9033f6b61d6dc869048241577566  datasets/lemon_v2/mec.txt"
    "1803c4f9479e7be2c0510103c6d2606a  datasets/lemon_v2/new.txt"
    "118b6b2603f604ae108fe0b065be48c7  datasets/lemon_v2/nov.txt"
    "67a72693ecd9df162a8a83af5cb37e51  datasets/mcsc/test.txt"
    "697efbbd49aa3769c22ecd77ed2d28ef  datasets/sighan_rev/sighan13.txt"
    "cb48500c501fbfd79214f5ea44c4950a  datasets/sighan_rev/sighan14.txt"
    "0c5d1b1799ad33be806d9a3f14903ae8  datasets/sighan_rev/sighan15.txt"
)

any_corrupted=false

for md5 in "${all_md5[@]}"; do
    md5sum=$(echo $md5 | cut -d' ' -f1)
    file=$(echo $md5 | cut -d' ' -f2)
    if [ -f "$file" ]; then
        if [ "$(uname)" == "Darwin" ]; then
            current_md5=$(md5 "$file" | cut -d'=' -f2 | cut -d' ' -f2)
        else
            current_md5=$(md5sum "$file" | cut -d' ' -f1)
        fi
        if [ "$md5sum" != "$current_md5" ]; then
            echo "File $file is corrupted. MD5 checksum does not match."
        fi
    else
        echo "File $file is missing."
        any_corrupted=true
    fi
done

if [ "$any_corrupted" = false ]; then
    echo "All datasets are complete and not corrupted."
    exit 0
fi

# download datasets
## Expected format:
#    <source>\t<target>\n

## ecspell datasets

## path: "datasets/ecspell"
#    - "ecspell/law_500.txt"
#    - "ecspell/med_500.txt"
#    - "ecspell/odw_500.txt"
## format: 
#    <n_error>\t<source>\t<target>\n
## download from:
mkdir -p datasets/ecspell
wget ${git_raw_url}/aopolin-lv/ECSpell/main/Data/domains_data/law.test -O datasets/ecspell/law_500.txt
wget ${git_raw_url}/aopolin-lv/ECSpell/main/Data/domains_data/med.test -O datasets/ecspell/med_500.txt
wget ${git_raw_url}/aopolin-lv/ECSpell/main/Data/domains_data/odw.test -O datasets/ecspell/odw_500.txt
## clean: remove the first column
clean datasets/ecspell law_500.txt "s/^[^$(echo -e '\t')]*$(echo -e '\t')//g"
clean datasets/ecspell med_500.txt "s/^[^$(echo -e '\t')]*$(echo -e '\t')//g"
clean datasets/ecspell odw_500.txt "s/^[^$(echo -e '\t')]*$(echo -e '\t')//g"

## sighan_rev datasets

## path: "datasets/sighan_rev"
#    - "sighan_rev/sighan15.txt"
#    - "sighan_rev/sighan14.txt"
#    - "sighan_rev/sighan13.txt"
## format:
#    <id>\t<source>\t<target>\n
## download from:
mkdir -p datasets/sighan_rev
wget ${git_raw_url}/blcuicall/yacsc/main/SIGHAN-REVISED/test_sighan13.para -O datasets/sighan_rev/sighan13.txt
wget ${git_raw_url}/blcuicall/yacsc/main/SIGHAN-REVISED/test_sighan14.para -O datasets/sighan_rev/sighan14.txt
wget ${git_raw_url}/blcuicall/yacsc/main/SIGHAN-REVISED/test_sighan15.para -O datasets/sighan_rev/sighan15.txt
## clean: remove the first column
clean datasets/sighan_rev sighan13.txt "s/^[^$(echo -e '\t')]*$(echo -e '\t')//g"
clean datasets/sighan_rev sighan14.txt "s/^[^$(echo -e '\t')]*$(echo -e '\t')//g"
clean datasets/sighan_rev sighan15.txt "s/^[^$(echo -e '\t')]*$(echo -e '\t')//g"
## clean: CRLF to LF
clean datasets/sighan_rev sighan13.txt "s/$(echo -e '\r')$//"
clean datasets/sighan_rev sighan14.txt "s/$(echo -e '\r')$//"
clean datasets/sighan_rev sighan15.txt "s/$(echo -e '\r')$//"

## lemon_v2 datasets

## path: "datasets/lemon_v2"
#    - "lemon_v2/car.txt"
#    - "lemon_v2/cot.txt"
#    - "lemon_v2/enc.txt"
#    - "lemon_v2/gam.txt"
#    - "lemon_v2/mec.txt"
#    - "lemon_v2/new.txt"
#    - "lemon_v2/nov.txt"
## format:
#    <source>\t<target>\n      (space separated each character)
## download from:
mkdir -p datasets/lemon_v2
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/car.txt -O datasets/lemon_v2/car.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/cot.txt -O datasets/lemon_v2/cot.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/enc.txt -O datasets/lemon_v2/enc.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/gam.txt -O datasets/lemon_v2/gam.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/mec.txt -O datasets/lemon_v2/mec.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/new.txt -O datasets/lemon_v2/new.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/nov.txt -O datasets/lemon_v2/nov.txt
## clean: remove whitespaces
clean datasets/lemon_v2 car.txt "s/ //g"
clean datasets/lemon_v2 cot.txt "s/ //g"
clean datasets/lemon_v2 enc.txt "s/ //g"
clean datasets/lemon_v2 gam.txt "s/ //g"
clean datasets/lemon_v2 mec.txt "s/ //g"
clean datasets/lemon_v2 new.txt "s/ //g"
clean datasets/lemon_v2 nov.txt "s/ //g"


## cscd_ime datasets

## path: "datasets/cscd_ime"
#    - "cscd_ime/test.txt"
## format:
#    <n_error>\t<source>\t<target>\n
## download from drive google
mkdir -p datasets/cscd_ime
curl -L -o datasets/cscd_ime/test.txt 'https://docs.google.com/uc?export=download&id=1oDf1iZBod9rvk7T3MNU-ILqTE9X5UhGb'
## clean: remove the first column
clean datasets/cscd_ime test.txt "s/^[^$(echo -e '\t')]*$(echo -e '\t')//g"

## mcsc datasets

## path: "datasets/mcsc"
#    - "mcsc/test.txt"
## format:
#    <source>\t<target>\n

mkdir -p datasets/mcsc
wget ${git_raw_url}/yzhihao/MCSCSet/main/data/mcsc_benchmark_dataset/test.txt -O datasets/mcsc/test.txt
## clean: CRLF to LF
clean datasets/mcsc test.txt "s/$(echo -e '\r')$//"

## MD5 checksums
echo "Your MD5 checksums:"
if [ "$(uname)" == "Darwin" ]; then
    md5 datasets/*/*.txt
else
    md5sum datasets/*/*.txt
fi

for md5 in "${all_md5[@]}"; do
    md5sum=$(echo $md5 | cut -d' ' -f1)
    file=$(echo $md5 | cut -d' ' -f2)
    if [ -f "$file" ]; then
        if [ "$(uname)" == "Darwin" ]; then
            current_md5=$(md5 "$file" | cut -d'=' -f2 | cut -d' ' -f2)
        else
            current_md5=$(md5sum "$file" | cut -d' ' -f1)
        fi
        if [ "$md5sum" != "$current_md5" ]; then
            echo "File $file is corrupted. MD5 checksum does not match."
        fi
    else
        echo "File $file is not correctly downloaded."
    fi
done
