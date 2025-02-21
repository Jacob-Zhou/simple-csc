#!/bin/bash

mkdir -p datasets

git_raw_url="https://raw.githubusercontent.com"
c2ec_dir="c2ec"

# Function to download and clean files
clean() {
    local path=$1 file=$2 clean_cmd=$3
    if [ "$(uname)" == "Darwin" ]; then
        sed -i '' "${clean_cmd}" "${path}"/"${file}"
    else
        sed -i "${clean_cmd}" "${path}"/"${file}"
    fi
}

c2ec_md5=(
    "e333cb9dd2a3f38ab6e93cd0518e5dd8  datasets/${c2ec_dir}/dev.txt"
    "8439e41b7ac7a5bd1c38d090f925f2db  datasets/${c2ec_dir}/test.txt"
)

any_corrupted=false

for md5 in "${c2ec_md5[@]}"; do
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
        echo "File $file is yet to be built."
        any_corrupted=true
    fi
done

if [ "$any_corrupted" = false ]; then
    echo "C2EC Dataset is correctly built."
    exit 0
fi

## CCTC datasets

## path: "datasets/cctc"
#    - "cctc/train.txt"
#    - "cctc/test.txt"
## format:
# cctc_v1.1.zip
## download from:
mkdir -p datasets/${c2ec_dir}/original_datasets/
# check if the file exists
if [ ! -f "datasets/${c2ec_dir}/cctc_v1.1.zip" ]; then
    echo "File datasets/${c2ec_dir}/cctc_v1.1.zip does not exist."
    exit 1
fi
# check if md5sum matches
if [ "$(uname)" == "Darwin" ]; then
    current_md5=$(md5 "datasets/${c2ec_dir}/cctc_v1.1.zip" | cut -d'=' -f2 | cut -d' ' -f2)
else
    current_md5=$(md5sum "datasets/${c2ec_dir}/cctc_v1.1.zip" | cut -d' ' -f1)
fi
if [ "$current_md5" != "ecd94ad85c33d7c0ace11b6da316f81e" ]; then
    echo "File datasets/${c2ec_dir}/cctc_v1.1.zip is corrupted. MD5 checksum does not match."
    exit 1
fi

# unzip with -o to overwrite the existing files
unzip -o datasets/${c2ec_dir}/cctc_v1.1.zip -d datasets/${c2ec_dir}/original_datasets/
# we will get the following files:
# cctc_v1.1/
# - cctc_test_wide.json 
# - cctc_train.json
# - README.md

python scripts/extract_cctc.py --input_file datasets/${c2ec_dir}/original_datasets/cctc_v1.1/cctc_test_wide.json --output_file datasets/${c2ec_dir}/original_datasets/cctc_v1.1/test.txt
python scripts/extract_cctc.py --input_file datasets/${c2ec_dir}/original_datasets/cctc_v1.1/cctc_train.json --output_file datasets/${c2ec_dir}/original_datasets/cctc_v1.1/train.txt

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
mkdir -p datasets/${c2ec_dir}/original_datasets/lemon_v2
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/car.txt -O datasets/${c2ec_dir}/original_datasets/lemon_v2/car.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/cot.txt -O datasets/${c2ec_dir}/original_datasets/lemon_v2/cot.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/enc.txt -O datasets/${c2ec_dir}/original_datasets/lemon_v2/enc.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/gam.txt -O datasets/${c2ec_dir}/original_datasets/lemon_v2/gam.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/mec.txt -O datasets/${c2ec_dir}/original_datasets/lemon_v2/mec.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/new.txt -O datasets/${c2ec_dir}/original_datasets/lemon_v2/new.txt
wget ${git_raw_url}/gingasan/lemon/main/lemon_v2/nov.txt -O datasets/${c2ec_dir}/original_datasets/lemon_v2/nov.txt
## clean: remove whitespaces
clean datasets/${c2ec_dir}/original_datasets/lemon_v2 car.txt "s/ //g"
clean datasets/${c2ec_dir}/original_datasets/lemon_v2 cot.txt "s/ //g"
clean datasets/${c2ec_dir}/original_datasets/lemon_v2 enc.txt "s/ //g"
clean datasets/${c2ec_dir}/original_datasets/lemon_v2 gam.txt "s/ //g"
clean datasets/${c2ec_dir}/original_datasets/lemon_v2 mec.txt "s/ //g"
clean datasets/${c2ec_dir}/original_datasets/lemon_v2 new.txt "s/ //g"
clean datasets/${c2ec_dir}/original_datasets/lemon_v2 nov.txt "s/ //g"

## MD5 checksums
echo "Checking Original Datasets MD5 checksums:"
if [ "$(uname)" == "Darwin" ]; then
    md5 datasets/${c2ec_dir}/original_datasets/*/*.txt
else
    md5sum datasets/${c2ec_dir}/original_datasets/*/*.txt
fi

# Check datasets wholeness
all_md5=(
    "e0e1a94b74d8bcdd1ed0f6cb6a3d9183  datasets/${c2ec_dir}/original_datasets/cctc_v1.1/test.txt"
    "5a8198f8a63adc3aaacfec36cb8488e3  datasets/${c2ec_dir}/original_datasets/cctc_v1.1/train.txt"
    "a782e9ccb106af4f06f805bfd816322f  datasets/${c2ec_dir}/original_datasets/lemon_v2/car.txt"
    "efcc84f12b93a7bbcde61cd2833f7a36  datasets/${c2ec_dir}/original_datasets/lemon_v2/cot.txt"
    "e69adf772610b720d2bdca0e544935a4  datasets/${c2ec_dir}/original_datasets/lemon_v2/enc.txt"
    "1aa09d22b09e83cbecde7b5d68b7b754  datasets/${c2ec_dir}/original_datasets/lemon_v2/gam.txt"
    "5d0f9033f6b61d6dc869048241577566  datasets/${c2ec_dir}/original_datasets/lemon_v2/mec.txt"
    "1803c4f9479e7be2c0510103c6d2606a  datasets/${c2ec_dir}/original_datasets/lemon_v2/new.txt"
    "118b6b2603f604ae108fe0b065be48c7  datasets/${c2ec_dir}/original_datasets/lemon_v2/nov.txt"
)

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
        echo "File $file is not correctly downloaded. Please try again."
        exit 1
    fi
done

now_dir=$(pwd)
cd datasets/${c2ec_dir}
mkdir -p metadata
python ${now_dir}/scripts/build_c2ec_from_index.py --source_files "original_datasets/cctc_v1.1/train.txt" --index_file "metadata/dev.index" --output_file dev.txt
python ${now_dir}/scripts/build_c2ec_from_index.py --source_files "original_datasets/cctc_v1.1/test.txt,original_datasets/lemon_v2/*.txt" --index_file "metadata/test.index" --output_file test.txt
cd ${now_dir}

echo "Build C2EC Dataset MD5 checksums:"
if [ "$(uname)" == "Darwin" ]; then
    md5 datasets/${c2ec_dir}/*.txt
else
    md5sum datasets/${c2ec_dir}/*.txt
fi

any_corrupted=false

for md5 in "${c2ec_md5[@]}"; do
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
            any_corrupted=true
        fi
    else
        echo "File $file is not correctly built."
        any_corrupted=true
    fi
done

if [ "$any_corrupted" = false ]; then
    echo "C2EC Dataset is correctly built."
else
    echo "Build C2EC Dataset Failed."
fi
