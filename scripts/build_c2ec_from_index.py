import argparse
from glob import glob

def build_index(source_file, index_dict):
    with open(source_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            index_dict[(source_file, i)] = line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_files", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    index_dict = {}

    for source_file in args.source_files.split(","):
        for file in glob(source_file):
            build_index(file, index_dict)

    data = []
    with open(args.index_file, "r") as f:
        for line in f:
            file, index = line.strip().split("\t")
            index = int(index)
            if not line:
                continue
            assert (file, index) in index_dict, f"Line {line} not found in index_dict"
            data.append(index_dict[(file, index)])

    with open(args.output_file, "w") as f:
        for item in data:
            f.write(f"{item}\n")
