import argparse
import pandas as pd
import json
import os
from tqdm import tqdm

def convert_json_to_parquet(json_files, output_parquet_path):
    dfs = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df.to_parquet(output_parquet_path, index=False)


def main(input_directory, output_directory):
    for kind in ["train", "test"]:
        os.makedirs(f"{output_directory}/{kind}", exist_ok=True)
        all_json_files = []
        if kind=="test":
            all_json_files = [[f"{input_directory}/{kind}/{json_file}" for json_file in os.listdir(f"{input_directory}/{kind}")]]
        else:
            json_files = []
            for json_file in os.listdir(f"{input_directory}/{kind}"):
                json_files.append(f"{input_directory}/{kind}/{json_file}")
                if len(json_files)==2000:
                    all_json_files.append(json_files)
                    json_files = []
            if len(json_files)>0:
                all_json_files.append(json_files)

        for ind, json_files in tqdm(enumerate(all_json_files)):
            output_parquet_path = f"{output_directory}/{kind}/{ind}.parquet"
            convert_json_to_parquet(json_files, output_parquet_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, help='path to input directory with subfolders test and train', required=True)
    parser.add_argument("--output_directory", type=str, help='path to output directory (where the parquet will be stored locally)', required=True)
    args = parser.parse_args()

    main(args.input_directory, args.output_directory)

