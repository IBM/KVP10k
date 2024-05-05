import argparse
import logging
import os
import json
from time import time
from utils.tfrq_utils.tfrq import tfrq_generator
import datasets
from utils.huggingface_utils.download_and_prepare_dataset import get_image, get_ocr, get_gt
from utils.huggingface_utils.parquets_to_jsons import parquets_to_jsons

def log_download_statistics(start_time, hf_download_time, parquets_to_jsons_time, preprocess_time, count, error_counter):
    print(f"HF download: {hf_download_time - start_time}")
    print(f"parquets_to_json: {parquets_to_jsons_time - hf_download_time}")
    print(f"preprocess: {preprocess_time - parquets_to_jsons_time}")

    for error in error_counter:
        print(f"{error} : {error_counter[error]}/{count}")


def preprocess_item(item, cache_dir, tessdata_path):
    try:
        image_url, page_number, hash_name = item["image_url"], item["page_number"], item["hash_name"]
        # check if the item was already pre-processed
        if not os.path.exists(f"{cache_dir}/items/{hash_name}.json"):

            image = get_image(image_url, page_number)
            ocr = get_ocr(image, tessdata_path)
            gt = get_gt(image, ocr, item)

            with open(f"{cache_dir}/items/{hash_name}.json", "w") as f:
                json.dump(item, f)
            with open(f"{cache_dir}/ocrs/{hash_name}.json", "w") as f:
                json.dump(ocr, f)
            with open(f"{cache_dir}/gts/{hash_name}.json", "w") as f:
                json.dump(gt, f)
            image.save(f"{cache_dir}/images/{hash_name}.png")
            return None
    except Exception as error:
        logging.error(f"preprocess_item() failed for: {hash_name} with: {type(error)} error: {error}")
        return type(error)

def load_dataset(path, split, cache_dir, tessdata_path):
    start_time = time()
    parquets = datasets.load_dataset(path, split=split)
    hf_download_time = time()

    item_list = parquets_to_jsons(parquets)
    parquets_to_jsons_time = time()

    os.makedirs(cache_dir, exist_ok=True)

    os.makedirs(f"{cache_dir}/items", exist_ok=True)
    os.makedirs(f"{cache_dir}/ocrs", exist_ok=True)
    os.makedirs(f"{cache_dir}/gts", exist_ok=True)
    os.makedirs(f"{cache_dir}/images", exist_ok=True)

    count = 0
    error_counter = {}

    for _result, error in tfrq_generator(preprocess_item, operator='*', params=[
                                                [item,
                                                 cache_dir,
                                                 tessdata_path]
                                                 for item in item_list]):
        if _result != None:
            result = _result[0]
        else:
            result = None
        count+=1
        if result is not None:
            if result not in error_counter:
                error_counter[result] = 1
            else:
                error_counter[result] += 1

    preprocess_time = time()

    log_download_statistics(start_time, hf_download_time, parquets_to_jsons_time, preprocess_time, count, error_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_folder", type=str, help='path to dataset root folder (where the dataset, preprocessed data, will be stored locally)', required=True)
    parser.add_argument("--tessdata_path", type=str, help='path to tessdata folder', required=True)
    args = parser.parse_args()

    os.makedirs(args.dataset_root_folder, exist_ok=True)

    for split in ['train', 'test']:
        load_dataset("ibm/KVP10k", split=split, cache_dir=args.dataset_root_folder + '/' + split, tessdata_path=args.tessdata_path)
