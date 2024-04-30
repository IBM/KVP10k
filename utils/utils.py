import json

from utils.dataset_utils import KVPDatasetUtils
from utils.lmdx.lmdx_utils import LmdxUtils

def get_tesseract_text_from_file(image_file_path):
    print('Getting text tesseract ')
    dataset_utils = KVPDatasetUtils()
    tesseract_file_path = dataset_utils.get_tesseract_file_path(image_file_path)
    with open(tesseract_file_path, encoding='utf-8') as tesseract_file:
        tesseract_dict = json.load(tesseract_file)

    page_0 = tesseract_dict['pages'][0]
    image_size = (page_0['width'], page_0['height'])

    lmdx_utils = LmdxUtils()
    lmdx_sentences_holder = lmdx_utils.get_lmdx_sentences_holder(tesseract_dict)
    result = lmdx_sentences_holder.get_prompt(use_center_coordinate_only=False)
    print('After get_lmdx_prompt')
    return result, image_size