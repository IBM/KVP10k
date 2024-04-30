import glob
import json
import time
import os
from torch.utils.data import Dataset

from utils.dataset_utils import KVPDatasetUtils
from utils.lmdx.lmdx_utils import LmdxUtils
from utils.prompt_utils import PromptUtils

class KVPDataset(Dataset):

    def __init__(self, dataset_path, max_length=None, **kwargs):
        self.lmdx_utils = LmdxUtils()
        self.dataset_utils = KVPDatasetUtils()
        self.dataset_path = dataset_path
        self.prompt = 'tess_common_gt'
        self.max_length = max_length

        images_list = self.__get_images_list()
        print('Found ', len(images_list), ' images in data set')
        self.files_in_dataset = self.__filter_out_non_existing_annotations(images_list)
        print('Found ', len(self.files_in_dataset),' files with matching annotations in dataset')
        self.files_in_dataset = self.__filter_dataset_by_tokens_quantity(self.files_in_dataset)

    def __get_images_list(self):
        images_folder = os.path.join(self.dataset_path.strip(), 'images')
        images_list = glob.glob(images_folder + "/*.png")

        return images_list

    def __filter_out_non_existing_annotations(self, image_files):
        filtered_files = []
        for image_file_path in image_files:
            annotation_file_path = self.dataset_utils.get_annotation_file_path(image_file_path)
            ocr_file_path = self.dataset_utils.get_tesseract_file_path(image_file_path)

            if os.path.exists(annotation_file_path) and os.path.exists(ocr_file_path):
                filtered_files.append(image_file_path)

        return filtered_files

    def __filter_dataset_by_tokens_quantity(self, files_2_filter):
        print('Filtering dataset with ', len(files_2_filter), ' items')
        filtering_start = time.time()
        filtered_files = []
        prompt_utils = PromptUtils('mistral')
        for index, image_file_path in enumerate(files_2_filter):
            try:
                annotation_file_path = self.dataset_utils.get_annotation_file_path(image_file_path)
                with open(annotation_file_path, encoding='utf-8') as annotation_file:
                    annotation_dict = json.load(annotation_file)

                tesseract_file_path = self.dataset_utils.get_tesseract_file_path(image_file_path)
                with open(tesseract_file_path, encoding='utf-8') as tesseract_file:
                    tesseract_dict = json.load(tesseract_file)

                res = self.get_item_from_dataset(annotation_dict, tesseract_dict)

                prompt = prompt_utils.make_prompt('tess_common_gt', res['context'], res['target'])
                token_num = prompt_utils.get_num_of_tokens(prompt)
                if token_num < 7040:
                    filtered_files.append(image_file_path)
            except Exception as e:
                # Failed to parse skipping
                pass
        print('Filtered dataset has ', len(filtered_files),' items')
        filtering_end = time.time()
        print('Filtering took ',(filtering_end-filtering_start),' sec')
        return filtered_files

    def get_item_from_dataset(self, annotation_dict, tesseract_dict):
        lmdx_sentences_holder = self.lmdx_utils.get_lmdx_sentences_holder(tesseract_dict)
        context = lmdx_sentences_holder.get_prompt(use_center_coordinate_only=False)
        gt_dict = self.__get_flat_kvps_gt(annotation_dict, lmdx_sentences_holder)
        gt_string = json.dumps(gt_dict)

        res = {
            'context': context,
            'target': gt_string,
            'prompt': self.prompt
        }

        return res

    def __len__(self):
        if self.max_length is None:
            return len(self.files_in_dataset)
        else:
            return min(self.max_length, len(self.files_in_dataset))


    def __getitem__(self, idx):
        return self.get_item_by_index(idx)

    def get_item_by_index(self, idx):
        image_file_path = self.files_in_dataset[idx]

        annotation_file_path = self.dataset_utils.get_annotation_file_path(image_file_path)
        with open(annotation_file_path, encoding='utf-8') as annotation_file:
            annotation_dict = json.load(annotation_file)

        tesseract_file_path = self.dataset_utils.get_tesseract_file_path(image_file_path)
        with open(tesseract_file_path, encoding='utf-8') as tesseract_file:
            tesseract_dict = json.load(tesseract_file)

        return self.get_item_from_dataset(annotation_dict, tesseract_dict)

    def __get_flat_kvps_gt(self, annotation_dict, lmdx_sentences_holder):
        flat_kvps_gt = {}
        kvp_list = []

        for gt_kvp in annotation_dict['kvps_list']:
            if gt_kvp['type'] == 'unkeyed':
                key_center_x, key_center_y, key_left, key_top, key_right, key_bottom = -1, -1, -1, -1, -1, -1
                key_word = 'implicit '+gt_kvp['key']['text']
            else:
                quantized_key_bbox = lmdx_sentences_holder.quantize_bbox(gt_kvp['key']['bbox'])
                key_center_x, key_center_y = lmdx_sentences_holder.get_box_center(quantized_key_bbox)
                key_left, key_top, key_right, key_bottom = quantized_key_bbox[0], quantized_key_bbox[1], \
                                                           quantized_key_bbox[2],quantized_key_bbox[3]

                key_word = gt_kvp['key']['text']
                if not key_word.strip():
                    #filtering invalid input
                    continue

            key_text = key_word + ' ' + str(key_left) + '|' + str(key_top)+'|' + str(key_right) + '|' + str(key_bottom)


            if gt_kvp['type'] == 'unvalued':
                value_center_x, value_center_y, value_left, value_top, value_right, value_bottom = -1, -1, -1, -1, -1, -1
                value_word = 'not presented'
            else:
                quantized_value_bbox = lmdx_sentences_holder.quantize_bbox(gt_kvp['value']['bbox'])
                value_center_x, value_center_y = lmdx_sentences_holder.get_box_center(quantized_value_bbox)
                value_left, value_top, value_right, value_bottom = quantized_value_bbox[0], quantized_value_bbox[1], \
                                                           quantized_value_bbox[2],quantized_value_bbox[3]

                value_word = gt_kvp['value']['text']
                if not value_word.strip():
                    #filtering invalid input
                    continue


            value_text = value_word + ' ' + str(value_left) + '|' + str(value_top) + '|' + str(value_right) + '|' + str(value_bottom)

            if gt_kvp['type'] == 'unkeyed':
                kvp_list.append((key_text, value_text, value_center_x, value_center_y))
            else:
                kvp_list.append((key_text, value_text, key_center_x, key_center_y))

        sorted_kvp_list = sorted(kvp_list, key=lambda item: (item[3], item[2]))

        sorted_kvp_list = [(item[0], item[1]) for item in sorted_kvp_list]
        flat_kvps_gt['kvps'] = sorted_kvp_list
        # return flat_kvps_gt
        return sorted_kvp_list