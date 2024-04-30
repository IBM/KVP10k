import os
from pathlib import Path


class KVPDatasetUtils():

    def get_annotation_file_path(self, image_file_path):
        image_base_name, dataset_root = self.__get_basename_and_dataset_root(image_file_path)
        annotation_file_path = os.path.join(dataset_root, "gts", image_base_name+'.json')
        return annotation_file_path

    def get_tesseract_file_path(self, image_file_path):
        image_base_name, dataset_root = self.__get_basename_and_dataset_root(image_file_path)
        tesseract_file_path = os.path.join(dataset_root, "ocrs", image_base_name+'.json')
        return tesseract_file_path

    def __get_basename_and_dataset_root(self, image_file_path):
        image_file_name = os.path.basename(image_file_path)
        image_base_name = os.path.splitext(image_file_name)[0]
        dataset_root = str(Path(image_file_path).parent.parent)
        return image_base_name, dataset_root