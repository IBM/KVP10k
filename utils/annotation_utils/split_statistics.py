import os

from utils.annotation_utils.annotation_reader import annotation_reader

IMAGES = "images"
ANNOTATION = "items"

class split_statistics:
    def __init__(self, dataset_root_folder, splits):
        self.dataset_root_folder = dataset_root_folder
        self.splits = splits
        pass

    def count_images(self):
        total_images = 0
        for split in self.splits:
            split_images_path = os.path.join(self.dataset_root_folder, split, IMAGES)
            total_images += sum(
                1 for entry in os.listdir(split_images_path) if entry != ".DS_Store")
        return total_images

    def count_annotation(self):
        total_annotation = 0
        for split in self.splits:
            split_annotation_path = os.path.join(self.dataset_root_folder, split, ANNOTATION)
            total_annotation += sum(
                1 for entry in os.listdir(split_annotation_path) if entry != ".DS_Store")
        return total_annotation

    def count_empty_annotation(self):
        total_empty_annotation = 0
        for split in self.splits:
            split_annotation_path = os.path.join(self.dataset_root_folder, split, ANNOTATION)
            split_file_list = os.listdir(split_annotation_path)
            for annotation_file in split_file_list:
                if annotation_file == ".DS_Store":
                    continue
                annotation_json_path = os.path.join(split_annotation_path, annotation_file)
                t_reader = annotation_reader(annotation_json_path)
                if len(t_reader.rectangles) == 0:
                    total_empty_annotation += 1
        return total_empty_annotation

    def count_unkeyed_value(self):
        total_unkeyed_value = 0
        for split in self.splits:
            split_annotation_path = os.path.join(self.dataset_root_folder, split, ANNOTATION)
            annotation_file_list = os.listdir(split_annotation_path)
            for annotation_file in annotation_file_list:
                if annotation_file == ".DS_Store":
                    continue
                annotation_json_path = os.path.join(split_annotation_path, annotation_file)
                t_reader = annotation_reader(annotation_json_path)
                total_unkeyed_value += t_reader.count_unkeyed_value()

            return total_unkeyed_value

    def count_unvalued_key(self):
        total_unvalued_key = 0
        for split in self.splits:
            split_annotation_path = os.path.join(self.dataset_root_folder, split, ANNOTATION)
            annotation_file_list = os.listdir(split_annotation_path)
            for annotation_file in annotation_file_list:
                if annotation_file == ".DS_Store":
                    continue
                annotation_json_path = os.path.join(split_annotation_path, annotation_file)
                t_reader = annotation_reader(annotation_json_path)
                total_unvalued_key += t_reader.count_unvalued_key()

            return total_unvalued_key

    def count_flat_kvp(self):
        flat_kvp = 0
        for split in self.splits:
            split_annotation_path = os.path.join(self.dataset_root_folder, split, ANNOTATION)
            annotation_file_list = os.listdir(split_annotation_path)
            for annotation_file in annotation_file_list:
                if annotation_file == ".DS_Store":
                    continue
                annotation_json_path = os.path.join(split_annotation_path, annotation_file)
                t_reader = annotation_reader(annotation_json_path)
                flat_kvp += t_reader.count_flat_kvp()

            return flat_kvp