import glob
import json
import os


class DatasetLoader:
    """
        This class loads KVP dataset from provided GT and prediction folders.

        The data should be arranged as json files in corresponding folders, when base name should be the same
        For instance if we use suffix 'gt' for ground truth and 'model' for predictions, File names will be as following
        filename.gt.json - for file in ground truth folder
        filename.model.json - for the corresponding file in prediction folder
    """

    def __init__(self):
        pass

    def load(self, gt_folder, prediction_folder):
        files_list = []
        dataset = {'files':files_list}

        files_to_process = glob.glob(gt_folder + "/*.json")
        for index, gt_file_path in enumerate(files_to_process):
            dataset_file = {}
            prediction_file_path = self._get_corresponding_prediction_file(gt_file_path, prediction_folder)
            with open(gt_file_path, encoding='utf-8') as gt_file:
                gt_dict = json.load(gt_file)

            with open(prediction_file_path, encoding='utf-8') as prediction_file:
                prediction_dict = json.load(prediction_file)

            dataset_file['gt'] = gt_dict
            dataset_file['prediction'] = prediction_dict
            files_list.append(dataset_file)

        return dataset

    def _get_corresponding_prediction_file(self, gt_file_path, prediction_folder):
        gt_file_name = os.path.basename(gt_file_path)
        base_name = os.path.splitext(gt_file_name)[0]
        prediction_file_path = os.path.join(prediction_folder, base_name + '.json')
        return prediction_file_path

    def filter_by_kvp_type(self, dataset, kvp_type):
        filtered_files = []
        for dataset_file in dataset['files']:
            gt_kvps_list = [d for d in dataset_file['gt']['kvps_list'] if d['type'] == kvp_type]
            gt = {'kvps_list': gt_kvps_list}

            prediction_kvps_list = [d for d in dataset_file['prediction']['kvps_list'] if d['type'] == kvp_type]
            prediction = {'kvps_list': prediction_kvps_list}

            filtered_file = {'gt' : gt, 'prediction' : prediction}
            filtered_files.append(filtered_file)

        filtered_dataset = {'files':filtered_files}
        return filtered_dataset
