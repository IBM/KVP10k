import textdistance


class MetricsCalculator:
    """
        This class calculates precision, recall, and f1 score metrics for KVP dataset.

        For calculation purposes we take into consideration KVPs location and text.
        GT and prediction KVPs location is considered to be the same if their intersection over union
        is bigger than provided threshold
        GT and prediction KVPs text is considered to be the same if normalized edit distance between them
        is smaller than provided threshold
    """

    DONT_CARE_IOU = -1
    DONT_CARE_EDIT_DISTANCE = 10

    def __init__(self):
        pass

    def calculate_with_text_only(self, dataset: dict, normalized_edit_distance_threshold: float = 0.2, average_per_doc=True) -> dict:
        """
        Calculates precision, recall, and f1 score metrics when KVPs' text only is taken into consideration

        :param dataset: dictionary with two keys 'gt' with the ground truth and 'prediction' with the prediction results
        by a model. Each one is the dictionary with a single key 'kvps_list' containing lsit of KVPs

        :param normalized_edit_distance_threshold: float between 0 and 1. KVPs text is considered to be the same if normalized edit distance
         between them is smaller than provided threshold
        :return dictionary with precision, recall, and f1 score:
        """
        return self.calculate(dataset, normalized_edit_distance_threshold, self.DONT_CARE_IOU, average_per_doc)

    def calculate_with_locations_only(self, dataset: dict, min_iou_4_same_location: float = 0.5, average_per_doc=True) -> dict:
        """
        Calculates precision, recall, and f1 score metrics when KVPs' locations only are taken into consideration

        :param dataset: dictionary with two keys 'gt' with the ground truth and 'prediction' with the prediction results
        by a model. Each one is the dictionary with a single key 'kvps_list' containing lsit of KVPs

        :param min_iou_4_same_location: float between 0 and 1. KVPs location is considered to be the same if
        intersection over union between them is bigger than provided threshold
        :return dictionary with precision, recall, and f1 score:
        """

        return self.calculate(dataset, self.DONT_CARE_EDIT_DISTANCE, min_iou_4_same_location, average_per_doc)

    def calculate(self, dataset:dict, normalized_edit_distance_threshold: float = 0.2,
                  min_iou_4_same_location: float = 0.5, average_per_doc=True) -> dict:
        """
        Calculates precision, recall, and f1 score metrics when both KVPs' locations and their text
         are taken into consideration

        :param dataset: dictionary with two keys 'gt' with the ground truth and 'prediction' with the prediction results
        by a model. Each one is the dictionary with a single key 'kvps_list' containing lsit of KVPs
        :param normalized_edit_distance_threshold: float between 0 and 1. KVPs text is considered to be the same if normalized edit distance
         between them is smaller than provided threshold
        :param min_iou_4_same_location: float between 0 and 1. KVPs location is considered to be the same if
        intersection over union between them is bigger than provided threshold
        :return dictionary with precision, recall, and f1 score:
        """
        classification_per_file = []
        for file_2_evaluate in dataset['files']:
            if file_2_evaluate['gt']['kvps_list']:
                file_classification = self._calculate_classification(file_2_evaluate, normalized_edit_distance_threshold, min_iou_4_same_location)
                classification_per_file.append(file_classification)

        tp_count_list, fp_count_list, fn_count_list = zip(*classification_per_file)
        if average_per_doc:
            return self._calculate_average_per_file(tp_count_list, fp_count_list, fn_count_list)
        else:
            return self._calculate_overall_average(tp_count_list, fp_count_list, fn_count_list)

    def _calculate_average_per_file(self, tp_count_list, fp_count_list, fn_count_list):

        metrics_per_file = []
        for tp_count, fp_count, fn_count in zip(tp_count_list, fp_count_list, fn_count_list):
            precision, recall, f1 = self._calculate_basic_metrics(tp_count, fp_count, fn_count)
            metrics_per_file.append((precision, recall, f1))

        transposed_metrics = list(zip(*metrics_per_file))

        # Calculate average of each value in tuple
        precision = sum(transposed_metrics[0]) / len(transposed_metrics[0])
        recall = sum(transposed_metrics[1]) / len(transposed_metrics[1])
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _calculate_overall_average(self, tp_count_list, fp_count_list, fn_count_list):
        total_tp_count = sum(tp_count_list)
        total_fp_count = sum(fp_count_list)
        total_fn_count = sum(fn_count_list)

        precision, recall, f1 = self._calculate_basic_metrics(total_tp_count, total_fp_count, total_fn_count)

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _calculate_classification(self, file_2_evaluate: dict, normalized_edit_distance_threshold: float,
                                  min_iou_4_same_location: float) -> tuple:
        gt_dict = file_2_evaluate['gt']
        prediction_dict = file_2_evaluate['prediction']

        self._reset_matches(gt_dict['kvps_list'])
        self._reset_matches(prediction_dict['kvps_list'])

        tp_count = 0
        for prediction_kvp in prediction_dict['kvps_list']:
            if self._has_match_in_gt(prediction_kvp, gt_dict, normalized_edit_distance_threshold, min_iou_4_same_location):
                tp_count += 1

        fp_count = sum(1 for d in prediction_dict['kvps_list'] if d['is_matched'] is False)
        fn_count = sum(1 for d in gt_dict['kvps_list'] if d['is_matched'] is False)
        return tp_count, fp_count, fn_count


    def _has_match_in_gt(self, prediction_kvp: dict, gt_dict: dict, normalized_edit_distance_threshold: float,
                         min_iou_4_same_location: float) -> bool:
        for gt_kvp in gt_dict['kvps_list']:
            if gt_kvp['is_matched']:
                continue

            if self._is_matched(gt_kvp, prediction_kvp, normalized_edit_distance_threshold, min_iou_4_same_location):
                gt_kvp['is_matched'] = True
                prediction_kvp['is_matched'] = True
                return True

        return False

    def _reset_matches(self, kvps_list: list) -> None:
        for kvp in kvps_list:
            kvp['is_matched'] = False

    def _is_matched(self, gt_kvp: dict, prediction_kvp: dict, normalized_edit_distance_threshold: float,
                    min_iou_4_same_location: float) -> bool:
        if gt_kvp['type'] != prediction_kvp['type']:
            return False

        return self._has_same_location(gt_kvp, prediction_kvp, min_iou_4_same_location) and \
               self._has_same_text(gt_kvp, prediction_kvp, normalized_edit_distance_threshold)

    def _has_same_text(self, gt_kvp: dict, prediction_kvp: dict, normalized_edit_distance_threshold: float) -> bool:
        if normalized_edit_distance_threshold >= 2:
            return True

        if gt_kvp['type'] != prediction_kvp['type']:
            return False

        if gt_kvp['type'] == 'unkeyed':
            return self._words_have_same_text(gt_kvp['value']['text'], prediction_kvp['value']['text'], normalized_edit_distance_threshold)

        if gt_kvp['type'] == 'unvalued':
            return self._words_have_same_text(gt_kvp['key']['text'], prediction_kvp['key']['text'], normalized_edit_distance_threshold)

        # Regular
        return self._words_have_same_text(gt_kvp['key']['text'], prediction_kvp['key']['text'], normalized_edit_distance_threshold) and \
               self._words_have_same_text(gt_kvp['value']['text'], prediction_kvp['value']['text'], normalized_edit_distance_threshold)

    def _words_have_same_text(self, word1: str, word2: str, normalized_edit_distance_threshold: float) -> bool:
        max_word_length = max(len(word1), len(word2))
        distance = textdistance.Levenshtein().distance(word1, word2)
        if max_word_length == 0:
            normalized_distance = 0
        else:
            normalized_distance = distance / max_word_length

        return normalized_distance < normalized_edit_distance_threshold

    def _has_same_location(self, gt_kvp: dict, prediction_kvp: dict, min_iou_4_same_location: float) -> bool:
        if gt_kvp['type'] != prediction_kvp['type']:
            return False

        if gt_kvp['type'] == 'unkeyed':
            return self._boxes_have_same_location(gt_kvp['value']['bbox'], prediction_kvp['value']['bbox'], min_iou_4_same_location)

        if gt_kvp['type'] == 'unvalued':
            return self._boxes_have_same_location(gt_kvp['key']['bbox'], prediction_kvp['key']['bbox'], min_iou_4_same_location)

        # Regular
        return self._boxes_have_same_location(gt_kvp['key']['bbox'], prediction_kvp['key']['bbox'], min_iou_4_same_location) and \
               self._boxes_have_same_location(gt_kvp['value']['bbox'], prediction_kvp['value']['bbox'], min_iou_4_same_location)

    def _boxes_have_same_location(self, box1: list, box2: list, min_iou_4_same_location: float) -> bool:
        return self._get_intersection_over_union(box1, box2) >= min_iou_4_same_location

    def _get_intersection_area(self, bbox1: list, bbox2: list) -> float:
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        overlap_area = x_overlap * y_overlap
        return overlap_area

    def _get_union_area(self, bbox1: list, bbox2: list) -> float:
        bbox1_width = bbox1[2] - bbox1[0]
        bbox1_height = bbox1[3] - bbox1[1]
        bbox2_width = bbox2[2] - bbox2[0]
        bbox2_height = bbox2[3] - bbox2[1]

        union_area = bbox1_width * bbox1_height + bbox2_width * bbox2_height - self._get_intersection_area(bbox1, bbox2)
        return union_area

    def _get_intersection_over_union(self, bbox1: list, bbox2: list) -> float:
        union = self._get_union_area(bbox1, bbox2)
        if union == 0:
            print('Internal bug in union calculation, never should be 0s')
            return 0
        return self._get_intersection_area(bbox1, bbox2) / union

    def _calculate_basic_metrics(self, tp: int, fp: int, fn: int) -> tuple:
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
        return precision, recall, f1


