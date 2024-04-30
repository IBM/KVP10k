import argparse
from benchmark.dataset_loader import DatasetLoader
from benchmark.metrics_calculator import MetricsCalculator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, required=True, help='path to ground_truth folder')
    parser.add_argument('--inference_folder', type=str, required=True, help='path to inference folder')
    args = parser.parse_args()

    common_gt_folder = args.gt_folder
    model_folder = args.inference_folder

    dataset_loader = DatasetLoader()
    dataset = dataset_loader.load(gt_folder=common_gt_folder, prediction_folder=model_folder)
    metrics_calculator = MetricsCalculator()
    results = metrics_calculator.calculate(dataset, min_iou_4_same_location=0.3, average_per_doc=True)
    print('All - ', results)
    results = metrics_calculator.calculate_with_text_only(dataset, average_per_doc=True)
    print('Text only - ', results)
    results = metrics_calculator.calculate_with_locations_only(dataset, min_iou_4_same_location=0.3, average_per_doc=True)
    print('Location only - ', results)

