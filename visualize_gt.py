import argparse

from utils.gt_utils.gt_visualizer import visualize_gt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, help='path to image file', required=True)
    parser.add_argument("--gt_file", type=str, help='path to ground_truth file', required=True)
    args = parser.parse_args()

    img = visualize_gt(args.image_file, args.gt_file)
    img.show()