import argparse

from utils.annotation_utils.annotation_visualizer import visualize_annotation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, help='path to image file', required=True)
    parser.add_argument("--annotation_file", type=str, help='path to annotation file', required=True)
    args = parser.parse_args()

    img = visualize_annotation(args.image_file, args.annotation_file)
    img.show()