import os
import torch
import lightning.pytorch as pl

from utils.config import Config
from model import Model
from utils.response_handler import ResponseHandler
import argparse
from utils.utils import get_tesseract_text_from_file

torch.set_float32_matmul_precision('medium')


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/kvp_inference.yaml', help='path to inference config file')
    parser.add_argument('--save_path', type=str, required=True, default=None, help='path where to save the inference output (JSON)')
    parser.add_argument('--pretrained', type=str, required=True, default=None, help='path to the pretrained model')
    parser.add_argument("--cache_dir", type=str, required=False, default='/tmp', help='path to transformer cache_dir')
    parser.add_argument('--test_folder', type=str, required=True, default=None,  help='path to test data folder')

    input_args = parser.parse_args()
    args = Config(input_args.config)
    args.save_path = input_args.save_path
    args.pretrained = input_args.pretrained
    args.tokenizer_cache_dir = input_args.cache_dir
    args.llm_cache_dir = input_args.cache_dir
    args.test_folder = input_args.test_folder

    # print arguments
    print(vars(args))

    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # seed
    pl.seed_everything(0)

    # model
    model = Model(args)
    model = model.eval()

    # when device_map is 'auto', running model.cuda() creates a problem
    if args.llm_device_map != 'auto' and torch.cuda.is_available():
        model = model.cuda()

    images_test_folder = args.test_folder + '/images'

    img_paths = [os.path.join(images_test_folder, x) for x in os.listdir(images_test_folder)]

    for i, img_path in enumerate(img_paths):
        image_size = (1, 1)
        try:
            print(f'Image index = {i}, image path = {img_path}')
            print('Current image: {}'.format(img_path))
            # load image and context
            context, image_size = get_tesseract_text_from_file(img_path)
            print('*' * 10, 'context:')

            # get response
            print(f'args.inference_text_prompt {args.inference_text_prompt}')
            print(f'context =  {context}')
            response = model(prompt=args.inference_text_prompt, context=context)
            print(f'Response = {response}')
        except Exception:
            print('Failed to process image producing empty response ', img_path)
            response = ''

        response_handler = ResponseHandler()
        if not response:
            response = '[]'
            image_size = (1, 1)

        image_basename= os.path.splitext(os.path.basename(img_path))[0]

        response = response_handler.convert_2_common_gt(response, image_size)
        json_path = os.path.join(args.save_path, image_basename + '.json')

        # save response to file
        if response is not None:
            with open(json_path, 'w') as outfile:
                outfile.write(response)