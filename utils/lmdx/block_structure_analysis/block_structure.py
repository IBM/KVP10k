import copy
import os
import json
import numpy as np

from utils.lmdx.block_structure_analysis.core.block_structure_analyzer_iou import BlockStructureAnalyzerIOU


class BlockStructure(object):

    def __init__(self):
        config_path = os.path.dirname(os.path.realpath(__file__)) + '/block_structure_config.json'
        with open(config_path) as components_config:
            self.analyzer_config = json.load(components_config)


    def get_block_structure_info(self, boxes, image_dims):
        """
        Args:
            boxes: list of bounding-boxes items. each item should have the following fields:
                * 'bbox': list of bounding-box coordinates in top-left bottom-right convention [x1, y1, x2, y2]
                * 'word': text within the box
        Returns: doc_text
        """
        boxes, scale_factor = self.__scale_input(boxes, image_dims)

        analyzer = BlockStructureAnalyzerIOU(self.analyzer_config['iou'])
        doc_text = analyzer.analyze(boxes)

        block_structure_info = self.__prepare_output(doc_text, boxes, 1.0 / scale_factor)

        return block_structure_info


    def __prepare_output(self, doc_text, boxes, scale_factor):

        output = {}
        blocks = []
        text_lines = []
        tokens = []

        for box in boxes:
            tokens.append({'bbox': self.__scale_array(box['bbox'], scale_factor), 'text': box['word'] })

        text_lines_counter = 0
        for block_idx, block in enumerate(doc_text):
            for line_idx, line in enumerate(block['lines']):
                text_lines.append({
                    'bbox': self.__scale_array(line['bbox'], scale_factor),
                    'text_line_id': text_lines_counter,
                    'text': line['text'],
                    'block_id': block_idx
                })
                for in_line_idx, token_idx in enumerate(line['idx']):
                    tokens[token_idx]['block_id'] = block_idx
                    tokens[token_idx]['text_line_id'] = text_lines_counter
                    tokens[token_idx]['index_in_line'] = in_line_idx
                text_lines_counter += 1

            blocks.append({
                'bbox': self.__scale_array(block['bbox'], scale_factor),
                'block_id': block_idx
            })
        output['blocks'] = blocks
        output['text_lines'] = text_lines
        output['tokens'] = tokens

        return output

    def __scale_input(self, boxes, image_dims):

        scale_factor = self.__get_scale_factor(image_dims)
        if scale_factor == 1.0:
            return boxes, scale_factor

        new_boxes = copy.deepcopy(boxes)

        if new_boxes is not None:
            for box_i, box in enumerate(new_boxes):
                new_boxes[box_i]['bbox'] = self.__scale_array(box['bbox'], scale_factor)

        return new_boxes, scale_factor


    def __get_scale_factor(self, image_dims):
        target_size = self.analyzer_config['resize_to']
        input_size = min(image_dims)
        scale_factor = target_size / input_size

        return scale_factor

    @staticmethod
    def __scale_array(in_array, scale_factor):
        return list(np.array(in_array) * scale_factor)