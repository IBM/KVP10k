from utils.lmdx.block_structure_analysis.block_structure import BlockStructure
from utils.lmdx.block_structure_analysis.core.block_structure_analyzer_iou import BlockStructureAnalyzerIOU
from utils.lmdx.lmdx_sentences_holder import LMDXSentencesHolder
import numpy as np


class LMDXCreator():

    SENTENCE_DELIMETERS = ['.', ';', '-']
    CHARS_TO_SPLIT_SENTENCE = 2.5
    A4_RATIO = (210,297)

    def __init__(self):
        self.blocks_analyzer = BlockStructure()
        # self.quantized_width = 400
        # self.quantized_height = round(self.quantized_width / self.A4_RATIO[0] * self.A4_RATIO[1])
        self.quantized_width = 100
        self.quantized_height = 100


    def create_sentences_from_ocr(self, ocr_dict)->LMDXSentencesHolder:
        page_0 = ocr_dict['pages'][0]

        image_size = (page_0['width'], page_0['height'])

        block_structure_info = self.create_blocks_info_from_ocr(ocr_dict)
        lines_info = self.get_lines(block_structure_info)
        sentences_info =  self.get_sentences(lines_info, image_size)
        return LMDXSentencesHolder(sentences_info, image_size, self.quantized_width, self.quantized_height)

    def create_blocks_info_from_ocr(self, ocr_dict):
        page_0 = ocr_dict['pages'][0]
        image_shape = (page_0['width'], page_0['height'])
        words = page_0['words']
        boxes_info_list = self.__get_boxes_info_list(words)
        block_structure_info = self.blocks_analyzer.get_block_structure_info(boxes_info_list, image_dims=image_shape)
        return block_structure_info

    def get_lines(self, block_structure_info):
        lines_list = []
        tokens = block_structure_info['tokens']
        max_lines_id = max(tokens, key=lambda x: x['text_line_id'])['text_line_id']
        for line_index in range(max_lines_id+1):
            line_boxes_list = list(filter(lambda x: x['text_line_id'] == line_index, tokens))
            sorted_boxes = sorted(line_boxes_list, key=lambda box_info: box_info['index_in_line'])
            lines_list.append(sorted_boxes)
        return lines_list

    def get_sentences(self, lines_info_list, image_size):
        sentences = []
        width_ratio = self.quantized_width / image_size[0]
        height_ratio = self.quantized_height / image_size[1]

        for line_info in lines_info_list:
            line_sentences = self.__split_line_2_sentences(line_info, width_ratio, height_ratio)
            sentences.extend(line_sentences)

        return sentences

    def __get_boxes_info_list(self, ocr_words):
        boxes_info_list = []
        for word in ocr_words:
            box_info = {'word': word['text'], 'bbox': word['bbox']}
            boxes_info_list.append(box_info)

        return boxes_info_list

    def __split_line_2_sentences(self, line_info, width_ratio, height_ratio):
        sentences = []
        if not line_info:
            return sentences

        curr_sentence = {'words':[]}
        line_height = self.__get_median_height(line_info)
        max_allowed_distance = line_height * self.CHARS_TO_SPLIT_SENTENCE / 2

        for index, box_info in enumerate(line_info):
            curr_sentence['words'].append(box_info)
            if index == len(line_info) - 1:
                distance_2_next_word = 0
            else:
                distance_2_next_word = line_info[index + 1]['bbox'][0]  - box_info['bbox'][2]

            if self.contains_char(box_info['text'], self.SENTENCE_DELIMETERS) or distance_2_next_word > max_allowed_distance:
                sentences.append(curr_sentence)
                curr_sentence = {'words': []}

        if curr_sentence['words']:
            sentences.append(curr_sentence)

        self.__add_general_sentence_info(sentences, width_ratio, height_ratio)
        return sentences

    def __get_median_height(self, boxes_info_list):
        heights_list = []
        for box_info in boxes_info_list:
            box_info['box_height'] = abs(box_info['bbox'][3] - box_info['bbox'][1])
            heights_list.append(box_info['box_height'])

        if heights_list:
            np_array = np.array(heights_list)
            return np.median(np_array)
        else:
            return 0

    def __add_general_sentence_info(self, sentences, width_ratio, height_ratio):
        for sentence in sentences:
            bbox = BlockStructureAnalyzerIOU.get_common_boundaries(sentence['words'])
            sentence['bbox'] = bbox
            quantized_bbox = [0,0,0,0]
            quantized_bbox[0] = round(bbox[0] * width_ratio)
            quantized_bbox[1] = round(bbox[1] * height_ratio)
            quantized_bbox[2] = round(bbox[2] * width_ratio)
            quantized_bbox[3] = round(bbox[3] * height_ratio)
            sentence['quantized_bbox'] = quantized_bbox

            sentence_text = ''
            for box_info in sentence['words']:
                sentence_text += box_info['text'] + ' '

            sentence_text = sentence_text[:-1]
            sentence['text'] = sentence_text
            center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
            center_y = bbox[1] + (bbox[3] - bbox[1]) / 2
            sentence['left'] = round(bbox[0] * width_ratio)
            sentence['top'] = round(bbox[1] * height_ratio)
            sentence['right'] = round(bbox[2] * width_ratio)
            sentence['bottom'] = round(bbox[3] * height_ratio)

            sentence['center_x'] = round(center_x * width_ratio)
            sentence['center_y'] = round(center_y * height_ratio)

    def contains_char(self, string_2_check, char_list):
        for char in string_2_check:
            if char in char_list:
                return True

        return False


