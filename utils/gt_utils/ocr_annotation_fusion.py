import logging

import math
from typing import List
import numpy as np

from utils.annotation_utils.labels import Labels, DepricatedLabels

LETTER_WIDTH_HEIGHT_RATIO = 0.5

class ReadingOrderCreator():

    def __init__(self):
        pass

    def get_simple_reading_order(self, boxes_info_list: List) -> str:
        median_height = self.__get_median_height(boxes_info_list)
        left_x_values = [d['bbox'][0] for d in boxes_info_list]
        min_left_x = min(left_x_values)
        tab_x_distance = (median_height / 2) * 6
        # lines_step = round(median_height / 2)
        # self.__set_center_y(boxes_info_list, lines_step)
        # sorted_boxes = sorted(boxes_info_list, key=lambda x: (x['bbox'][1], x['center_y']))
        sorted_boxes = sorted(boxes_info_list, key=lambda x: (x['line_num'], x['bbox'][0]))
        # text_boxes = [x['text'] for x in sorted_boxes]
        # sorted_string = ' '.join(text_boxes)
        # print(sorted_string)

        simple_reading_order_text = ''
        last_line = -1
        last_y = -1
        last_x = -1
        for box_info in sorted_boxes:
            if box_info['line_num'] == last_line:
                if last_x == -1:  # first word in the line
                    leading_tabs = self.__get_tabs_according_2_distance(tab_x_distance,
                                                                        box_info['bbox'][0] - min_left_x)
                    simple_reading_order_text += leading_tabs + box_info['text']
                elif (box_info['bbox'][0] - last_x) > tab_x_distance:
                    tabs = self.__get_tabs_according_2_distance(tab_x_distance, box_info['bbox'][0] - last_x)
                    simple_reading_order_text += tabs + box_info['text']
                else:
                    simple_reading_order_text += ' ' + box_info['text']

                last_x = box_info['bbox'][2]
            else:  # new line
                if last_line == -1:  # first line
                    leading_tabs = self.__get_tabs_according_2_distance(tab_x_distance,
                                                                        box_info['bbox'][0] - min_left_x)
                    simple_reading_order_text += leading_tabs + box_info['text']
                else:
                    new_lines = self.__get_new_lines_according_2_distance(median_height * 3,
                                                                          abs(box_info['bbox'][1] - last_y))
                    leading_tabs = self.__get_tabs_according_2_distance(tab_x_distance,
                                                                        box_info['bbox'][0] - min_left_x)
                    simple_reading_order_text += new_lines + leading_tabs + box_info['text']
                last_x = box_info['bbox'][2]
                last_y = box_info['bbox'][3]
                last_line = box_info['line_num']

        return simple_reading_order_text

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

    def __set_center_y(self, boxes_info_list, lines_step):
        for box_info in boxes_info_list:
            box_info['center_y'] = round((box_info['bbox'][1] + box_info['box_height'] / 2) / lines_step)

    def __get_tabs_according_2_distance(self, tab_x_distance, distance_2_next_word):
        num_of_tabs = math.floor(distance_2_next_word / tab_x_distance)
        return '\t' * num_of_tabs

    def __get_new_lines_according_2_distance(self, line_space, distance_2_next_line):
        num_of_lines = max(1, math.ceil(distance_2_next_line / line_space))
        return '\n' * num_of_lines

class OrderedLinesSetter:

    def __init__(self, bounding_boxes_array):
        self.last_phrase_num = 0
        self.MIN_LETTERS_TO_SPLIT_PHRASE = 500
        # self.vertical_boxes, self.horizontal_boxes = self.__divide_2_groups(self.bounding_boxes_array)
        self.horizontal_boxes = bounding_boxes_array

        # self.__split_to_vertical_phrases(self.vertical_boxes)
        self.__split_to_horizontal_phrases(self.horizontal_boxes)

    def set_lines_order(self):
        sorted_phrases = sorted(self.horizontal_phrases, key=lambda k: k['average_y'])

        for line_number, phrase in enumerate(sorted_phrases):
            for box_info in phrase['children']:
                box_info['line_num'] = line_number

    def __split_to_horizontal_phrases(self, bounding_boxes_array):
        self.horizontal_phrases = []
        sorted_array = sorted(bounding_boxes_array, key=lambda k: k['bbox'][0])

        for bounding_box in sorted_array:
            phrase = self.__find_horizontal_phrase(bounding_box, self.horizontal_phrases)
            if phrase is None:
                self.horizontal_phrases.append(self.__create_new_horizontal_phrase(bounding_box))
            else:
                self.__append_to_horizontal_phrase(phrase, bounding_box)

        self.__set_phrases_average_y(self.horizontal_phrases)

    def __set_phrases_average_y(self, phrases):
        for phrase in phrases:
            average_y = abs((phrase['children'][0]['bbox'][1] + phrase['children'][-1]['bbox'][3])) / 2
            phrase['average_y'] = phrase['children'][0]['bbox'][1]

    def __split_to_vertical_phrases(self, bounding_boxes_array):
        self.vertical_phrases = []
        sorted_array = sorted(bounding_boxes_array, key=lambda k: k['bbox'][1])

        for bounding_box in sorted_array:
            phrase = self.__find_vertical_phrase(bounding_box, self.vertical_phrases)
            if phrase is None:
                self.vertical_phrases.append(self.__create_new_vertical_phrase(bounding_box))
            else:
                self.__append_to_vertical_phrase(phrase, bounding_box)

    def __find_horizontal_phrase(self, bounding_box, phrases):
        letter_width = self.__get_pixels_for_box_letter(bounding_box)
        max_allowed_distance_4_box = self.MIN_LETTERS_TO_SPLIT_PHRASE * letter_width

        for phrase in phrases:
            if phrase['is_finished']:
                continue

            max_allowed_distance_4_phrase = max(phrase['max_allowed_distance'], max_allowed_distance_4_box)
            x_distance_2_phrase = bounding_box['bbox'][0] - phrase['last_x']
            if x_distance_2_phrase > max_allowed_distance_4_phrase:
                phrase['is_finished'] = True
                continue

            box_height = bounding_box['rotated_height']
            box_center_y = self.__get_center_y(bounding_box)
            if abs(phrase['center_y'] - box_center_y) > (10 * box_height):
                continue

            exact_distance_2phrase = self.__get_exact_horizontal_distance_2phrase(bounding_box, phrase)
            if exact_distance_2phrase > max_allowed_distance_4_phrase:
                continue

            line_m = phrase['m']
            line_n = phrase['n']

            distance_2line = self.__get_distance_2line_from_left_edge(line_m, line_n, bounding_box)

            if distance_2line < (box_height / 3):
                return phrase

        return None

    def __find_vertical_phrase(self, bounding_box, phrases):
        letter_width = self.__get_pixels_for_box_letter(bounding_box)
        max_allowed_distance_4_box = self.MIN_LETTERS_TO_SPLIT_PHRASE * letter_width

        for phrase in phrases:
            if phrase['is_finished']:
                continue

            max_allowed_distance_4_phrase = max(phrase['max_allowed_distance'], max_allowed_distance_4_box)
            y_distance_2_phrase = bounding_box['bbox'][1] - phrase['last_y']
            if y_distance_2_phrase > max_allowed_distance_4_phrase:
                phrase['is_finished'] = True
                continue

            box_height = bounding_box['rotated_height']
            box_center_x = self.__get_center_x(bounding_box)
            if abs(phrase['center_x'] - box_center_x) > (10 * box_height):
                continue

            exact_distance_2phrase = self.__get_exact_vertical_distance_2phrase(bounding_box, phrase)
            if exact_distance_2phrase > max_allowed_distance_4_phrase:
                continue

            line_m = phrase['m']
            line_n = phrase['n']

            distance_2line = self.__get_distance_2line_from_top_edge_4_vertical(line_m, line_n, bounding_box)

            if distance_2line >= (box_height / 3):
                continue

            if phrase['script'] != bounding_box['script']:
                continue

            if not self.is_there_horizontal_line_between(phrase, bounding_box):
                return phrase

        return None

    def __append_to_horizontal_phrase(self, phrase, bounding_box):
        x1, y1 = phrase['left_edge_coords']
        x2, y2 = self.__get_right_edge_center_coords(bounding_box)
        m, n = self.__get_line_equation_by_2points(x1, y1, x2, y2)
        phrase['m'] = (phrase['m'] * phrase['length_in_pixels'] + m * bounding_box['rotated_width']) / (
                    phrase['length_in_pixels'] + bounding_box['rotated_width'])
        phrase['n'] = (phrase['n'] * phrase['length_in_pixels'] + n * bounding_box['rotated_width']) / (
                    phrase['length_in_pixels'] + bounding_box['rotated_width'])
        bounding_box['line_num'] = phrase['name']
        phrase['last_x'] = bounding_box['bbox'][2]
        phrase['right_edge_coords'] = self.__get_right_edge_center_coords(bounding_box)
        letter_width = self.__get_pixels_for_box_letter(bounding_box)
        max_allowed_distance_4_box = self.MIN_LETTERS_TO_SPLIT_PHRASE * letter_width
        phrase['max_allowed_distance'] = max(max_allowed_distance_4_box, phrase['max_allowed_distance'])
        phrase['average_height'] = (phrase['average_height'] * len(phrase['children']) + bounding_box[
            'rotated_height']) / (len(phrase['children']) + 1)
        phrase['length_in_pixels'] += bounding_box['rotated_width']
        phrase['children'].append(bounding_box)

    def __append_to_vertical_phrase(self, phrase, bounding_box):
        x1, y1 = phrase['top_edge_coords']
        x2, y2 = self.__get_bottom_edge_center_coords_4_vertical(bounding_box)
        m, n = self.__get_line_equation_by_2points(x1, y1, x2, y2)
        phrase['m'] = (phrase['m'] * phrase['length_in_pixels'] + m * bounding_box['rotated_width']) / (
                    phrase['length_in_pixels'] + bounding_box['rotated_width'])
        phrase['n'] = (phrase['n'] * phrase['length_in_pixels'] + n * bounding_box['rotated_width']) / (
                    phrase['length_in_pixels'] + bounding_box['rotated_width'])
        bounding_box['line_num'] = phrase['name']
        phrase['last_y'] = bounding_box['bbox'][3]
        phrase['bottom_edge_coords'] = self.__get_bottom_edge_center_coords_4_vertical(bounding_box)
        letter_width = self.__get_pixels_for_box_letter(bounding_box)
        max_allowed_distance_4_box = self.MIN_LETTERS_TO_SPLIT_PHRASE * letter_width
        phrase['max_allowed_distance'] = max(max_allowed_distance_4_box, phrase['max_allowed_distance'])
        phrase['length_in_pixels'] += bounding_box['rotated_width']
        phrase['children'].append(bounding_box)

    def __create_new_horizontal_phrase(self, bounding_box):
        phrase = {}
        m, n = self.__get_line_equation(bounding_box)
        phrase['is_finished'] = False

        phrase['m'] = m
        phrase['n'] = n
        phrase['left_edge_coords'] = self.__get_left_edge_center_coords(bounding_box)
        phrase['right_edge_coords'] = self.__get_right_edge_center_coords(bounding_box)
        letter_width = self.__get_pixels_for_box_letter(bounding_box)
        max_allowed_distance = self.MIN_LETTERS_TO_SPLIT_PHRASE * letter_width
        phrase['max_allowed_distance'] = max_allowed_distance
        phrase['center_y'] = self.__get_center_y(bounding_box)
        self.last_phrase_num += 1
        phrase['name'] = str(self.last_phrase_num)
        phrase['children'] = []
        bounding_box['line_num'] = phrase['name']
        bounding_box['m'] = m
        phrase['last_x'] = bounding_box['bbox'][2]
        phrase['average_height'] = bounding_box['rotated_height']
        phrase['children'].append(bounding_box)
        phrase['length_in_pixels'] = bounding_box['rotated_width']
        return phrase

    def __create_new_vertical_phrase(self, bounding_box):
        phrase = {}
        m, n = self.__get_line_equation(bounding_box)
        phrase['is_finished'] = False
        phrase['script'] = bounding_box['script']
        phrase['script_confidence'] = bounding_box['script_confidence']
        if 'handwriting_confidence' in bounding_box:
            phrase['handwriting_confidence'] = bounding_box['handwriting_confidence']
        phrase['m'] = m
        phrase['n'] = n
        phrase['top_edge_coords'] = self.__get_top_edge_center_coords_4_vertical(bounding_box)
        phrase['bottom_edge_coords'] = self.__get_bottom_edge_center_coords_4_vertical(bounding_box)
        letter_width = self.__get_pixels_for_box_letter(bounding_box)
        max_allowed_distance = self.MIN_LETTERS_TO_SPLIT_PHRASE * letter_width
        phrase['max_allowed_distance'] = max_allowed_distance
        phrase['center_x'] = self.__get_center_x(bounding_box)
        self.last_phrase_num += 1
        phrase['name'] = str(self.last_phrase_num)
        phrase['children'] = []
        bounding_box['line_num'] = phrase['name']
        bounding_box['m'] = m
        phrase['last_y'] = bounding_box['bbox'][3]
        phrase['length_in_pixels'] = bounding_box['rotated_height']
        phrase['children'].append(bounding_box)
        return phrase

    def __get_pixels_for_box_letter(self, bounding_box):
        height = bounding_box['rotated_height']
        return LETTER_WIDTH_HEIGHT_RATIO * height

    def __get_center_y(self, bounding_box):
        bbox = bounding_box['bbox']
        height = bbox[3] - bbox[1]
        center_y = bbox[1] + height / 2
        return center_y

    def __get_center_x(self, bounding_box):
        bbox = bounding_box['bbox']
        height = bbox[2] - bbox[0]
        center_x = bbox[0] + height / 2
        return center_x

    def __get_line_equation_by_2points(self, x1, y1, x2, y2):
        if x2 == x1:
            # let's shift x1 a little bit - it doesn't harm too much the accuracy
            # but allows us to stay with generic equation
            x1 = x1 - 0.01

        m = (y2 - y1) / (x2 - x1)
        n = y1 - (m * x1)
        return m, n

    def __get_line_equation(self, bounding_box):
        # line equation doesn't depend on vertical/horizontal box since
        # points are arranged in such way that the first one (index of 0) is the
        # second one on the height side (clockwise)
        polygon = bounding_box["polygon"]
        x1 = (polygon[0][0] + polygon[3][0]) / 2
        y1 = (polygon[0][1] + polygon[3][1]) / 2

        x2 = (polygon[1][0] + polygon[2][0]) / 2
        y2 = (polygon[1][1] + polygon[2][1]) / 2

        return self.__get_line_equation_by_2points(x1, y1, x2, y2)

    def __get_left_edge_center_coords(self, bounding_box):
        polygon = bounding_box["polygon"]
        x0 = (polygon[0][0] + polygon[3][0]) / 2
        y0 = (polygon[0][1] + polygon[3][1]) / 2
        return x0, y0

    def __get_right_edge_center_coords(self, bounding_box):
        polygon = bounding_box["polygon"]
        x0 = (polygon[1][0] + polygon[2][0]) / 2
        y0 = (polygon[1][1] + polygon[2][1]) / 2
        return x0, y0

    def __get_top_edge_center_coords_4_vertical(self, bounding_box):
        polygon = bounding_box["polygon"]

        y0 = (polygon[0][1] + polygon[3][1]) / 2
        y1 = (polygon[1][1] + polygon[2][1]) / 2
        if y0 < y1:
            x0 = (polygon[0][0] + polygon[3][0]) / 2
            return x0, y0
        else:
            x1 = (polygon[1][0] + polygon[2][0]) / 2
            return x1, y1

    def __get_bottom_edge_center_coords_4_vertical(self, bounding_box):
        polygon = bounding_box["polygon"]

        y0 = (polygon[0][1] + polygon[3][1]) / 2
        y1 = (polygon[1][1] + polygon[2][1]) / 2
        if y0 > y1:
            x0 = (polygon[0][0] + polygon[3][0]) / 2
            return x0, y0
        else:
            x1 = (polygon[1][0] + polygon[2][0]) / 2
            return x1, y1

    def __get_distance_2line_from_left_edge(self, line_m, line_n, bounding_box):
        x0, y0 = self.__get_left_edge_center_coords(bounding_box)

        distance = abs(-1 * line_m * x0 + y0 - line_n) / math.sqrt(line_m * line_m + 1)
        return distance

    def __get_distance_2line_from_top_edge_4_vertical(self, line_m, line_n, bounding_box):
        x0, y0 = self.__get_top_edge_center_coords_4_vertical(bounding_box)

        distance = abs(-1 * line_m * x0 + y0 - line_n) / math.sqrt(line_m * line_m + 1)
        return distance

    def __get_distance_2line_from_center(self, line_m, line_n, bounding_box):
        polygon = bounding_box["polygon"]
        x0 = (polygon[0][0] + polygon[1][0] + polygon[2][0] + polygon[3][0]) / 4
        y0 = (polygon[0][1] + polygon[1][1] + polygon[2][1] + polygon[3][1]) / 4

        distance = abs(-1 * line_m * x0 + y0 - line_n) / math.sqrt(line_m * line_m + 1)
        return distance

    def __get_exact_horizontal_distance_2phrase(self, bounding_box, phrase):
        x_distance_2_phrase = bounding_box['bbox'][0] - phrase['last_x']
        if x_distance_2_phrase < 0:
            return 0
        x1, y1 = phrase['right_edge_coords']
        x2, y2, = self.__get_left_edge_center_coords(bounding_box)
        width = x2 - x1
        height = y2 - y1
        return math.sqrt(width * width + height * height)

    def __get_exact_vertical_distance_2phrase(self, bounding_box, phrase):
        y_distance_2_phrase = bounding_box['bbox'][1] - phrase['last_y']
        if y_distance_2_phrase < 0:
            return 0
        x1, y1 = phrase['bottom_edge_coords']
        x2, y2, = self.__get_top_edge_center_coords_4_vertical(bounding_box)
        width = x2 - x1
        height = y2 - y1
        return math.sqrt(width * width + height * height)

    def __divide_2_groups(self, bounding_boxes_array):
        vertical_boxes = []
        horizontal_boxes = []
        for bounding_box in bounding_boxes_array:
            if bounding_box['is_vertical']:
                vertical_boxes.append(bounding_box)
            else:
                horizontal_boxes.append(bounding_box)

        return vertical_boxes, horizontal_boxes

class OcrAnnotationFusion:
    WORD_MATCH_THRESHOLD = 0.6
    total_links = 0

    def __init__(self):
        self.quantized_width = 100
        self.quantized_height = 100

    def convert_file(self, image, ocr, item):
        telus_annotation_dict = item
        tesseract_page0 = ocr['pages'][0]['words']
        image_dims = image.size
        width_ratio = self.quantized_width / image_dims[0]
        height_ratio = self.quantized_height / image_dims[1]

        self.__add_common_rectangles_2_telus(telus_annotation_dict, image_dims)
        self.__add_common_rectangles_2_tesseract(tesseract_page0)
        tesseract_words = tesseract_page0

        telus_entities = []
        kvps_list = []

        word_id = 1
        for rectangle in telus_annotation_dict['annotations']['rectangles']:
            label = rectangle['label'].lower().strip()
            if label == DepricatedLabels.IGNORE_ZONE:
                continue

            if label == Labels.EMPTY_TEXT or label == DepricatedLabels.TABLE or label == DepricatedLabels.CHECKBOX_SELECTED or label == DepricatedLabels.CHECKBOX_UNSELECTED:
                telus_entities.append(self.__get_empty_common_gt_entity(rectangle, word_id, label))
            elif label == Labels.TEXT or label == Labels.HW_TEXT or label == Labels.FLOATING_DATE or \
                    label == Labels.FLOATING_NAME or label == Labels.FLOATING_ADDRESS or \
                    label == Labels.FLOATING_DOCUMENT_TITLE or label == Labels.FLOATING_PHONE or \
                    label == Labels.FLOATING_DOCUMENT_TYPE or label == Labels.FLOATING_EMAIL or \
                    label == Labels.FLOATING_WEBSITE or label == Labels.FLOATING_TEXT or label == Labels.FLOATING_YEAR or \
                    label == Labels.UNVALUED_KEY:
                inner_words = self.__find_inner_words(rectangle, tesseract_words)
                if not inner_words:
                    logging.debug('No match was found \'' + label + '\'')
                    annotation_text = ''
                    number_of_lines = 0
                    is_handwritten = False
                else:
                    annotation_text, number_of_lines = self.__get_common_text(inner_words)
                    is_handwritten = False
                telus_entities.append(
                    self.__get_common_gt_entity(rectangle, annotation_text, word_id, label, is_handwritten,
                                                number_of_lines, width_ratio, height_ratio))

            else:
                logging.debug(' non recognized ', label)
            word_id += 1

        regular_kvps = self.__add_regular_kvps(telus_entities)
        unkeyed_kvps = self.__add_unkeyed_kvps(telus_entities)
        unvalued_kvps = self.__add_unvalued_kvps(telus_entities)
        kvps_list.extend(regular_kvps)
        kvps_list.extend(unkeyed_kvps)
        kvps_list.extend(unvalued_kvps)

        common_gt_dict = {'kvps_list': kvps_list}
        return common_gt_dict

    def __add_common_rectangles_2_telus(self, telus_annotation_dict, image_dims):
        image_width, image_height = image_dims
        for rectangle in telus_annotation_dict['annotations']['rectangles']:
            telus_coords = rectangle['coordinates']
            xs = [round(coord['x'] * image_width) for coord in telus_coords]
            ys = [round(coord['y'] * image_height) for coord in telus_coords]
            general_rect = {'left': min(xs), 'right': max(xs), 'top': min(ys), 'bottom': max(ys)}
            general_rect['width'] = general_rect['right'] - general_rect['left']
            general_rect['height'] = general_rect['bottom'] - general_rect['top']
            rectangle['general_rect'] = general_rect

    def __add_common_rectangles_2_tesseract(self, tesseract_page0):
        for word in tesseract_page0:

            bbox = self.__get_bbox_4_tesseract_location(word['bbox'])
            word['bbox'] = bbox
            general_rect = {'left': bbox[0], 'right': bbox[2], 'top': bbox[1], 'bottom': bbox[3]}
            general_rect['width'] = general_rect['right'] - general_rect['left']
            general_rect['height'] = general_rect['bottom'] - general_rect['top']
            for key in general_rect:
                general_rect[key] = round(general_rect[key])
            word['general_rect'] = general_rect

    def __get_bbox_4_tesseract_location(self, location):
        return [location['left'], location['top'],
                location['left'] + location['width'], location['top'] + location['height']]

    def __find_inner_words(self, rectangle, tesseract_words):
        inner_words = []
        for tesseract_word in tesseract_words:
            pct_inside = self.get_percentage_inside(tesseract_word['general_rect'], rectangle['general_rect'])
            if pct_inside < self.WORD_MATCH_THRESHOLD:
                continue

            inner_words.append(tesseract_word)
        return inner_words

    def __add_regular_kvps(self, telus_entities):
        regular_kvps = []
        for telus_entity in telus_entities:
            if telus_entity['telus_parent_key_id'] is None:
                continue

            parent_entity = self.__get_parent_telus_entity(telus_entities, telus_entity['telus_parent_key_id'])
            if parent_entity is None:
                if telus_entity['label'] == Labels.TEXT or telus_entity['label'] == Labels.HW_TEXT:
                    logging.debug('Couldnt find kvp linking, probably ocr was not recognized inside telus recognized')
                continue

            regular_kvp = self.__create_regular_kvp(parent_entity, telus_entity)
            regular_kvps.append(regular_kvp)
            self.total_links += 1

        return regular_kvps

    def __add_unkeyed_kvps(self, telus_entities):
        unkeyed_kvps = []
        for telus_entity in telus_entities:
            if 'floating' not in telus_entity['label']:
                continue

            unkeyed_kvp = self.__create_unkeyed_kvp(telus_entity)
            unkeyed_kvps.append(unkeyed_kvp)
            self.total_links += 1

        return unkeyed_kvps

    def __add_unvalued_kvps(self, telus_entities):
        unvalued_kvps = []
        for telus_entity in telus_entities:
            if telus_entity['label'] != Labels.UNVALUED_KEY:
                continue

            unvalued_kvp = self.__create_unvalued_kvp(telus_entity)
            unvalued_kvps.append(unvalued_kvp)
            self.total_links += 1

        return unvalued_kvps

    def __get_common_text(self, words):
        boxes_info_list = []
        for word in words:
            boxes_info_list.append(self.__get_box_info(word))

        simple_reading_order_creator = ReadingOrderCreator()
        order_setter = OrderedLinesSetter(boxes_info_list)
        order_setter.set_lines_order()
        number_of_lines = 0
        for box in boxes_info_list:
            line_num = box['line_num']
            if line_num > number_of_lines:
                number_of_lines = line_num
        number_of_lines += 1
        text = simple_reading_order_creator.get_simple_reading_order(boxes_info_list)
        text = text.replace("\n", " ")
        text = text.replace("\t", " ").strip()
        return text, number_of_lines

    def __get_box_info(self, word):
        box_info = {}
        text = word['text']
        bbox = word['bbox']
        box_info['bbox'] = bbox
        box_info['text'] = text
        box_info["polygon"] = [[bbox[0], bbox[1]],
                               [bbox[2], bbox[1]],
                               [bbox[2], bbox[3]],
                               [bbox[0], bbox[3]]]

        box_info['rotated_height'] = bbox[3] - bbox[1]
        box_info['rotated_width'] = bbox[2] - bbox[0]

        return box_info

    def __get_common_gt_entity(self, telus_rectangle, annotation_text, id, label, is_handwritten, number_of_lines,
                               width_ratio, height_ratio):
        common_gt_entity = {'id': id, 'telus_id': telus_rectangle['_id'], 'label': label,
                            'word': annotation_text, 'handwritten': is_handwritten,
                            'number_of_lines': number_of_lines}

        general_rect = telus_rectangle['general_rect']
        rectangle_attributes = telus_rectangle['attributes']
        common_gt_entity['bbox'] = [general_rect['left'], general_rect['top'], general_rect['right'],
                                    general_rect['bottom']]
        center_x, center_y = self.__get_bbox_quantized_center(common_gt_entity['bbox'], width_ratio, height_ratio)
        common_gt_entity['center_x'] = center_x
        common_gt_entity['center_y'] = center_y
        if 'Linking' in rectangle_attributes and \
                (rectangle_attributes['Linking'] != None or rectangle_attributes["attributes"]["Linking"]["value"] != "NA"): #Inbar
            common_gt_entity['telus_parent_key_id'] = rectangle_attributes['Linking']['value']
        else:
            common_gt_entity['telus_parent_key_id'] = None

        return common_gt_entity

    def __get_empty_common_gt_entity(self, telus_rectangle, id, label):
        telus_entity = {'id': id, 'telus_id': telus_rectangle['_id'], 'label': label, 'word': '',
                        'handwritten': False}

        rectangle_attributes = telus_rectangle['attributes']
        general_rect = telus_rectangle['general_rect']
        telus_entity['bbox'] = [general_rect['left'], general_rect['top'], general_rect['right'],
                                general_rect['bottom']]

        if 'Linking' in rectangle_attributes and \
                (rectangle_attributes['Linking'] != None or rectangle_attributes["attributes"]["Linking"]["value"] != "NA"): #Inbar
            telus_entity['telus_parent_key_id'] = rectangle_attributes['Linking']['value']
        else:
            telus_entity['telus_parent_key_id'] = None
        return telus_entity

    def get_percentage_inside(self, rect1, rect2):
        intersection = self.get_intersection_area(rect1, rect2)
        first_rect_area = rect1['width'] * rect1['height']
        if first_rect_area == 0:
            return 0
        return intersection / first_rect_area

    def get_intersection_over_union(self, rect1, rect2):
        union = self.get_union_area(rect1, rect2)
        if union == 0:
            logging.debug('oh no')
            return 0
        return self.get_intersection_area(rect1, rect2) / union

    def get_intersection_area(self, rect1, rect2):
        x_overlap = max(0, min(rect1['right'], rect2['right']) - max(rect1['left'], rect2['left']))
        y_overlap = max(0, min(rect1['bottom'], rect2['bottom']) - max(rect1['top'], rect2['top']))
        overlap_area = x_overlap * y_overlap
        return overlap_area

    def get_union_area(self, rect1, rect2):
        union_area = rect1['width'] * rect1['height'] + rect2['width'] * rect2['height'] \
                     - self.get_intersection_area(rect1, rect2)
        return union_area

    def __get_parent_telus_entity(self, telus_enities, parent_id):
        for telus_entity in telus_enities:
            if telus_entity['telus_id'] == parent_id:
                return telus_entity

        return None

    def __create_regular_kvp(self, parent_entity, child_entity):
        key = {'text': parent_entity['word'], 'bbox': parent_entity['bbox']}
        value = {'text': child_entity['word'], 'bbox': child_entity['bbox']}
        regular_kvp = {'type': 'kvp', 'key': key, 'value': value}

        regular_kvp['center_x'] = parent_entity['center_x']
        regular_kvp['center_y'] = parent_entity['center_y']

        return regular_kvp

    def __create_unkeyed_kvp(self, telus_entity):
        key_text = telus_entity['label'].replace('floating', '').strip().lower()
        key = {'text': key_text}
        value = {'text': telus_entity['word'], 'bbox': telus_entity['bbox']}
        unkeyed_kvp = {'type': 'unkeyed', 'key': key, 'value': value}
        unkeyed_kvp['center_x'] = telus_entity['center_x']
        unkeyed_kvp['center_y'] = telus_entity['center_y']
        return unkeyed_kvp

    def __create_unvalued_kvp(self, telus_entity):
        key = {'text': telus_entity['word'], 'bbox': telus_entity['bbox']}
        unvalued_kvp = {'type': 'unvalued', 'key': key}
        unvalued_kvp['center_x'] = telus_entity['center_x']
        unvalued_kvp['center_y'] = telus_entity['center_y']

        return unvalued_kvp

    def __get_bbox_quantized_center(self, bbox, width_ratio, height_ratio):
        center_x = round(((bbox[2] + bbox[0]) / 2) * width_ratio)
        center_y = round(((bbox[3] + bbox[1]) / 2) * height_ratio)
        return center_x, center_y