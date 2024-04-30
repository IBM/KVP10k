import json

from utils.lmdx.lmdx_utils import LmdxUtils
from utils.safe_load.json_safe_loader import JsonSafeLoader


class ResponseHandler():

    def __init__(self):
        self.lmdx_utils = LmdxUtils()
        self.quantized_width = 100
        self.quantized_height = 100

    def convert_2_common_gt(self, response, page_dimensions):
        json_loader = JsonSafeLoader()
        kvps, load_type = json_loader.load_json(response)

        width_ratio = self.quantized_width / page_dimensions[0]
        height_ratio = self.quantized_height / page_dimensions[1]

        common_gt_list = []
        common_gt_dict = {'kvps_list': common_gt_list}
        for kvp in kvps:
            try:
                common_gt_entry = self.__create_common_gt_entry(kvp, width_ratio, height_ratio)
                common_gt_list.append(common_gt_entry)
            except:
                # entry is invalid, so we add nothing
                pass

        return json.dumps(common_gt_dict)


    def __create_common_gt_entry(self, lmdx_kvp, width_ratio, height_ratio):
        key_text = self.lmdx_utils.get_lmdx_entity_text(lmdx_kvp[0])
        key_bbox = self.__get_image_coordinates_bbox(self.lmdx_utils.get_lmdx_entity_bbox(lmdx_kvp[0]), width_ratio, height_ratio)
        value_text = self.lmdx_utils.get_lmdx_entity_text(lmdx_kvp[1])
        value_bbox = self.__get_image_coordinates_bbox(self.lmdx_utils.get_lmdx_entity_bbox(lmdx_kvp[1]), width_ratio, height_ratio)
        if 'implicit' in key_text.lower():
            return self.__create_unkeyed_kvp(key_text, value_text, value_bbox)
        elif 'not presented' in value_text.lower():
            return self.__create_unvalued_kvp(key_text, key_bbox)
        else:
            return self.__create_regular_kvp(key_text, key_bbox, value_text, value_bbox)

    def __get_image_coordinates_bbox(self, quantized_bbox, width_ratio, height_ratio):
        image_coordinates_bbox = [0,0,0,0]
        image_coordinates_bbox[0] = round(quantized_bbox[0] / width_ratio)
        image_coordinates_bbox[1] = round(quantized_bbox[1] / height_ratio)
        image_coordinates_bbox[2] = round(quantized_bbox[2] / width_ratio)
        image_coordinates_bbox[3] = round(quantized_bbox[3] / height_ratio)
        return image_coordinates_bbox

    def __create_regular_kvp(self, key_text, key_bbox, value_text, value_bbox):
        key = {'text':key_text, 'bbox' : key_bbox}
        value = {'text':value_text, 'bbox' : value_bbox}
        regular_kvp = {'type' : 'kvp', 'key' : key, 'value':value}
        return regular_kvp

    def __create_unkeyed_kvp(self, key_text, value_text, value_bbox):
        key_text = key_text.replace('implicit', '').strip().lower()
        key = {'text':key_text}
        value = {'text':value_text, 'bbox' : value_bbox}
        unkeyed_kvp = {'type' : 'unkeyed', 'key' : key, 'value':value}
        return unkeyed_kvp

    def __create_unvalued_kvp(self, key_text, key_bbox):
        key = {'text':key_text, 'bbox' : key_bbox}
        unvalued_kvp = {'type' : 'unvalued', 'key' : key}
        return unvalued_kvp