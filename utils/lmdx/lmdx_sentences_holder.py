class LMDXSentencesHolder():
    def __init__(self, sentences_info_list, image_shape, quantized_width, quantized_height):
        # self.sentences_info_list = sentences_info_list
        self.sentences_info_list = sorted(sentences_info_list, key=lambda x: (x['center_y'], x['center_x']))
        self.quantized_width = quantized_width
        self.quantized_height = quantized_height
        self.image_shape = image_shape
        self.width_ratio = self.quantized_width / image_shape[0]
        self.height_ratio = self.quantized_height / image_shape[1]


    def get_sentences(self):
        return self.sentences_info_list

    def remove_new_lines(self):
        for sentence in self.sentences_info_list:
            sentence['text'] = sentence['text'].replace("\n", "")

    def get_prompt(self, use_center_coordinate_only = True):
        prompt = ''
        for sentence in self.sentences_info_list:
            if use_center_coordinate_only:
                prompt += sentence['text'] + ' '+str(sentence['center_x'])+'|'+str(sentence['center_y'])+'\n'
            else:
                prompt += sentence['text'] + ' '+str(sentence['left'])+'|'+str(sentence['top']) + '|'+str(sentence['right'])+'|'+str(sentence['bottom'])+'\n'

        prompt = prompt[:-1]
        return prompt

    def quantize_bbox(self, bbox):
        normalized_bbox = [0,0,0,0]
        normalized_bbox[0] = round(bbox[0] * self.width_ratio)
        normalized_bbox[1] = round(bbox[1] * self.height_ratio)
        normalized_bbox[2] = round(bbox[2] * self.width_ratio)
        normalized_bbox[3] = round(bbox[3] * self.height_ratio)
        return normalized_bbox

    def get_box_center(self, bbox):
        center_x = round((bbox[2] + bbox[0]) / 2)
        center_y = round((bbox[3] + bbox[1]) / 2)
        return center_x, center_y
