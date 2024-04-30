from utils.lmdx.lmdx_creator import LMDXCreator


class LmdxUtils():

    def get_lmdx_entity_text(self, lmdx_entity):
        value_words = lmdx_entity.split()[:-1]
        return ' '.join(value_words)

    def get_lmdx_entity_bbox(self, lmdx_entity):
        split_entity = lmdx_entity.split()
        if len(split_entity) < 2: # not valid lmdx
            return [-1000, -1000, 1000, -1000]

        coordinates_pair_str = split_entity[-1]
        coordinates = coordinates_pair_str.split('|')
        if len(coordinates) == 2:
            return [int(coordinates[0]), int(coordinates[1])]
        elif len(coordinates) == 4:
            return [int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])]
        else: # not valid lmdx
            return [-1000, -1000, 1000, -1000]

    def get_lmdx_sentences_holder(self, tesseract_dict):
        ocr_output = tesseract_dict
        lmdx_creator = LMDXCreator()
        lmdx_sentences_holder = lmdx_creator.create_sentences_from_ocr(ocr_output)
        lmdx_sentences_holder.remove_new_lines()
        return lmdx_sentences_holder
