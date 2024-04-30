import glob
import json

from utils.safe_load.output_safe_loader import SafeLoadType, OutputSafeLoader


class JsonSafeLoader(OutputSafeLoader):

    def __init__(self):
        super().__init__()

    def load_json(self, json_string):
        try:
            json_dict = json.loads(json_string)
            # lmdx_list = json_dict['kvps']
            lmdx_list = json_dict

            is_json_structure_correct = True
            if self._is_valid_lmdx_list(lmdx_list):
                return lmdx_list, SafeLoadType.VALID

        except Exception as e:
            is_json_structure_correct = False

        try:
            if is_json_structure_correct:
                safe_json_dict = json_dict
            else:
                safe_json_dict = self.__safe_parse_json(json_string)

            # lmdx_list = safe_json_dict['kvps']
            lmdx_list = safe_json_dict

            valid_entries_list = self._remove_invalid_entities(lmdx_list)
            filtered_list = self._remove_tail_duplications(valid_entries_list)

            if len(filtered_list) > 0 or len(lmdx_list) == 0:
                return filtered_list, SafeLoadType.CORRECTED
            else:
                return [], SafeLoadType.INVALID
        except:
            return [], SafeLoadType.INVALID

    def __safe_parse_json(self, json_string):
        closing_brackets_positions = [i for i, char in enumerate(json_string) if char == ']']
        for bracket_position in reversed(closing_brackets_positions):
            try:
                json_sub_string = json_string[:bracket_position + 1]
                # json_sub_string += ']}'
                json_sub_string += ']'
                safe_json_dict = json.loads(json_sub_string)
                return safe_json_dict
            except Exception as e:
                pass

        return {}

if __name__ == '__main__':
    test_path = r'C:\Users\YEVGENYYAROKER\Desktop\doc_gpt\telus_h\full_coordinates'

    json_safe_loader = JsonSafeLoader()
    files_to_process = glob.glob(test_path + "/*.json")
    valid = corrected = invalid = 0
    for index, file_path in enumerate(files_to_process):
        if '.iocr' in file_path:
            continue

        print('Processing ', file_path)
        try:
            with open(file_path, 'r') as file:
                json_string = file.read()

                json_dict, load_type = json_safe_loader.load_json(json_string)
                print(load_type)
                if load_type == SafeLoadType.VALID:
                    valid += 1
                elif load_type == SafeLoadType.INVALID:
                    print("Invalid json:", json_string)
                    invalid += 1
                elif load_type == SafeLoadType.CORRECTED:
                    corrected += 1

        except FileNotFoundError:
            print(f"The file '{file_path}' does not exist.")

        print('There are ', valid,' valid ', corrected,' corrected ', invalid,' invalid ')
