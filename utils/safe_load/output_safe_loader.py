from abc import ABC, abstractmethod

from enum import Enum

class SafeLoadType(Enum):
    VALID = 1
    CORRECTED = 2
    INVALID = 3

class OutputSafeLoader(ABC):

    def __init__(self):
        super().__init__()

    def _is_valid_lmdx_list(self, lmdx_list):
        if not type(lmdx_list) is list:
            return False

        for list_entry in lmdx_list:
            if len(list_entry) != 2:
                return False

            if not self._is_valid_lmdx_entity(list_entry[0]):
                return False

            if not self._is_valid_lmdx_entity(list_entry[1]):
                return False

        return True

    def _remove_invalid_entities(self, lmdx_list):
        filtered_list = []
        if not type(lmdx_list) is list:
            return filtered_list


        for list_entry in lmdx_list:
            if len(list_entry) != 2:
                continue

            if not self._is_valid_lmdx_entity(list_entry[0]):
                continue

            if not self._is_valid_lmdx_entity(list_entry[1]):
                continue

            filtered_list.append(list_entry)

        return filtered_list

    def _is_valid_lmdx_entity(self, string_to_check):
        split_string = string_to_check.split()
        if len(split_string) < 2:
            return False

        coordinates = split_string[-1]
        num_of_coordinates = len(coordinates.split('|'))
        if num_of_coordinates != 2 and num_of_coordinates != 4:
            return False

        return True

    def _remove_tail_duplications(self, lmdx_list):
        filtered_list = []
        previous_key = previous_value = ''
        number_of_duplications = 0
        for list_entry in lmdx_list:
            if list_entry[0] == previous_key and list_entry[1] == previous_value:
                number_of_duplications += 1
                continue

            previous_key = list_entry[0]
            previous_value = list_entry[1]
            filtered_list.append(list_entry)

        # print('Number of tail duplications ', number_of_duplications)
        return filtered_list
