import json

class annotation_reader:

    def __init__(self, annotation_file):
        with open(annotation_file, encoding='utf-8') as f:
            telus_dict = json.load(f)
            self.rectangles = telus_dict["annotations"]["rectangles"]

    def get_linked_item(self,item):
        linked_item_id = item["attributes"]["Linking"]["value"]
        for linked_item in self.rectangles:
            if linked_item.get('_id') == linked_item_id:
                return linked_item
        return None

    def count_unkeyed_value(self):
        unkeyed_value_items = [item for item in self.rectangles if item['label'].startswith("Floating")]
        return len(unkeyed_value_items)

    def count_unvalued_key(self):
        unvalued_key_items = [item for item in self.rectangles if item['label']=="unvalued_key"]
        return len(unvalued_key_items)

    def count_flat_kvp(self):
        kvp_values_items = [item for item in self.rectangles if \
                            "Linking" in item["attributes"] and item["attributes"]["Linking"]["value"] != "NA" and \
                            self.get_linked_item(item) is not None]
        return len(kvp_values_items)