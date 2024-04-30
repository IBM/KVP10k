import json
import math
import os

from PIL import ImageDraw, Image, ImageFont

from utils.visualization.common import draw_arrowed_line


def visualize_gt(image_file, common_gt_file):
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)

    with open(common_gt_file, "r") as f:
        kvp_json = json.load(f)
        kvps = kvp_json['kvps_list']

        for kvp_dict in kvps:
            if kvp_dict['type'] == 'unkeyed':
                draw_unkeyed_kvp(kvp_dict['key'], kvp_dict['value'], draw)
            elif kvp_dict['type'] == 'kvp':
                draw_flat_kvp(kvp_dict['key'], kvp_dict['value'], draw)
            elif kvp_dict['type'] == 'unvalued':
                draw_unvalued_kvp(kvp_dict['key'], draw)

    return img

def draw_bbox_with_text(bbox, text, word_color, draw, font_size=12, padding=2, rect_width=2):
    #draw bbox
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=word_color, width=rect_width)

    #draw text
    font_file = os.path.dirname(os.path.realpath(__file__)) + '/arial.ttf'
    font = ImageFont.truetype(font_file, font_size)

    boxwidth = bbox[2] - bbox[0]
    (width, height), (offset_x, offset_y) = font.font.getsize(text)
    tx0 = bbox[0] + (boxwidth - width) / 2
    ty0 = bbox[1] - font_size - padding
    draw.rectangle([(tx0 - padding, ty0 - padding), (tx0 + width + padding, bbox[1])],
                   fill=(51, 102, 0, 150), width=rect_width)
    text_start = (bbox[0] + (boxwidth - width) / 2, bbox[1] - font_size - padding)
    draw.text(text_start, text, (255, 255, 255), font=font)

def draw_kvp_entity(kvp_entity, draw, entity_color):
    radius = 5
    circle_color = (255, 0, 0)  # Red
    kvp_bbox = kvp_entity['bbox']

    draw_bbox_with_text(kvp_bbox, kvp_entity['text'], entity_color, draw)

    location_x = round((kvp_bbox[2] + kvp_bbox[0]) / 2)
    location_y = round((kvp_bbox[3] + kvp_bbox[1]) / 2)

    # Draw a circle
    draw.ellipse(
        (location_x - radius, location_y - radius, location_x + radius, location_y + radius),
        outline=circle_color, fill=circle_color, width=2)


def draw_flat_kvp(key, value, draw):
    key_entity = {'bbox' : key['bbox'], 'text' : key['text']}
    value_entity = {'bbox': value['bbox'], 'text': value['text']}
    draw_kvp_entity(key_entity, draw, entity_color='blue')
    draw_kvp_entity(value_entity, draw, entity_color='green')

    key_center_x = round((key['bbox'][2] + key['bbox'][0]) / 2)
    key_center_y = round((key['bbox'][3] + key['bbox'][1]) / 2)
    value_center_x = round((value['bbox'][2] + value['bbox'][0]) / 2)
    value_center_y = round((value['bbox'][3] + value['bbox'][1]) / 2)

    draw_arrowed_line(draw, (value_center_x, value_center_y), (key_center_x, key_center_y), color='green')

def draw_unkeyed_kvp(key, value, draw):
    entity = {'bbox' : value['bbox'], 'text' : 'unkeyed '+key['text']+' - '+value['text']}
    draw_kvp_entity(entity, draw, entity_color='orange')

def draw_unvalued_kvp(key,  draw):
    entity = {'bbox' : key['bbox'], 'text' : 'Unvalued - '+key['text']}
    draw_kvp_entity(entity, draw, entity_color='magenta')
