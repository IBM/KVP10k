import json
import math

from PIL import Image, ImageDraw

from utils.visualization.common import draw_arrowed_line


def coordinates_to_xyxy(coordinates, w, h):
    xy1, xy2, xy3, xy4 = coordinates
    xy1 = {'x': xy1['x'] * w, 'y': xy1['y'] * h}
    xy2 = {'x': xy2['x'] * w, 'y': xy2['y'] * h}
    xy3 = {'x': xy3['x'] * w, 'y': xy3['y'] * h}
    xy4 = {'x': xy4['x'] * w, 'y': xy4['y'] * h}

    x1 = min(xy1['x'], xy2['x'], xy3['x'], xy4['x'])
    x2 = max(xy1['x'], xy2['x'], xy3['x'], xy4['x'])
    y1 = min(xy1['y'], xy2['y'], xy3['y'], xy4['y'])
    y2 = max(xy1['y'], xy2['y'], xy3['y'], xy4['y'])

    return([x1, y1, x2, y2])


def visualize_annotation(image_file, annotation_file):
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    with open(annotation_file, "r") as f:
        kvp_json = json.load(f)
        lines = kvp_json["annotations"]["lines"]
        rectangles = kvp_json["annotations"]["rectangles"]

        link_pair = {}
        rect_ids = {}
        color_id = {}

        for rect in rectangles:
            rect_ids.update({rect["_id"]: rect["coordinates"]})
            color_id.update({rect["_id"]: rect["color"]})
            if "Linking" in rect["attributes"] and rect["attributes"]["Linking"]["value"] != "NA":
                link_pair.update({rect["_id"]: rect["attributes"]["Linking"]["value"]})
            else:
                link_pair.update({rect["_id"]: rect["_id"]})

        for key in link_pair:
            key_coordinates = rect_ids[key]
            draw.rectangle(coordinates_to_xyxy(key_coordinates, w, h), outline=color_id[key], width=2)
            if link_pair[key] in rect_ids:
                value_coordinates = rect_ids[link_pair[key]]
                lp = link_pair[key]
                draw.rectangle(coordinates_to_xyxy(value_coordinates, w, h), outline=color_id[link_pair[key]], width=2)

        for line in lines:
            x1 = line["points"]['p1']['x'] * w
            y1 = line["points"]['p1']['y'] * h
            x2 = line["points"]['p2']['x'] * w
            y2 = line["points"]['p2']['y'] * h
            draw_arrowed_line(draw, (x2, y2), (x1, y1), color=line["color"], width=2)
            #draw.line(((x1, y1), (x2, y2)), fill=line["color"], width=3)

    return img