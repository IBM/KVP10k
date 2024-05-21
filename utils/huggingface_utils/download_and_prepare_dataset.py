import requests
# for get_image
import pdf2image
from pdf2image import convert_from_bytes
# for get_ocr
from tesserocr import RIL, iterate_level, PyTessBaseAPI, PSM, OEM
# for get_gt
from utils.gt_utils.ocr_annotation_fusion import OcrAnnotationFusion


def get_image(image_url, page_number):

    # GET request to fetch the content of the PDF file in the given url
    # raise requests.exceptions.HTTPError - HTTP error e.g., 404, 501, etc.
    # raise requests.exceptions.ConnectionError - if url not valid
    # raise requests.exceptions.Timeout - Request timed out
    # raise requests.exceptions.SSLError
    # raise requests.exceptions.RequestException
    # raise pdf2image.exceptions.PDFPageCountError
    if page_number==0:
        # raise IndexError if the page_number does not exists
        raise pdf2image.exceptions.PDFPageCountError(f"{image_url} - page_number 0")
    try:
        response = requests.get(image_url)

        if response.status_code != 200:
            response.raise_for_status()

        # Memory will quickly fill if all the pages are converted
        images = convert_from_bytes(response.content, dpi=300, use_cropbox=True, first_page=page_number, last_page=page_number)
        
        if len(images) == 0:
            # raise IndexError if the page_number does not exists
            raise pdf2image.exceptions.PDFPageCountError(f"{image_url} - page_number 0")

        return images[0]

            
    except Exception as e:
        info = f"{image_url} - page_number {page_number}"
        raise type(e)(info)


def get_ocr(image, tessdata_path):
    ocr_format = {'pages': []}
    page_0 = {}

    page_dimensions = image.size
    page_0['width'] = page_dimensions[0]
    page_0['height'] = page_dimensions[1]

    ocr_format['pages'].append(page_0)

    api = PyTessBaseAPI(path=tessdata_path, psm=PSM.AUTO, oem=OEM.LSTM_ONLY)
    api.SetImage(image)
    api.Recognize()

    words = []
    ri = api.GetIterator()
    level = RIL.WORD
    for il in iterate_level(ri, level):
        try:
            box_info = {}
            box = il.BoundingBox(level)
            l, t, r, b = box
            location = {}
            location["left"] = l
            location["top"] = t
            location["width"] = r - l
            location["height"] = b - t
            box_info['bbox'] = location
            box_info['text'] = il.GetUTF8Text(level)
        except Exception as error:
            print(error)
            box_info = {'text': "", 'bbox': None}

        if box_info['bbox'] == None:
            continue

        if box_info['text'].isspace():
            continue

        words.append(box_info)
    api.End()

    page_0['words'] = words

    return ocr_format


def get_gt(image, ocr, item):

    convertor = OcrAnnotationFusion()
    gt_dict = convertor.convert_file(image,ocr,item)
    return gt_dict

