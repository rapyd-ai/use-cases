import requests
import json
import cv2
from collections import Counter
from operator import itemgetter
from output_layer import cv2_write_image
import pandas as pd
import numpy as np  

# Vision API - Text service
def text_extract_detections(response_json):
    try:
        result = json.loads(response_json.text.encode('utf8'))['result']
        provider = json.loads(response_json.text.encode('utf8'))['meta']['provider']
    except:
        print(response_json.text.encode('utf8'))
    
    if provider == "aws":
        json_data = {
            'meta': json.loads(response_json.text.encode('utf8'))['meta'],
        }
        textDetection = result["textDetections"]
        json_data['textDetections'] = textDetection
        response_json = json.dumps(json_data)
    
    if provider == "gcp":
        json_data = {
            'meta': json.loads(response_json.text.encode('utf8'))['meta'],
        }
        json_data['textDetections'] = result
        response_json = json.dumps(json_data)
    
    if provider == "azure":
        json_data = {
            'meta': json.loads(response_json.text.encode('utf8'))['meta'],
        }
        textDetection = result['regions']
        json_data['textDetections'] = textDetection
        response_json = json.dumps(json_data)

    return response_json

def text_extract_labels(response_json):
    provider = json.loads(response_json)['meta']['provider']
    textDetections = json.loads(response_json)['textDetections']

    if provider == "aws":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        text = [item["detectedText"] for item in textDetections]
        json_data['labels'] = text
        response_json = json.dumps(json_data)
    
    if provider == "gcp":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        text = [item["description"] for item in textDetections]
        json_data['labels'] = text
        response_json = json.dumps(json_data)
    
    if provider == "azure":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in textDetections:
            for lines in item['lines']:
                words = lines["words"]
                text = [value['text'] for value in words]
        json_data['labels'] = text
        response_json = json.dumps(json_data)
        
    return response_json

def text_bounding_boxes(response_json, img_width, img_height):
    provider = json.loads(response_json)['meta']['provider']
    result = json.loads(response_json)['textDetections']
    boxes = []

    if provider == "gcp":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in result:
            rectangles = item["boundingPoly"]
            y1 = rectangles[3]['y']
            y2 = rectangles[1]['y']
            x1 = rectangles[3]['x']
            x2 = rectangles[1]['x']
            bounding_box = {
                'y1': y1,
                'y2': y2,
                'x1': x1,
                'x2': x2
            }
            boxes.append(bounding_box)

        json_data['bounding_box'] = boxes
        response_json = json.dumps(json_data)
    
    if provider == "aws":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in result:
            rectangles = item["geometry"]["boundingBox"]
            bounding_box = {
                'y1': round(rectangles['top'] * img_height),
                'y2': round((rectangles['top'] * img_height) + (rectangles['height'] * img_height)),
                'x1': round(rectangles['left'] * img_width),
                'x2': round((rectangles['left'] * img_width) + (rectangles['width'] * img_width))
            }
            boxes.append(bounding_box)

        json_data['bounding_box'] = boxes
        response_json = json.dumps(json_data)
    
    if provider == "azure":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in result:
            for lines in item['lines']:
                words = lines['words']
                values = [value["boundingBox"] for value in words]
                bounds = [list(each.split(",")) for each in values]
                for rectangles in bounds:
                    bounding_box = {
                        'y1': int(rectangles[1]),
                        'y2': int(rectangles[3]),
                        'x1': int(rectangles[0]),
                        'x2': int(rectangles[2])
                    }
                    boxes.append(bounding_box)
        json_data['bounding_box'] = boxes
        response_json = json.dumps(json_data)

    return response_json

def text_filter_entity(labels, word_list):
    pd.options.mode.chained_assignment = None # To avoid the warnings
    # NOTE: ---------------------------------
    # labels parameter must be the return value of text_extract_labels() function
    # ---------------------------------------
    input_df = pd.DataFrame()
    input_df['labels'] = json.loads(labels)['labels']

    words=(map(lambda x: [x.capitalize(), x.lower(), x.upper()], word_list)) # Append the list with the words that might help refine the search for exipry date
    terms = []
    [terms.extend(i) for i in list(words)]

    # Matching regex pattern for the words we want to find
    match_pattern = fr"\b(?:{'|'.join(terms)})\b"
    return_df = input_df['labels'].str.contains(match_pattern, case=False)

    return return_df
    
# Vision API - localize service
def localize_filter_label(response_json, label):
    try:
        result = json.loads(response_json.text.encode('utf8'))['result']
        provider = json.loads(response_json.text.encode('utf8'))['meta']['provider']
    except:
        print(response_json.text.encode('utf8'))
    
    if label == None:
        json_data = {
            'meta': json.loads(response_json.text.encode('utf8'))['meta'],
        }
        if provider == "aws":
            json_data['labels'] = result['labels']
        if provider == "gcp":
            json_data['labels'] = result
        if provider == "azure":
            json_data['labels'] = result['objects']
        response_json = json.dumps(json_data)

    else:
        if provider == "aws":
            json_data = {
                'meta': json.loads(response_json.text.encode('utf8'))['meta']
            }
            labels = result['labels']
            value = [item  for item in labels if (item['name'] == label.capitalize() or item['name'] == label.lower() or item['name'] == label.upper())]
            json_data['labels'] = value
            response_json = json.dumps(json_data)

        if provider == "gcp":
            json_data = {
                'meta': json.loads(response_json.text.encode('utf8'))['meta']
            }
            value = [item for item in result if (item['description'] == label.capitalize() or item['description'] == label.lower() or item['description'] == label.upper())]
            json_data['labels'] = value
            response_json = json.dumps(json_data)
        
        if provider == "azure":
            json_data = {
                'meta': json.loads(response_json.text.encode('utf8'))['meta']
            }
            labels = result['objects']
            value = [item  for item in labels if (item['object'] == label.capitalize() or item['object'] == label.lower() or item['object'] == label.upper())]
            json_data['labels'] = value
            response_json = json.dumps(json_data)

        if provider == "auto":
            print("This AI task is not supported by the service provider. Please choose another one.")
            return None
    
    return response_json

def localize_filter_confidence(response_json, confidence):
    provider = json.loads(response_json)['meta']['provider']
    labels = json.loads(response_json)['labels']

    if provider == "azure":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        value = [item for item in labels if item['confidence'] > confidence]
        json_data['labels'] = value
        response_json = json.dumps(json_data)

    if provider == "aws":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in labels:
            instances = item['instances']
            item['instances'] = list(filter(None, [obj for obj in instances if (obj['confidence'] > confidence*100)]))
        json_data['labels'] = labels
        response_json = json.dumps(json_data)

    if provider == "gcp":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        value = [item for item in labels if item['score'] > confidence]
        json_data['labels'] = value
        response_json = json.dumps(json_data)

    return response_json

def localize_bounding_boxes(response_json, img_width, img_height):
    provider = json.loads(response_json)['meta']['provider']
    labels = json.loads(response_json)['labels']
    boxes = []

    if provider == "azure":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in labels:
            rectangles = item['rectangle']
            y1 = rectangles['y']
            y2 = rectangles['y'] + rectangles['h']
            x1 = rectangles['x']
            x2 = rectangles['x'] + rectangles['w']
            bounding_box = {
                'y1': y1,
                'y2': y2,
                'x1': x1,
                'x2': x2
            }
            boxes.append(bounding_box)

        json_data['bounding_box'] = boxes
        response_json = json.dumps(json_data)  
    
    if provider == "gcp":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in labels:
            rectangles = item['normalizedBoundingPoly']
            y1 = int(min([value['y'] for value in rectangles])*img_height)
            y2 = int(max([value['y'] for value in rectangles])*img_height)
            x1 = int(min([value['x'] for value in rectangles])*img_width)
            x2 = int(max([value['x'] for value in rectangles])*img_width)
            bounding_box = {
                'y1': y1,
                'y2': y2,
                'x1': x1,
                'x2': x2
            }
            boxes.append(bounding_box)

        json_data['bounding_box'] = boxes
        response_json = json.dumps(json_data)
    
    if provider == "aws":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in labels:
            instances = item['instances']
            for obj in instances:
                rectangles = obj['boundingBox']
                bounding_box = {
                    'y1': round(rectangles['top'] * img_height),
                    'y2': round((rectangles['top'] * img_height) + (rectangles['height'] * img_height)),
                    'x1': round(rectangles['left'] * img_width),
                    'x2': round((rectangles['left'] * img_width) + (rectangles['width'] * img_width))
                }
                boxes.append(bounding_box)
        json_data['bounding_box'] = boxes
        response_json = json.dumps(json_data)
    
    return response_json

def localize_aggregate(response_json, aggregate_function):
    provider = json.loads(response_json)['meta']['provider']
    labels = json.loads(response_json)['labels']

    if aggregate_function == "count":
        if provider == "azure":
            count = Counter(map(itemgetter("object"), labels))
            aggregate_response = dict(count)

        if provider == "aws":
            aggregate_response = {}
            for item in labels:
                if len(item['instances']) != 0:
                    aggregate_response[item['name']] = len(item['instances'])
                else:
                    aggregate_response[item['name']] = 1
        
        if provider == "gcp":
            count = Counter(map(itemgetter("description"), labels))
            aggregate_response = dict(count)

    if aggregate_function == "max_conf":
        aggregate_response = None

    if aggregate_function == "min_conf":
        aggregate_response = None

    return aggregate_response

# Vision API - landmark service
def landmark_bounding_boxes(response_json):
    provider = json.loads(response_json)['meta']['provider']
    result = json.loads(response_json)['result']
    boxes = []

    if provider == "gcp":
        json_data = {
            'meta': json.loads(response_json)['meta']
        }
        for item in result:
            description = item['description'] 
            rectangles = item["boundingPoly"]
            y1 = rectangles[3]['y']
            y2 = rectangles[1]['y']
            x1 = rectangles[3]['x']
            x2 = rectangles[1]['x']
            bounding_box = {
                'y1': y1,
                'y2': y2,
                'x1': x1,
                'x2': x2
            }
            boxes.append(bounding_box)

        json_data['bounding_box'] = boxes
        response_json = json.dumps(json_data)
    
    return response_json

# NLP API - entities service

def entities_extract_type(response_json, entity_type):
    try:
        result = json.loads(response_json.text.encode('utf8'))['result']
        provider = json.loads(response_json.text.encode('utf8'))['meta']['provider']
    except:
        print(response_json.text.encode('utf8'))
    
    if provider == "aws":
        json_data = {
            'meta': json.loads(response_json.text.encode('utf8'))['meta'],
        }
        entities = result['entities']
        types = [value["text"] for value in entities if (value["type"] == entity_type.upper() or value["type"] == entity_type.lower() or value["type"] == entity_type.capitalize())]
        json_data['types'] = types
        response_json = json.dumps(json_data)
    
    if provider == "gcp":
        json_data = {
            'meta': json.loads(response_json.text.encode('utf8'))['meta'],
        }
        types = [value["name"] for value in result if (value["type"] == entity_type.upper() or value["type"] == entity_type.lower() or value["type"] == entity_type.capitalize())]
        json_data['types'] = types
        response_json = json.dumps(json_data)
    
    if provider == "azure":
        json_data = {
            'meta': json.loads(response_json.text.encode('utf8'))['meta'],
        }
        types = [value["text"] for value in result if (value["category"] == entity_type.upper() or value["category"] == entity_type.lower() or value["category"] == entity_type.capitalize())]
        json_data['types'] = types
        response_json = json.dumps(json_data)

    return response_json


def cv2_transform_image(filepath, bounding_boxes, output_file, transformation):
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    
    # Defining Box params
    color = (0, 0, 255)
    thickness = 2

    bounding_boxes = json.loads(bounding_boxes)["bounding_box"]
    if (bounding_boxes == [] or None):
        pass
    else:
        for value in bounding_boxes:
            y1 = value['y1']
            y2 = value['y2']
            x1 = value['x1']
            x2 = value['x2']
            final = cv2.rectangle(image, (x1, y1), (x2,y2), color, thickness)
            
        cv2_write_image(final, output_file)

