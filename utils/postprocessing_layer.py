import requests
import json
import cv2
from preprocessing_layer import PreprocessingLayer

class PostprocessingLayer:

    def localize_filter_label(self, response_json, label):
        try:
            result = json.loads(response_json.text.encode('utf8'))['result']
            provider = json.loads(response_json.text.encode('utf8'))['meta']['provider']
        except:
            print(response_json.text.encode('utf8'))
        
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
    
    # Filter based on confidence
    def localize_filter_confidence(self, response_json, confidence):
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
            value = []
            for item in labels:
                instances = item['instances']
                value = [obj for obj in instances if (obj['confidence'] > confidence*100)]
            json_data['labels'] = value
            response_json = json.dumps(json_data)
        
        if provider == "gcp":
             json_data = {
                 'meta': json.loads(response_json)['meta']
             }
             value = [item for item in labels if item['score'] > confidence]
             json_data['labels'] = value
             response_json = json.dumps(json_data)

        return response_json

        # Extract bounding boxes
        def localize_bounding_boxes(self, response_json, img_width, img_height):
            provider = json.loads(response_json)['meta']['provider']
            labels = json.loads(response_json)['labels']
            boxes = []

            if provider == "azure":
                json_data = {
                    'meta': json.loads(response_json)['meta']
                }
                for item in labels:
                    rectangle = item['rectangle']
                    y1 = rectangle['y']
                    y2 = rectangle['y'] + rectangle['h']
                    x1 = rectangle['x']
                    x2 = rectangle['x'] + rectangle['w']
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
                    rectangles = item['boundingBox']
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

            # Aggregate functions - count, min_conf, max_conf
            def localize_aggregate(self, response_json, aggregate_function):
                # provider = json.loads(response_json)['meta']['provider']
                labels = json.loads(response_json)['labels']

                if aggregate_function == "count":
                    aggregate_value = len(labels)
  
                if aggregate_function == "max_conf":
                    aggregate_value = None
  
                if aggregate_function == "min_conf":
                    aggregate_value = None
  
                return aggregate_value
            
            # Transform the image - draw boxes, any other
            def cv2_transform_image(self, filepath, bounding_boxes, transformation):
                image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

                # defining the box parameters
                color = (0, 0, 255)  
                thickness = 2

                bounding_boxes = json.loads(bounding_boxes)["bounding_box"]
                for value in bounding_boxes:
                    y1 = value['y1']
                    y2 = value['y2']
                    x1 = value['x1']
                    x2 = value['x2']
                    final = cv2.rectangle(image, (x1, y1), (x2,y2), color, thickness)

                PreprocessingLayer.cv2_write_image(final, 'output.jpg') 