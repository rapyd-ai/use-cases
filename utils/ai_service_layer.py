import requests
import json

def rapydai_vision(filepath, task, provider = "gcp", accountid = '', token = ''):
    url = "https://api.rapyd.ai/v1/vision/"+task
    payload = {}
    files = [('file', open(filepath, 'rb'))]
    headers = {
        'PROVIDER': provider,
        'ACCOUNT-ID': accountid,
        'Authorization': 'Bearer '+ token
    }
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return response
    
def rapydai_nlp(text, task, provider = "gcp", accountid = '', token = ''):
    url = "https://api.rapyd.ai/v1/nlp/"+task
    headers = {
        'ACCOUNT-ID': 'your-accountid',
        'Authorization': 'Bearer your-token',
        'Content-Type': 'application/json'
    }
    payload = {
        "text": text,
        "provider": provider,
        "language": "auto"
    }
    response = requests.request("POST", url, headers=headers, json = payload)
    return response

def rapydai_vision_result(filepath, task, provider = "gcp", accountid = '', token = ''):
    url = "https://api.rapyd.ai/v1/vision/"+task
    payload = {}
    files = [('file', open(filepath, 'rb'))]
    headers = {
        'PROVIDER': provider,
        'ACCOUNT-ID': accountid,
        'Authorization': 'Bearer '+ token
    }
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    result = json.loads(response.text.encode('utf8'))['result']
    return result

def rapydai_nlp_result(text, task, provider = "gcp", accountid = '', token = ''):
    url = "https://api.rapyd.ai/v1/nlp/"+task
    headers = {
        'ACCOUNT-ID': 'your-accountid',
        'Authorization': 'Bearer your-token',
        'Content-Type': 'application/json'
    }
    payload = {
        "text": text,
        "provider": provider,
        "language": "auto"
    }
    response = requests.request("POST", url, headers=headers, json = payload)
    result = json.loads(response.text.encode('utf8'))['result']
    return result
