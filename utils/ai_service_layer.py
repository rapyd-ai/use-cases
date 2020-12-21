import requests

class AIService:

    def rapydai_vision(self, filepath, task, provider = "gcp", accountid = '', token = ''):
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
    
    def rapydai_nlp(self, text, task, provider = "gcp", accountid = '', token = ''):
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