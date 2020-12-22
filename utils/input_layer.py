import requests
import json
from google.colab import files

# Upload from URL
def file_upload_url(URL):
    r = requests.get(URL, allow_redirects=True)
    open('input.jpg', 'wb').write(r.content)
    json_data = {
        'Option': 'Upload from URL',
        'Response': r.status_code,
        'Content': r.url
    }

    return json_data
    
# Upload from Sys
def file_upload_colab():
    json_data = {
        'Option': 'Upload from sys'
    }
    out = json.loads(json_data)
    uploaded = files.upload()
    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
        out.update({'name': fn})
        
    json_data = json.dumps(json_data)
    return json_data
 
