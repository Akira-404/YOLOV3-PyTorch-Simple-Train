import cv2
import time
import requests
import base64
import json

img_path = '../work.jpeg'

while True:
    frame = cv2.imread(img_path)
    image = cv2.imencode('.jpg', frame)[1]
    image_base64 = str(base64.b64encode(image))[2:-1]
    payload = json.dumps({
        "img": [image_base64]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    t1 = time.time()
    # response = requests.request("POST", 'http://192.168.2.7:5000/server', headers=headers, data=payload)
    response = requests.request("POST", 'http://192.168.86.96:5000/server', headers=headers, data=payload)
    # response = requests.request("POST", 'http://192.168.86.73:5000/server', headers=headers, data=payload)
    t2 = time.time()
    print(f'model use time:{round(t2 - t1, 2)}')
    data = response.text
    # data = eval(data)
    print(data)
