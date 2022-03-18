# AI请求格式和数据返回格式

## 人体检测&安全帽检测&人头检测

请求:  
格式：http

```python
payload = json.dumps({
    "img": ['image base64', 'image base64', ]
})  # 无头标签
headers = {
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
```

服务器返回  
格式：json

```json
{
  "code": 200,
  "data": [
    {
      "height": int
      "label": str,
      "left": int,
      "score": float,
      "top": int,
      "width": int
    }
  ],
  "message": "success"
}
```

## 越界检测

请求  
格式：http

```python
 payload = json.dumps({
    "img": ['image_base64', 'image_base64', ...],  # 不包含图片头
    "polys": [[(int, int), (int, int), (int, int), ...], [...]]  # 多边形矩形坐标list
})
headers = {
    'Content-Type': 'application/json'
}
response = requests.request("POST", _Url.area, headers=headers, data=payload)
```

返回  
格式:json

```json
{
  'code': 200,
  'data': [
    {
      'height': int,
      'label': str,
      'left': int,
      'score': float,
      'top': int,
      'width': int
    },
    {
      ...
    }
  ],
  'message': 'success'
}

```