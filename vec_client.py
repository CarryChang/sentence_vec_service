import requests
import json

body = {
    'input_text': '你好，我叫张振'
}
import time

st = time.time()
print(requests.post(url='http://127.0.0.1:8001/embeddings_api', data=json.dumps(body)).json())
print('time used: {}'.format(time.time() - st))
