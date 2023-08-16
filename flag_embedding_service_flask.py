# -*- coding: utf-8 -*-
import json

from flask import Flask, jsonify, request

app = Flask(__name__)

from FlagEmbedding import FlagModel as SentenceModel
llm_path = 'BAAI/bge-small-zh'
s_model = SentenceModel(llm_path, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
s_model.encode(["样例数据-1"])
print('model warm up')


@app.route("/embeddings_api", methods=['POST'])
def embeddings_api():
    input_text = json.loads(request.data)["input_text"].strip()
    embeddings = s_model.encode(input_text)
    return {'emb': embeddings.tolist()}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
