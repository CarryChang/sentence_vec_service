# -*- coding: utf-8 -*-


import uvicorn
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware

from loguru import logger
from pydantic import BaseModel, Field
import numpy as np
from FlagEmbedding import FlagModel as SentenceModel


class Item(BaseModel):
    input_text: str = Field(..., max_length=512)


s_model = SentenceModel('BAAI/bge-small-zh', query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
s_model.encode(["样例数据-1"])
print('model warm up')

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.post('/emb')
async def emb(item: Item):
    embeddings = s_model.encode(item.input_text)
    # result_dict = {'emb': embeddings.tolist()}
    result_dict = {'emb': embeddings.shape}
    return result_dict


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
