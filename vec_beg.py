from FlagEmbedding import FlagModel


# 添加句子内容
model = FlagModel('BAAI/bge-small-zh', query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
embeddings_1 = model.encode("样例数据-1")
# batch
# embeddings_1 = model.encode(["样例数据-1"])
import numpy as np

# (1, 512)
import time
print(np.asarray(embeddings_1).shape)


sentences = ["我喜欢这家民宿的床，非常舒服，和家里的感觉一样", "如家酒店的住宿和家里的差距比较大"]

embeddings_1 = model.encode(sentences)
embeddings_2 = model.encode(sentences)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
