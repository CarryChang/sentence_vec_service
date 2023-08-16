from sentence_transformers import SentenceTransformer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sentences = ["问一下怎么调节动能回收", "电动车的能量回收怎么优化"]
llm_path = 'BAAI/bge-small-zh'

model = SentenceTransformer(llm_path)
embeddings_1 = model.encode(sentences, normalize_embeddings=True)
embeddings_2 = model.encode(sentences, normalize_embeddings=True)
similarity = embeddings_1.astype('float16') @ embeddings_2.T.astype('float16')
print(similarity)



from FlagEmbedding import FlagModel

model = FlagModel(llm_path, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
embeddings_1 = model.encode(sentences)
embeddings_2 = model.encode(sentences)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)