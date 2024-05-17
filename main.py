from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import pandas as pd
import numpy as np
import numpy as np
import csv
from tqdm import tqdm
import pymorphy2

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df_qa_pairs = pd.read_excel('QA_pairs.xlsx')
df_knowledge_base = pd.read_excel('knowledge_base.xlsx')

# Берём подготовленные эмбеддинги
df_qa_embeddings = pd.read_csv('qa_embeddings.csv')
df_knowledge_base_embeddings = pd.read_csv("chunk_embeddings.csv")

# Это нужно для лемматизации текста
morph = pymorphy2.MorphAnalyzer()

def lemmatize(text):
    words = text.split() # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return ' '.join(res)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Загружаем модель для эмбеддингов
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large') # passage query
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# Скачиваем стоп-слова из NLTK
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('russian'))


# Инициализируем модель с помощью pipeline из библиотеки transformers
llama_model = "daryl149/llama-2-7b-chat-hf"

llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)

pipeline = transformers.pipeline(
    "text-generation",
    model=llama_model,
    torch_dtype=torch.float16,
    device_map="auto",
)


# Основной вопрос
question = "Здрасьте"

################ Ретривер ищет похожие вопросы из БД по эмбеддингам

text = question.replace("ПВЗ", "пункт выдачи заказов")
text = text.replace("ШК", "штрих код")

tokens = word_tokenize(text)

filtered_tokens = [word for word in tokens if word not in stop_words]
text = " ".join(filtered_tokens)
text = lemmatize(text).replace(" ?", "?")
# print(text)

batch_dict = tokenizer(question, max_length=512, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

nums2 = embeddings.detach().numpy()[0]
cos_sim_list = []

for index, embedding in df_qa_embeddings.iterrows():
  nums1 = list(embedding)

  cosine_similarity = np.dot(nums1, nums2) / (np.linalg.norm(nums1) * np.linalg.norm(nums2)) # cross-encoder
  cos_sim_list.append(cosine_similarity)

cos_sim_list_sort = cos_sim_list.copy()
cos_sim_list_sort.sort(reverse=True)

question_list = []
answer_list = []
for i in range(3):
  index = cos_sim_list.index(cos_sim_list_sort[i])
  question_list.append(df_qa_pairs['question'][index])
  answer_list.append(df_qa_pairs['answer'][index])

############# Ретривер ищет похожие чанки из БД по эмбеддингам

batch_dict = tokenizer(question, max_length=512, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

nums2 = embeddings.detach().numpy()[0]
cos_sim_list = []

for index, embedding in df_knowledge_base_embeddings.iterrows():
  nums1 = list(embedding)

  cosine_similarity = np.dot(nums1, nums2) / (np.linalg.norm(nums1) * np.linalg.norm(nums2)) # cross-encoder
  cos_sim_list.append(cosine_similarity)

cos_sim_list_sort = cos_sim_list.copy()
cos_sim_list_sort.sort(reverse=True)

sentences_list = []
for i in range(3):
  index = cos_sim_list.index(cos_sim_list_sort[i])
  sentences_list.append(df_knowledge_base['chunk'][index])

##############

sequences = pipeline(
    f'Ты - сотрудник службы поддержки Wildberries. Тебе поступил вопрос от клиента, ответь на него на русском языке. Если не знаешь ответа - напиши что ответа нет\nПример похожего вопроса: {question_list[1]}\nОтвет на похожий вопрос:{answer_list[1]}\nПример второго похожего вопроса: {question_list[2]}\nОтвет на второй похожий вопрос:{answer_list[2]}\nДополнительная информация: {sentences_list[0]}\nВопрос: {question}',
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=llama_tokenizer.eos_token_id,
    max_length=700,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")