# Core Packages
import numpy as np
import pandas as pd
from gtts import gTTS
import joblib
import sklearn

normal = []
with open('corpus/nonhate.txt', encoding="utf8") as f:
    for line in f:
        normal.append(line[:-1])

offensive = []
with open('corpus/hate.txt', encoding="utf8") as f:
    for line in f:
        offensive.append(line[:-1])

data = {
  'good': normal,
  'bad': offensive,
}

df = pd.read_csv('corpus/Mental_Health.csv')

tfidf = joblib.load('models/vectorizer.pkl')
model = joblib.load('models/model.pkl')

categories = {word: key for key, words in data.items() for word in words}

embeddings_index = {}
with open('embeddings/glove.6B.300d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        embed = np.array(values[1:], dtype=np.float32)
        embeddings_index[word] = embed

data_embeddings = {key: value for key, value in embeddings_index.items() if key in categories.keys()}

def calc_thresold(query):
    query_embed = embeddings_index[query]
    scores = {}
    for word, embed in data_embeddings.items():
        category = categories[word]
        dist = query_embed.dot(embed)
        dist /= len(data[category])
        scores[category] = scores.get(category, 0) + dist
    return scores

def cosine_similarity(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    s2 = s2[:-1]
    v1 = np.mean([embeddings_index[word] for word in s1.split() if word in embeddings_index.keys()], axis=0)
    v2 = np.mean([embeddings_index[word] for word in s2.split() if word in embeddings_index.keys()], axis=0)
    cosine = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cosine

def match_answer(s):
    df['Similarity'] = df.apply(lambda x: cosine_similarity(s, x['Questions']), axis = 1)
    ans = df[df.Similarity == df.Similarity.max()]['Answers'].iloc[0]
    myobj = gTTS(text=ans, lang='en', slow=False)
    myobj.save("audio.mp3")
    return ans

def get_prediction(s):
    text = tfidf.transform([s])
    return model.predict_proba(text)
