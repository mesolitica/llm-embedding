from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import KDTree
import orjson as json
import random
import numpy as np
import pickle
import os
import mp

class Pointer:
    def __init__(self, filename):
        self.filename = filename
        self.index = -1

    def _save(self):
        with open(self.filename, 'wb') as fopen:
            pickle.dump(self.index, fopen)

    def increment(self):
        self.index += 1
        self._save()

    def load(self):
        if not os.path.exists(self.filename):
            return
        with open(self.filename, 'rb') as fopen:
            self.index = pickle.load(fopen)

def dedup(strings):
    unique_neg = []
    elements = set()

    for n in strings:
        x_lower = n.lower()
        if x_lower not in elements:
            elements.add(x_lower)
            unique_neg.append(n)
    return unique_neg

data = []

for f in glob('news-*.jsonl'):
    with open(f) as fopen:
        for x in tqdm(fopen):
            try:
                data.append(json.loads(x))
            except:
                pass

vectors, texts = [], []

for d in data:
    vectors.append(d['v'])
    texts.append(d['text'])

concat = np.array(vectors)
kd_tree = KDTree(concat, leafsize = 40)
os.system('mkdir news-hard')

lower_bound = 0.57
upper_bound = 1.3

def loop(data):
    data, index = data
    filename = f'news-hard/{index}.jsonl'
    fopen = open(filename, 'a')
    pointer = Pointer(f'{filename}.pickle')
    pointer.load()
    for n in tqdm(range(len(data))):
        x = data[n]
        if n > pointer.index:
            dist, ind = kd_tree.query(concat[x], k=len(concat))

            query = texts[x]

            pos_indices = [k for k in ind[dist < lower_bound]]
            neg_indices = [k for k in ind[dist > upper_bound]]
            
            if len(pos_indices) > 6:
                pos_indices = random.sample(pos_indices,6)
            if len(neg_indices) > 5:
                neg_indices = random.sample(neg_indices,5)

            pos = [texts[i] for i in pos_indices if texts[i] != query and len(texts[i]) > 1]
            pos = dedup(pos)

            if len(pos) == 0:
                continue

            neg = [texts[i] for i in neg_indices if texts[i] != query and len(texts[i]) > 1]
            neg = dedup(neg)

            if len(neg) == 0:
                continue


            d = {'query':query,'pos':pos,'neg':neg}
            fopen.write(f'{json.dumps(d).decode()}\n')
            fopen.flush()
            pointer.index = n
            pointer._save()

mp.multiprocessing(range(len(data)), loop, cores = 30, returned = False)