from .models import Docs
import os
import Levenshtein
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def read_file(n):
	path = Docs.objects.get(id = n).address;
	file = open(path)
	transcripts = json.load(file)
	return transcripts

def read_tfidf_file(n):
        path = Docs.objects.get(id=n).address;
	return path
	file = open(path)
	vectorizer = pickle.load(file)
	return vectorizer

def read_tfidf_numpy(n):
        path = Docs.objects.get(id=n).address;
	file = open(path)
        tfidf_numpy = np.load(file)
	return tfidf_numpy

def _edit(query, msg):
    return Levenshtein.distance(query.lower(), msg.lower())

def find_similar(q):
	transcripts = read_file(1)
	result = []
	for transcript in transcripts:
		for item in transcript:
			m = item['text']
			result.append(((_edit(q, m)), m))

	return sorted(result, key=lambda tup: tup[0])

def tfidf_sim(q):
    dir_cur = os.path.dirname(__file__)
    file_path_obj = os.path.join(dir_cur,"../jsons/tfidf_obj.pk")
    file_path_np = os.path.join(dir_cur,"../jsons/tf_idf.npy")
    file_features = os.path.join(dir_cur,"../jsons/feature_names.json")
    course_desc_path = os.path.join(dir_cur,"../jsons/course_desc_list.json")
    course_url_path = os.path.join(dir_cur,"../jsons/course_url_list.json")
    course_name_path = os.path.join(dir_cur,"../jsons/course_name_list.json")
    with open(file_path_obj,"rb") as fp:
        vectorizer = pickle.load(fp)
    with open(file_path_np,"rb") as fp:
        doc_by_vocab = np.load(fp)
    with open(file_features,"r") as fp:
        content = fp.read()
        features = json.loads(content)#.decode("utf-8","ignore"))
    with open(course_desc_path,"r") as fp:
        content = fp.read()
	desc_list = json.loads(content.decode("utf-8","ignore"))
    with open(course_url_path,"r") as fp:
        content = fp.read()
	url_list = json.loads(content.decode("utf-8","ignore"))
    with open(course_name_path,"r") as fp:
        content = fp.read()
	name_list = json.loads(content.decode("utf-8","ignore"))
    new_tfidf =  TfidfVectorizer(vocabulary=features,max_features=5000, stop_words="english",norm="l2")
    query = new_tfidf.fit_transform([q]).toarray()
    final = np.dot(query,doc_by_vocab.T)
    final = final.flatten()
    sorted_final = np.argsort(final)[::-1]
    ret_list = []
    for x in sorted_final:
        ret_list.append((name_list[x],url_list[x]))
    return ret_list

