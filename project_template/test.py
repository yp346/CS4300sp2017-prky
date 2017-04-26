from .models import Docs
import os
import Levenshtein
import nltk
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def read_file(n):
	path = Docs.objects.get(id = n).address;
	file = open(path)
	transcripts = json.load(file)
	return transcripts

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
    query_list = q.split(",")
    query_by_vocab = new_tfidf.fit_transform(query_list).toarray()
    sim_matrix = np.dot(query_by_vocab,doc_by_vocab.T)
    #sim_matrix = final.flatten()
    sorted_sim = np.argsort(-sim_matrix)
    return_list = []
    for x in sorted_sim:
    	temp = []
    	for y in x[:5]:
        	temp.append((name_list[y],url_list[y]))
        return_list.append(temp)

    return return_list


def tfidf_sentiment_sim(q):
    dir_cur = os.path.dirname(__file__)
    file_path_obj = os.path.join(dir_cur,"../jsons/tfidf_obj.pk")
    file_path_np = os.path.join(dir_cur,"../jsons/tf_idf.npy")
    file_features = os.path.join(dir_cur,"../jsons/feature_names.json")
    course_desc_path = os.path.join(dir_cur,"../jsons/course_desc_list.json")
    course_url_path = os.path.join(dir_cur,"../jsons/course_url_list.json")
    course_name_path = os.path.join(dir_cur,"../jsons/course_name_list.json")
    coursera_senti_score_path = os.path.join(dir_cur,"../jsons/senti_score_dict.json")
    
    with open(coursera_senti_score_path,"r") as fp:
        content = fp.read()
        coursera_senti_dict = json.loads(content.decode("utf-8","ignore"))
    
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
    query_list = q.split(",")
    query_by_vocab = new_tfidf.fit_transform(query_list).toarray()
    sim_matrix = np.dot(query_by_vocab,doc_by_vocab.T)
    #sim_matrix = final.flatten()
    sorted_sim = np.argsort(-sim_matrix)
    sorted_values = np.sort(sim_matrix)
    sorted_values = np.fliplr(sorted_values)

    return_list = []
    for x in sorted_sim:
        temp = []
        for y in x[:10]:
            temp.append((name_list[y],url_list[y]))
        return_list.append(temp)

    senti_score_list = []
    for skill in return_list:
        temp_senti = []
        for t in skill:
            temp_senti.append((coursera_senti_dict[t[1]],t[0],t[1]))
        senti_score_list.append(temp_senti)

    final_return_list = []
    for skill in senti_score_list:
        skill.sort(reverse = True)
        temp = []
        for t in skill[:5]:
            temp.append((str(t[0]),t[1],t[2]))
        final_return_list.append(temp)

    return final_return_list




