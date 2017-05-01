from .models import Docs
import os
import Levenshtein
import nltk
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import operator

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

def tfidf_sentiment_sim_weighted(q):
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
    #sorted_sim = np.argsort(-sim_matrix)
    #sorted_values = np.sort(sim_matrix)
    #sorted_values = np.fliplr(sorted_values)

    #return_list = []
    #for x in sorted_sim:
    #    temp = []
    #    for y in x[:5]:
    #        temp.append((name_list[y],url_list[y]))
    #    return_list.append(temp)

    senti_score_list = []
    for skill in query_list:
        temp_senti = []
        for url in url_list:
        	if coursera_senti_dict[url]:
            		temp_senti.append(coursera_senti_dict[url])
            	else:
            		temp_senti.append(0.0)
        senti_score_list.append(temp_senti)
        
    senti_score_matrix = np.array(senti_score_list)
    
    final_score_matrix = senti_score_matrix + sim_matrix
    
    sorted_final_score = np.argsort(-final_score_matrix)
    
    return_list = []
    for i in range(sorted_final_score.shape[0]):
        temp = []
        for j in sorted_final_score[i][:5]:
            temp.append((str(round(senti_score_matrix[i][j],3)),name_list[j],url_list[j]))
        return_list.append(temp)

    #final_return_list = []
    #for skill in senti_score_list:
    #    skill.sort(reverse = True)
    #    temp = []
    #    for t in skill[:5]:
    #        temp.append((str(t[0]),t[1],t[2]))
    #    final_return_list.append(temp)

    review_tag_list = get_review_tags(return_list)

    final_list = []

    for i in range(len(return_list)):
        temp = []
        skill_tag_list = review_tag_list[i]
        for j in range(len(return_list[i])):
            t = return_list[i][j]
            temp.append((t[0],t[1],t[2],', '.join(skill_tag_list[j][3:][::-1]),', '.join(skill_tag_list[j][:3][::-1])))
        final_list.append(temp)

    return final_list

def tfidf_sentiment_sim_weighted_2(q):
    dir_cur = os.path.dirname(__file__)
    file_path_obj = os.path.join(dir_cur, "../jsons/tfidf_obj.pk")
    file_path_np = os.path.join(dir_cur, "../jsons/tf_idf.npy")
    file_features = os.path.join(dir_cur, "../jsons/feature_names.json")
    course_desc_path = os.path.join(dir_cur, "../jsons/course_desc_list.json")
    course_url_path = os.path.join(dir_cur, "../jsons/course_url_list.json")
    course_name_path = os.path.join(dir_cur, "../jsons/course_name_list.json")
    coursera_senti_score_path = os.path.join(dir_cur, "../jsons/senti_score_dict.json")

    with open(coursera_senti_score_path, "r") as fp:
        content = fp.read()
        coursera_senti_dict = json.loads(content.decode("utf-8", "ignore"))

    with open(file_path_obj, "rb") as fp:
        vectorizer = pickle.load(fp)
    with open(file_path_np, "rb") as fp:
        doc_by_vocab = np.load(fp)
    with open(file_features, "r") as fp:
        content = fp.read()
        features = json.loads(content)  # .decode("utf-8","ignore"))
    with open(course_desc_path, "r") as fp:
        content = fp.read()
    desc_list = json.loads(content.decode("utf-8", "ignore"))
    with open(course_url_path, "r") as fp:
        content = fp.read()
    url_list = json.loads(content.decode("utf-8", "ignore"))
    with open(course_name_path, "r") as fp:
        content = fp.read()
    name_list = json.loads(content.decode("utf-8", "ignore"))
    new_tfidf = TfidfVectorizer(vocabulary=features, max_features=5000, stop_words="english", norm="l2")
    query_list = q.split(",")
    query_by_vocab = new_tfidf.fit_transform(query_list).toarray()
    sim_matrix = np.dot(query_by_vocab, doc_by_vocab.T)
    # sim_matrix = final.flatten()
    sorted_sim = np.argsort(-sim_matrix)
    sorted_values = np.sort(sim_matrix)
    sorted_values = np.fliplr(sorted_values)

    return_list = []
    for x in sorted_sim:
        temp = []
        for y in x[:5]:
            temp.append((name_list[y], url_list[y]))
        return_list.append(temp)

    senti_score_list = []
    for skill in return_list:
        temp_senti = []
        for t in skill:
            temp_senti.append((coursera_senti_dict[t[1]], t[0], t[1]))
        senti_score_list.append(temp_senti)

    final_return_list = []
    for skill in senti_score_list:
        skill.sort(reverse=True)
        temp = []
        for t in skill[:5]:
            temp.append((str(round(t[0],3)), t[1], t[2]))
        final_return_list.append(temp)

    review_tag_list = get_review_tags(final_return_list)

    final_list = []

    for i in range(len(return_list)):
        temp = []
        skill_tag_list = review_tag_list[i]
        for j in range(len(final_return_list[i])):
            t = final_return_list[i][j]
            temp.append((t[0], t[1], t[2], ', '.join(skill_tag_list[j][3:][::-1]), ', '.join(skill_tag_list[j][:3][::-1])))
        final_list.append(temp)

    return final_list

def get_review_tags(course_list):
    dir_cur = os.path.dirname(__file__)
    review_tag_path = os.path.join(dir_cur, "../jsons/course_review_tags_senti_scores.json")

    with open(review_tag_path,"r") as fp:
        content = fp.read()
        review_tag_dict = json.loads(content.decode("utf-8","ignore"))

    return_tag_list = []
    for skill in course_list:
        skill_tag_list = []
        for course in skill:
            course_url = course[2]
            review_tag_list = review_tag_dict[course_url]
            temp = []
            for tag in review_tag_list:
                temp.append(tag[1])
            skill_tag_list.append(temp)
        return_tag_list.append(skill_tag_list)

    return return_tag_list

def get_overall_courses(course_list):
    course_rank_score = {}
    course_skill_count = {}
    course_total_score = {}
    course_url_name = {}

    for skill in course_list:
        skill_rank = 5
        for course in skill:
            course_name = course[1]
            course_url = course[2]
            skill_rank -= 1
            course_rank_score[course_url] = course_rank_score.get(course_url,0) + skill_rank
            course_skill_count[course_url] = course_skill_count.get(course_url,0) + 1
            course_url_name[course_url] = course_name

    for course_url in course_rank_score.keys():
        course_total_score[course_url] = (course_skill_count[course_url],course_rank_score[course_url])

    sorted_course = sorted(course_total_score.items(), key = operator.itemgetter(1), reverse=True)

    return_list = []

    for course in sorted_course[:5]:
        course_url = course[0]
        course_name = course_url_name[course_url]
        return_list.append((course_name, course_url))

    return return_list



