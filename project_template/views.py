from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from .test import tfidf_sim
from .test import tfidf_sentiment_sim
from .test import tfidf_sentiment_sim_weighted
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import nltk

# Create your views here.
def index(request):
    output_list = ''
    output=''
    search=''
    original_search=""
    zipped=[]
    second_select = "False"
    version = request.GET.get('change')
    known_courses = ""
    #version = request.GET.get('submit_button')
    if request.GET.get('known_courses'):
        known_courses1 = request.GET.getlist('known_courses')
        known_courses = ",".join(known_courses1)
    if request.GET.get('search'):
    	original_search = request.GET.get('search')
    	if version == 'First':
    		search = request.GET.get('search')
    		output_list = find_similar(search)
    		output = tfidf_sim(search)
    		search = search.split(",")
    		zipped = zip(search, output)
        elif version == 'Second':
        	patterns = """
        				NP: {<NNP>+}
    					{<DT|PP\$>?<JJ>*<NN*>+}
    					"""
    		NPChunker = nltk.RegexpParser(patterns)
        	search = request.GET.get('search')
        	words = nltk.word_tokenize(search)
        	tags = nltk.pos_tag(words)
        	tree = NPChunker.parse(tags)
    		nps = []
    		for subtree in tree.subtrees():
    			if subtree.label() == 'NP':
    				t = subtree
    				t = ' '.join(word for word, tag in t.leaves())
    				nps.append(t)
        	#search = ",".join(nps)
        	#output_list = find_similar(search)
        	if known_courses!="":
		    	output = tfidf_sentiment_sim(known_courses)
		    	zipped = zip(known_courses1,output)
		    	second_select = "False"
		else:
		    	output = "s"*len(nps)
        		#search = search.split(",")
        	    	#zipped = zip(search, output)
        	    	zipped = zip(nps,output)
			second_select = "True"
	else:
        	patterns = """
        				NP: {<NNP>+}
    					{<DT|PP\$>?<JJ>*<NN*>+}
    					"""
    		NPChunker = nltk.RegexpParser(patterns)
        	search = request.GET.get('search')
        	words = nltk.word_tokenize(search.lower())
        	tags = nltk.pos_tag(words)
        	tree = NPChunker.parse(tags)
    		nps = []
    		for subtree in tree.subtrees():
    			if subtree.label() == 'NP':
    				t = subtree
    				t = ' '.join(word for word, tag in t.leaves())
				if t not in nps:
    					nps.append(t)
        	#search = ",".join(nps)
        	#output_list = find_similar(search)
        	if known_courses!="":
		    	output = tfidf_sentiment_sim_weighted(known_courses)
		    	zipped = zip(known_courses1,output)
		    	second_select = "False"
		else:
		    	output = "s"*len(nps)
        		#search = search.split(",")
        	    	#zipped = zip(search, output)
        	    	zipped = zip(nps,output)
			second_select = "True"
        #paginator = Paginator(output_list1, 4)
        #page = request.GET.get('page')
        #try:
        #    output = paginator.page(page)
        #except PageNotAnInteger:
        #    output = paginator.page(1)
        #except EmptyPage:
        #    output = paginator.page(paginator.num_pages)
    #if request.GET.get('version') == "First":
    
    return render_to_response('project_template/index.html', 
                              {'output': output,
                               'search': search,
                               'zipped': zipped,
			       'second_select': second_select,
			       'known_courses': known_courses,
			       'original_search':original_search,
			       'version': version,
                               'magic_url': request.get_full_path(),
                               })
