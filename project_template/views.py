from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from .test import tfidf_sim
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import nltk

# Create your views here.
def index(request):
    output_list = ''
    output=''
    search=''
    version="First"
    zipped=[]
    version = request.GET.get('change')
    if request.GET.get('search'):
    	if request.GET.get(version == 'First'):
        	search = request.GET.get('search')
        	output_list = find_similar(search)
        	output = tfidf_sim(search)
        	search = search.split(",")
        	zipped = zip(search, output)
        else:
        	patterns = """
        				NP: {<NNP>+}
    					{<DT|PP\$>?<JJ>*<NN*>+}
    					"""
    		NPChunker = nltk.RegexpParser(patterns)
        	search = request.GET.get('search')
        	sentences = nltk.sent_tokenize(search)
    		sentences = [nltk.word_tokenize(sent) for sent in sentences]
    		sentences = [nltk.pos_tag(sent) for sent in sentences]
    		sentences = [NPChunker.parse(sent) for sent in sentences]
    		nps = []
    		for tree in sentences:
        		for subtree in tree.subtrees():
        			if subtree.label() == 'NP':
        				t = subtree
        				t = ' '.join(word for word, tag in t.leaves())
        				nps.append(t)
        	search = ",".join(nps)
        	output_list = find_similar(search)
        	output = tfidf_sim(search)
        	search = search.split(",")
        	zipped = zip(search, output)
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
			       'version': version,
                               'magic_url': request.get_full_path(),
                               })
    '''elif request.GET.get('version') == "Second":
         return render_to_response('project_template/index_proto2.html',
	                        {'output': output,
				'search': search,
				'zipped': zipped,
				'magic_url': request.get_full_path(),
				})
    elif request.GET.get('version') == "Final":
        return render_to_response('project_template/index_proto3.html',
	                         {'output': output,
				'search': search,
				'zipped': zipped,
		 		'magic_url': request.get_full_path(),
				})'''
