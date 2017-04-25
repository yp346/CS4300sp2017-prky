from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from .test import tfidf_sim
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# Create your views here.
def index(request):
    output_list = ''
    output=''
    search=''
    version="First"
    zipped=[]
    if request.GET.get('search'):
        search = request.GET.get('search')
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
    version = request.GET.get('change')
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
