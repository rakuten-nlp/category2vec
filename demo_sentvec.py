#!/usr/bin/env python

import sys, os
import urllib
from sent2vec import Sentence2Vec
from sentences import SampledWikiSentence as WikiSentence
import utils
import logging

jawiki_name = "jawiki.tsv.gz"
jawiki_url = "http://junki.me/misc/jawiki-20141122-pages_plaintext.tsv.gz"
enwiki_name = "enwiki.tsv.gz"
enwiki_url = "http://junki.me/misc/enwiki-20141106-pages_plaintext.tsv.gz"
model_dir = "models"
logger = logging.getLogger("demo")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    wikip_data = current_dir+"/"+enwiki_name
    s2v_model_name = current_dir+"/"+model_dir+"/"+ enwiki_name + "_sent.model"
    if not os.path.exists(current_dir+"/"+model_dir):
        os.mkdir(current_dir+"/"+model_dir)
    if not os.path.isfile(wikip_data):
        logger.info("downloading Wikipedia data")
        urllib.urlretrieve(enwiki_url, wikip_data)
        logger.info("downloaded in %s" % wikip_data)
    
    sentences = WikiSentence(wikip_data)
    if not os.path.isfile(s2v_model_name):
        model = Sentence2Vec(sentences,iteration=10, model="cb", hs = 1, negative = 0, size=300, update_mode = 0)
        model.save(s2v_model_name)
    else:
        model = Sentence2Vec.load(s2v_model_name)
    
    print "Input an article title (type EXIT to exit)"
    sys.stdout.write("Name: ")
    line = sys.stdin.readline()
    while line:
        line = utils.to_unicode(line.rstrip())
        if line == "EXIT":
            break
        try:
            if model.sent_no_hash.has_key(line):
                sent_no = model.sent_no_hash[line]
                sent_vec = model.sents[sent_no]
                nsents = model.most_similar_sentence(sent_vec, 11)
                print "Similar articles              similarity"
                print "-"*45
                for nsent in nsents[1:]:
                    print nsent[0], " "*(max(30 - len(utils.to_utf8(nsent[0])), 0)), nsent[1]
                print
            else:
                print "we couldn't find the specified category/article"
                print
        except Exception:
            print "something wrong is happened"

        print "Input a category name or an article title (type EXIT to exit)"
        sys.stdout.write("Name: ")
        line = sys.stdin.readline()
