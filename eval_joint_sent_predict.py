#!/usr/bin/env python

"""
Evaluation Tool for Predicting Categories

For usage, see help: `python %(program)s -h`
"""

import sys, os
sys.path.append("../oss")
from sent2vec import Sentence2Vec, REAL, nearest_sent_fast
from sentences import CatSentence
import logging
import utils
import matutils
from threading import Thread
from Queue import Queue
import numpy as np
import time,re
import cPickle as pickle
from argparse import ArgumentParser
from multiprocessing import cpu_count

logger = logging.getLogger("join_sent_eval")

def readSentence(sent):
    sent_cat = {}
    for tpl in sent:
        sent_id = tpl[1]
        cat_id = tpl[2]
        sent_cat[sent_id] = cat_id
    return sent_cat

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(relativeCreated)d : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    parser = ArgumentParser(description="Evaluation tool for joint paragraph vector models")
    parser.add_argument("--split", dest="split", action="store_true", help="use this option for split training data", default=False)
    parser.add_argument("--modelfile1", dest="modelfile1", type=str, help="trained model file 1")
    parser.add_argument("--modelfile2", dest="modelfile2", type=str, help="trained model file 2")
    parser.add_argument("--test", dest="test", type=str, help="test file")
    parser.add_argument("--train", dest="train", type=str, help="input file")
    parser.set_defaults(maxn=sys.maxint)
    parser.add_argument("--maxN", dest="maxn", type=int, help="")
    parser.set_defaults(thread=cpu_count())
    parser.add_argument("-t", "--thread", dest="thread", type=int, help="the number of threads")
    parser.set_defaults(knn=1)
    parser.add_argument("-k","--knn", dest="knn", type=int, help="use k of the nearest neighbors (default 1)")
    args = parser.parse_args()
    test_file = args.test
    topK = args.knn
    maxN = args.maxn
    if not args.modelfile1 or not args.modelfile2:
        print "Specify modelfile1 and modelfile2"
        quit(-1)

    logging.info("load trained model file")
    modelfile1 = args.modelfile1
    model1 = Sentence2Vec.load(modelfile1)
    modelfile2 = args.modelfile2
    model2 = Sentence2Vec.load(modelfile2)
    
    sent_cat = readSentence(CatSentence(args.train, cont_col=3, split=args.split))
    test_sentences = CatSentence(test_file)
    confusion_mtx = {}
    def prepare_sentences():
        count = 0
        for sent_tuple in test_sentences:
            yield sent_tuple
            count += 1
            if count > maxN: break

    def worker_infer():
        while True:
            job = jobs.get()
            if job is None:
                break
            diff = 0.
            work = np.zeros(model1.layer1_size, dtype=REAL)
            neu1 = matutils.zeros_aligned(model1.layer1_size, dtype=REAL)
            for sent_tuple in job:
                cat_id_gold = sent_tuple[2]
                sent_vec1 = model1.train_single_sent_id([sent_tuple[0]], 20, work, neu1)
                sims1 = np.empty(model1.sents_len, dtype=REAL)
                nearest_sent_fast(model1, sent_vec1, 0, sims1)
                sent_vec2 = model2.train_single_sent_id([sent_tuple[0]], 20, work, neu1)
                sims2 = np.empty(model2.sents_len, dtype=REAL)
                nearest_sent_fast(model2, sent_vec2, 0, sims2)
                sims1 += sims2
                neighbors = np.argsort(sims1)[::-1]
                cat_ids = {}
                nearest = []
                ident_cat = True
                for top_cand in neighbors:
                    sent_id = model1.sent_id_list[top_cand]
                    cat_id = sent_cat[sent_id]
                    if not ident_cat or not cat_ids.has_key(cat_id):
                        cat_ids[cat_id] = 1
                        nearest.append(cat_id)
                        if len(nearest) == topK: break
                diff += 1. if cat_id_gold in nearest else 0.
                print nearest,cat_id_gold
                confusion_mtx.setdefault(cat_id_gold, {})
                confusion_mtx[cat_id_gold].setdefault(nearest[0], 0)
                confusion_mtx[cat_id_gold][nearest[0]] += 1
            qout.put(diff)
    
    jobs = Queue(maxsize=50)
    qout = Queue(maxsize=20000)
    threads = [Thread(target=worker_infer) for _ in xrange(args.thread)]
    sent_num = 0
    for t in threads:
        t.daemon = True
        t.start()
    
    for job_no, job in enumerate(utils.grouper(prepare_sentences(), 100)):
        logger.info("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
        jobs.put(job)
        sent_num += len(job)
    logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())

    for _ in xrange(args.thread):
        jobs.put(None)
    for t in threads:
        t.join()

    avg = 0.0
    while not qout.empty():
        val = qout.get()
        avg += val
    avg /= sent_num
    print avg
    info_file = open("result_joint_cat_eval.txt","a")
    info_file.write("infer_cat_k%d\t%f\tm1:%s\tm2:%s\n" % (topK, avg, modelfile1, modelfile2))
    info_file.close()
    
    pickle.dump(confusion_mtx, open("joint_"+model1.identifier()+"_"+model2.identifier()+".cmat", "w"))

    program = os.path.basename(sys.argv[0])
    logging.info("finished running %s" % program)
