#!/usr/bin/env python

"""
Evaluation Tool for Predicting Categories

For usage, see help: `python %(program)s -h`
"""

import sys, os
sys.path.append("../oss")
from cat2vec import Category2Vec, REAL
from sentences import CatSentence
import logging
import utils
import matutils
from threading import Thread
from Queue import Queue
import numpy as np
import time,re
import cPickle as pickle

logger = logging.getLogger("cat_predict_eval")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(relativeCreated)d : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    parser = Category2Vec.arg_parser()
    parser.add_argument("--split", dest="split", action="store_true", help="use this option for split training data", default=False)
    parser.add_argument("--modelfile", dest="modelfile", type=str, help="trained model file")
    parser.add_argument("--test", dest="test", type=str, help="test file")
    parser.set_defaults(maxn=sys.maxint)
    parser.add_argument("--maxN", dest="maxn", type=int, help="")
    parser.set_defaults(knn=1)
    parser.add_argument("-k","--knn", dest="knn", type=int, help="use k of the nearest neighbors (default 1)")
    args = parser.parse_args()
    test_file = args.test
    topK = args.knn
    maxN = args.maxn
    if args.modelfile:
        logging.info("load trained model file")
        modelfile = args.modelfile
        model = Category2Vec.load(modelfile)
    else:
        input_file = args.train[0]
        p_dir = re.compile("^.*/")
        basename = p_dir.sub("",input_file)
        if args.outdir:
            outdir = args.outdir
        else:
            m = p_dir.search(input_file)
            outdir = m.group(0) if m else ""
        if args.split:
            input_file = args.train
        logging.info("train from input file")
        model = Category2Vec(CatSentence(input_file, cont_col=3, split=args.split), iteration=args.iteration, model=args.model, hs = args.hs, negative = args.neg, workers = args.thread, alpha=args.alpha, size=args.dim, update_mode = args.update, normalize_each_epoch = args.norm)
        modelfile = "%s%s_%s.model" % (outdir, basename, model.identifier())
        model.save(modelfile)

    logging.info("initializing pairnorm")
    model.init_pairnorm()
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
            work = matutils.zeros_aligned(model.layer1_size + 8, dtype=REAL)
            neu1 = matutils.zeros_aligned(model.layer1_size + 8, dtype=REAL)
            for sent_tuple in job:
                cat_id = sent_tuple[2]
                ret = model.infer([sent_tuple[0]], iteration=20, k=topK, work=work, neu1=neu1)
                diff += 1. if cat_id in ret[2] else 0.
                print ret[2],cat_id
                confusion_mtx.setdefault(cat_id, {})
                confusion_mtx[cat_id].setdefault(ret[2][0], 0)
                confusion_mtx[cat_id][ret[2][0]] += 1
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
    info_file = open("result_cat_eval.txt","a")
    info_file.write("infer_cat_k%d\t%f\t%s\n" % (topK, avg, modelfile))
    info_file.close()
    
    pickle.dump(confusion_mtx, open(modelfile+".cmat", "w"))

    program = os.path.basename(sys.argv[0])
    logging.info("finished running %s" % program)
