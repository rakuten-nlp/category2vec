#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Rakuten NLP Project
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Distributed representation models via category vector's "CV-DM and CV-DBoW models", using either
hierarchical softmax or negative sampling [1]_ [2]_ [3]_ [4]_.

The algorithms for training word vectors were originally ported from C package https://code.google.com/p/word2vec/
and extended with additional functionality and optimization implemented in Cython [5]_.

.. [1] Junki Marui, and Masato Hagiwara. Category2Vec: 単語・段落・カテゴリに対するベクトル分散表現. 言語処理学会第21回年次大会(NLP2015).
.. [2] Quoc Le, and Tomas Mikolov. Distributed Representations of Sentence and Documents. In Proceedings of ICML 2014.
.. [3] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations 
in Vector Space. In Proceedings of Workshop at ICLR, 2013.                                                    
.. [4] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations 
of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
.. [5] Radim Rehurek, Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""

import logging
import sys
import os
import heapq
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, random, dtype, get_include, amax,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
from numpy.linalg import norm as np_norm

try:
    from sent2vec_bind import train_sent_vec, sentvec_sim, IS_DOUBLE, FAST_VERSION
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    from sent2vec_pp import train_sent_vec, sentvec_sim, IS_DOUBLE, FAST_VERSION

if IS_DOUBLE:
    from numpy import float64 as REAL
else:
    from numpy import float32 as REAL


logger = logging.getLogger("sent2vec")

import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange
from word2vec import Word2Vec, Vocab
from multiprocessing import cpu_count
from argparse import ArgumentParser

SENT2VEC_VERSION = "0.01"

class Sentence2Vec(utils.SaveLoad):
    def __init__(self, sentences, model_file=None, size=200, alpha=0.025, window=5, min_count = 5,
                 sample=0, seed=1, workers=16, min_alpha=0.0001, model="cb", hs=1, negative=0, cbow_mean=0,
                 iteration=1, word_learn=1, init_adjust=True, update_mode = 0, normalize_each_epoch = False):
        self.sg = 1 if model == "sg" or model == "dbow" else 0
        self.table = None # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.sample = sample
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.iteration = iteration
        self.word_learn = int(word_learn)
        self.layer1_size = size
        self.min_count = min_count
        self.sent_no_hash = {} #mapping sent_id to index of self.sents
        self.sent_id_list = [] #mapping sent_no to sent_id
        self.sane_vec_len = 100000 #for sanity check
        self.sane_max_sim10 = 0.9 #for sanity check
        self.init_adjust = init_adjust #for adjustment of initialization
        self.update_mode = update_mode #0:SGD, 1: AdaGrad, 2:AdaDelta, (3:ADAM not implemented)
        self.normalize_each_epoch = normalize_each_epoch

        if sentences:
            if model_file:
                self.w2v = Word2Vec.load(model_file)
                self.vocab = self.w2v.vocab
                self.layer1_size = self.w2v.layer1_size
                self.build_vec(sentences, has_vocab = True)
            else:
                self.word_learn = 1
                self.w2v = Word2Vec(None, self.layer1_size, self.alpha, self.window, self.min_count, self.sample, self.seed, self.workers, self.min_alpha, self.sg, self.hs, self.negative, self.cbow_mean)
                self.build_vec(sentences, has_vocab = False)
            self.train_iteration(sentences, iteration=iteration)

            
    def build_vec(self, sentences, has_vocab = False):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting vectors for sentences")
        if not has_vocab :
            logger.info("build vocabulary and")
        logger.info("resetting vectors")
        random.seed(self.seed)
        sentence_no, vocab = -1, {}
        total_words = 0
        self.sents_len = 0 #the num of sentence ids
        self.total_sents = 0 #the num of sentences
        for sentence_no, sent_tuple in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
            sentence = sent_tuple[0]
            for word in sentence:
                total_words += 1
                if word in vocab:
                    vocab[word].count += 1
                else:
                    vocab[word] = Vocab(count=1)
            sent_id = sent_tuple[1]
            self.total_sents += 1
            if not self.sent_no_hash.has_key(sent_id):
                self.sent_no_hash[sent_id] = self.sents_len
                self.sent_id_list.append(sent_id)
                self.sents_len += 1

        logger.info("collected %i word types from a corpus of %i words and %i sentences(ident:%i)" %
                    (len(vocab), total_words, self.total_sents, self.sents_len))

        self.build_vocab(vocab)
        self.sents = matutils.zeros_aligned((self.sents_len, self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        self.reset_weights()


    def build_vocab(self, vocab):
        # assign a unique index to each word
        self.w2v.vocab, self.w2v.index2word = {}, []
        for word, v in iteritems(vocab):
            if v.count >= self.w2v.min_count:
                v.index = len(self.w2v.vocab)
                self.w2v.index2word.append(word)
                self.w2v.vocab[word] = v
        logger.info("total %i word types after removing those with count<%s" % (len(self.w2v.vocab), self.w2v.min_count))

        if self.hs:
            # add info about each word's Huffman encoding
            self.w2v.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.w2v.make_table()
        # precalculate downsampling thresholds
        self.w2v.precalc_sampling()
        self.w2v.reset_weights()
        self.vocab = self.w2v.vocab
        # initialization adjustment
        if self.init_adjust:
            self.w2v.syn0 *= sqrt(self.layer1_size)
            if self.hs: self.w2v.syn1 *= sqrt(self.layer1_size)
            if self.negative: self.w2v.syn1neg *= sqrt(self.layer1_size)

            
    def reset_weights(self):
        if self.init_adjust:
            denom = sqrt(self.layer1_size)
        else:
            denom = self.layer1_size
        for idx in xrange(self.sents_len):
            self.sents[idx] = (random.rand(self.layer1_size) - 0.5) / denom
        # gradients for vectors
        self.syn0_grad = self.init_grad_weight(len(self.w2v.vocab))
        self.syn1_grad = self.init_grad_weight(len(self.w2v.vocab)) if self.hs > 0 else zeros(0, dtype=REAL)
        self.syn1neg_grad = self.init_grad_weight(len(self.w2v.vocab)) if self.negative > 0 else zeros(0, dtype=REAL)
        self.sents_grad = self.init_grad_weight(self.sents_len)


    def init_grad_weight(self, length):
        grad_size = 0
        if self.update_mode == 1:
            grad_size = self.layer1_size
        elif self.update_mode == 2:
            grad_size = 2 * self.layer1_size
        elif self.update_mode == 3:
            grad_size = 2 * self.layer1_size + 3
        grad = matutils.zeros_aligned((length, grad_size), dtype=REAL)
        if self.update_mode == 3:
            grad[:,grad_size - 3] = ADAM_BETA1
            grad[:,grad_size - 2] = ADAM_BETA1
            grad[:,grad_size - 1] = ADAM_BETA2
        return grad


    def train_iteration(self, sentences, iteration=None):
        if not iteration:
            iteration = self.iteration
        i = 0
        while i < iteration:
            logger.info("-------------iteration:%i-------------" % (i+1))
            self.train(sentences)
            (flag, warn_str) = self.sanity_check()
            if self.normalize_each_epoch:
                logger.info("normalize vectors")
                self.normalize_vectors()
            if not flag :
                logger.info("Warning: %s" % warn_str)
            i += 1


    def train(self, sentences, total_words=None, word_count=0, sent_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        logger.info("training model with %i workers on %i sentences and %i features, "
                    "using 'skipgram'=%s 'hierarchical softmax'=%s 'subsample'=%s and 'negative sampling'=%s" %
                    (self.workers, self.sents_len, self.layer1_size, self.sg, self.hs, self.sample, self.negative))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        sent_count = [sent_count]
        total_words = total_words or sum(v.count * v.sample_probability for v in itervalues(self.vocab))
        total_sents = self.total_sents #it's now different from self.sents_len
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = matutils.zeros_aligned(self.layer1_size + 8, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size + 8, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                if self.update_mode == 0:
                    alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                else:
                    alpha = self.alpha
                job_words = sum(train_sent_vec(self, self.sents[sent_no], sentence, alpha, work, neu1, self.sents_grad[sent_no])
                                for sent_no, sentence in job)
                with lock:
                    word_count[0] += job_words
                    sent_count[0] += chunksize
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% sents, alpha %.05f, %.0f words/s" %
                                    (100.0 * sent_count[0] / total_sents, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for sent_tuple in sentences:
                sentence = sent_tuple[0]
                sent_id  = sent_tuple[1]
                sent_no = self.sent_no_hash[sent_id]
                sampled = [self.vocab.get(word, None) for word in sentence
                           if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= random.random_sample())]
                yield (sent_no, sampled)

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
                    (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]

    def sanity_check(self):
        if self.sg:
            return True, None
        rand_indices = random.randint(len(self.w2v.vocab),size=10)
        sim_top10_avg = 0
        for idx in rand_indices:
            w = self.w2v.index2word[idx]
            sim_words = self.w2v.most_similar(positive=[w],topn=10)
            sim_top10_avg += sim_words[9][1]
        sim_top10_avg /= len(rand_indices)
        logger.info("average similarity: %f"% sim_top10_avg)
        if sim_top10_avg > self.sane_max_sim10:
            return False, "insane average similarity > %f" % (self.sane_max_sim10)
        return True, None


    def normalize_vectors(self):
        for i in xrange(self.w2v.syn0.shape[0]):
            self.w2v.syn0[i, :] /= sqrt((self.w2v.syn0[i, :] ** 2).sum(-1))
        if self.hs:
            for i in xrange(self.w2v.syn1.shape[0]):
                self.w2v.syn1[i, :] /= sqrt((self.w2v.syn1[i, :] ** 2).sum(-1))
        if self.negative:
            for i in xrange(self.w2v.syn1neg.shape[0]):
                self.w2v.syn1neg[i, :] /= sqrt((self.w2v.syn1neg[i, :] ** 2).sum(-1))
        for i in xrange(self.sents.shape[0]):
            self.sents[i, :] /= sqrt((self.sents[i, :] ** 2).sum(-1))

    
    def train_single_sent_id(self, sentences, iteration, work=None, neu1=None):
        if work is None: work = zeros(self.layer1_size, dtype=REAL)
        if neu1 is None: neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
        num_of_grad = 0
        if (self.update_mode == 1): num_of_grad = self.layer1_size
        elif (self.update_mode == 2): num_of_grad = 2 * self.layer1_size
        elif (self.update_mode == 3): num_of_grad = 2 * self.layer1_size + 3
        sent_grad = zeros(num_of_grad, dtype=REAL)

        if self.init_adjust:
            denom = sqrt(self.layer1_size)
        else:
            denom = self.layer1_size

        new_sent = (random.rand(self.layer1_size).astype(REAL) - 0.5) / denom
        for i in range(iteration):
            alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * i / iteration)) if self.update_mode == 0 else self.alpha
            for sentence in sentences:
                sampled = [self.vocab.get(word, None) for word in sentence]
                train_sent_vec(self, new_sent, sampled, alpha, work, neu1, sent_grad)
        
        return new_sent

    def infer_sent(self, sentences, iteration=5, k=1, work=None, neu1=None, candidate_list=None):
        new_sent = self.train_single_sent_id(sentences, iteration, work, neu1)
        neighbors = self.most_similar_sentence(new_sent, k, candidate_list=candidate_list)
        sent_ids = []
        similarity = []
        for neighbor in neighbors:
            sent_ids.append(neighbor[0])
            similarity.append(neighbor[1])
        return new_sent, sent_ids, similarity

    def most_similar_sentence(self, vec, num, candidate_list=None):
        sims = empty(self.sents_len,dtype=REAL)
        if FAST_VERSION:
            sentvec_sim(self,vec,num,sims)
        else:
            vec_len = np_norm(vec)
            for idx in xrange(self.sents_len):
                vec2 = self.sents[idx]
                vec2_len = np_norm(vec2)
                sims[idx] = dot(vec,vec2) / vec_len / vec2_len
        nearest = []
        topN = argsort(sims)[::-1]
        for top_sent in topN:
            sent_id = self.sent_id_list[top_sent]
            if candidate_list is not None and not sent_id in candidate_list:
                continue
            nearest.append((sent_id,float(sims[top_sent])))
            if len(nearest) == num: break
        return nearest

    def save(self, fname, separately=None, sep_limit=10 * 1024**2, ignore=[]):
        ignore.append("w2v")
        self.w2v.save(fname+"_w2v", separately, sep_limit)
        super(Sentence2Vec, self).save(fname, separately, sep_limit, ignore)
    
    @classmethod
    def load(cls, fname, mmap=None):
        model = super(Sentence2Vec, cls).load(fname, mmap)
        if os.path.isfile(fname+"_w2v"):
            model.w2v = Word2Vec.load(fname+"_w2v", mmap)
            model.vocab = model.w2v.vocab
        return model
    
    @classmethod
    def arg_parser(cls):
        parser = ArgumentParser(description="Sentence2Vec ver." + SENT2VEC_VERSION)
        parser.set_defaults(model="cb")
        parser.set_defaults(hs=0)
        parser.set_defaults(neg=0)
        parser.set_defaults(sample=0)
        parser.set_defaults(alpha=0.025)
        parser.set_defaults(dim=200)
        parser.set_defaults(iteration=20)
        parser.set_defaults(thread=cpu_count())
        parser.set_defaults(update=0)
        parser.add_argument("--version", action="version", version="Sentence2Vec version: " + SENT2VEC_VERSION)
        parser.add_argument("-m", "--model", dest="model", type=str, help="specify model(cb for cbow/dm, sg for skip-gram/dbow)")
        parser.add_argument("--hs", dest="hs", type=int, help="hierarchical softmax 0:disable 1:enable")
        parser.add_argument("--neg", dest="neg", type=int, help="negative sampling 0:disable >=1:number of sampling")
        parser.add_argument("-s", "--sample", dest="sample", type=float, help="subsampling")
        parser.add_argument("-a", "--alpha", dest="alpha", type=float, help="(initial) learning rate")
        parser.add_argument("-d", "--dim", dest="dim", type=int, help="dimension")
        parser.add_argument("-i", "--iteration", dest="iteration", type=int, help="iterations / epochs")
        parser.add_argument("-t", "--thread", dest="thread", type=int, help="threads")
        parser.add_argument("-u", "--update", dest="update", type=int, help="update mode 0:SGD(default) 1:AdaGrad 2:AdaDelta 3:ADAM")
        parser.add_argument("-o", "--outdir", dest="outdir", type=str, help="output directory")
        parser.add_argument("--train", nargs="+", help="training file(s)")
        return parser

    def identifier(self):
        name = "sent%d" % (self.layer1_size)
        if self.sg:
            name += "_sg"
        else:
            name += "_cb"
        if self.hs:
            name += "_hs"
        else:
            name += "_neg%d" % self.negative
        name += "_a%g" % self.alpha
        name += "_it%d" % self.iteration
        if self.normalize_each_epoch:
            name += "_n"
        if self.update_mode == 0:
            name += "_sgd"
        elif self.update_mode == 1:
            name += "_adagrad"
        elif self.update_mode == 2:
            name += "_adadel"
        elif self.update_mode == 3:
            name += "_adam"
        return name

    def sent_vec_similarity(self, sent_id1, sent_id2):
        """
        Compute cosine similarity between two sentences. sent1 and sent2 are
        the indexs in the train file.

        Example::

          >>> trained_model.sent_vec_similarity(sent_id1, sent_id1)
          1.0

          >>> trained_model.sent_vec_similarity(sent_id1, sent_id3)
          0.73

        """
        return dot(matutils.unitvec(self.sents[self.sent_no_hash[sent_id1]]), matutils.unitvec(self.sents[self.sent_no_hash[sent_id2]]))

# Example: ./word2vec.py ~/workspace/word2vec/text8 ~/workspace/word2vec/questions-words.txt ./text8
if __name__ == "__main__":
    import re
    from sentences import CatSentence
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    logging.info("using optimization %s" % FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    parser = Sentence2Vec.arg_parser()
    parser.add_argument("--split", dest="split", action="store_true", help="use this option for split training data", default=False)
    args = parser.parse_args()

    seterr(all='raise')  # don't ignore numpy errors

    input_file = args.train[0]
    p_dir = re.compile("^.*/")
    basename = p_dir.sub("",input_file)
    if args.outdir:
        outdir = args.outdir
    else:
        m = p_dir.search(input_file)
        outdir = m.group(0) if m else ""
    logging.info("save to %s%s_{model_id}.model" % (outdir, basename))
    if args.split and len(args.train) > 1:
        input_file = args.train
    model = Sentence2Vec(CatSentence(input_file, split=args.split), iteration=args.iteration, model=args.model, hs = args.hs, negative = args.neg, workers = args.thread, size=args.dim, update_mode = args.update)
    model.save("%s%s_%s.model" % (outdir, basename, model.identifier()))
    
    program = os.path.basename(sys.argv[0])
    logging.info("finished running %s" % program)
