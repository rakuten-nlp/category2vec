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

try:
    from cat2vec_bind import train_from_job, train_cat_vec, catvec_sim, sentvec_sim, catsentvec_sim_concat, catsentvec_sim_sum, init_pairtable, FAST_VERSION, IS_DOUBLE, ADAM_BETA1, ADAM_BETA2
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    from cat2vec_pp import train_from_job, train_cat_vec, catvec_sim, sentvec_sim, catsentvec_sim_concat, catsentvec_sim_sum, init_pairtable, FAST_VERSION, IS_DOUBLE, ADAM_BETA1, ADAM_BETA2

if IS_DOUBLE:
    from numpy import float64 as REAL
else:
    from numpy import float32 as REAL

from numpy import exp, dot, zeros, outer, random, dtype, get_include, amax,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
from numpy.linalg import norm as np_norm

logger = logging.getLogger("cat2vec")

import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange
from word2vec import Word2Vec, Vocab
from multiprocessing import cpu_count
from argparse import ArgumentParser

CAT2VEC_VERSION = "0.01"

class Category2Vec(utils.SaveLoad):
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
        self.cat_learn = 1
        self.layer1_size = size
        self.min_count = min_count
        self.sent_no_hash = {} # mapping sent_id to index of self.sents
        self.sent_id_list = [] # mapping sent_no to sent_id
        self.cat_no_hash = {} # mapping cat_id to index of self.cats
        self.cat_id_list = [] # mapping cat_no to cat_id
        self.sane_vec_len = 100000 # for sanity check
        self.sane_max_sim10 = 0.9 # for sanity check
        self.init_adjust = init_adjust # for adjustment of initialization
        self.update_mode = update_mode # 0:SGD, 1: AdaGrad, 2:AdaDelta, 3:ADAM
        self.normalize_each_epoch = normalize_each_epoch # normalize vectors after each epoch

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
        if not has_vocab :
            logger.info("build vocabulary and")
        logger.info("resetting vectors")
        random.seed(self.seed)
        sentence_no, vocab = -1, {}
        total_words = 0
        self.sents_len = 0 #the num of sentence ids
        self.total_sents = 0 #the num of sentences
        self.cat_len = 0 #the num of category ids
        sent_cat_hash = {} #hash table for sent_no and cat_no
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
            cat_id = sent_tuple[2]
            self.total_sents += 1
            if not self.cat_no_hash.has_key(cat_id):
                self.cat_no_hash[cat_id] = self.cat_len
                self.cat_id_list.append(cat_id)
                self.cat_len += 1
            if not self.sent_no_hash.has_key(sent_id):
                self.sent_no_hash[sent_id] = self.sents_len
                self.sent_id_list.append(sent_id)
                self.sents_len += 1
            sent_cat = str(self.sent_no_hash[sent_id])+" "+str(self.cat_no_hash[cat_id])
            sent_cat_hash.setdefault(sent_cat,0)
            sent_cat_hash[sent_cat] += 1

        logger.info("collected %i word types from a corpus of %i words and %i sentences(ident:%i)  with %i categories" %
                    (len(vocab), total_words, self.total_sents, self.sents_len, self.cat_len))

        self.build_vocab(vocab)
        self.sents = matutils.zeros_aligned((self.sents_len, self.layer1_size), dtype=REAL)
        self.cats = matutils.zeros_aligned((self.cat_len, self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        self.reset_weights()

        # make sent_cat_pair
        self.sent_cat_pair = empty((len(sent_cat_hash),2), dtype=uint32)
        self.pair_len = len(sent_cat_hash)
        idx = 0
        for sent_cat in sent_cat_hash.keys():
            tpl = sent_cat.split(" ")
            self.sent_cat_pair[idx][0] = uint32(tpl[0])
            self.sent_cat_pair[idx][1] = uint32(tpl[1])
            idx += 1
        #sort by cat_no, sent_no in place
        self.sent_cat_pair.view('u4,u4').sort(order=['f1','f0'], axis=0)


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
        for idx in xrange(self.cat_len):
            self.cats[idx] = (random.rand(self.layer1_size) - 0.5) / denom
        # gradients for vectors
        self.syn0_grad = self.init_grad_weight(len(self.w2v.vocab))
        self.syn1_grad = self.init_grad_weight(len(self.w2v.vocab)) if self.hs > 0 else zeros(0, dtype=REAL)
        self.syn1neg_grad = self.init_grad_weight(len(self.w2v.vocab)) if self.negative > 0 else zeros(0, dtype=REAL)
        self.sents_grad = self.init_grad_weight(self.sents_len)
        self.cats_grad = self.init_grad_weight(self.cat_len)
        self.pairnorm = None


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
                job_words = train_from_job(self, job, alpha, work, neu1)
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
                cat_id = sent_tuple[2]
                sent_no = self.sent_no_hash[sent_id]
                cat_no = self.cat_no_hash[cat_id]
                sampled = [self.vocab.get(word, None) for word in sentence
                           if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= random.random_sample())]
                yield (cat_no, sent_no, sampled)

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
        veclens = empty(self.cat_len, dtype=REAL)
        for i in xrange(self.cat_len):
            veclens[i] = np_norm(self.cats[i])
        max_len = amax(veclens)
        logger.info("max vector length: %f" % max_len)
        if max_len > self.sane_vec_len:
            return False, "insane max vector length > %f" % (self.sane_vec_len)
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
        for i in xrange(self.cats.shape[0]):
            self.cats[i, :] /= sqrt((self.cats[i, :] ** 2).sum(-1))


    def init_pairnorm(self):
        # avoid initializing from multiple threads
        lock = threading.Lock()
        with lock:
            if getattr(self, 'pairnorm', None) is not None: return
            self.pairnorm = matutils.zeros_aligned((self.pair_len, self.layer1_size), dtype=REAL)
            init_pairtable(self)
            

    def train_single_sent_id(self, sentences, iteration, work=None, neu1=None, sent_vec=None, cat_vec=None):
        if work is None: work = matutils.zeros_aligned(self.layer1_size + 8, dtype=REAL)
        if neu1 is None: neu1 = matutils.zeros_aligned(self.layer1_size + 8, dtype=REAL)
        sent_grad = self.init_grad_weight(1)
        cat_grad = self.init_grad_weight(1)
        
        if sent_vec is None:
            sent_vec = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            if self.init_adjust:
                denom = sqrt(self.layer1_size)
            else:
                denom = self.layer1_size
            sent_vec[:] = (random.rand(self.layer1_size).astype(REAL) - 0.5) / denom
        if cat_vec is None:
            cat_vec = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            self.cat_learn = 0
    
        for i in range(iteration):
            alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * i / iteration)) if self.update_mode == 0 else self.alpha
            for sentence in sentences:
                sampled = [self.vocab.get(word, None) for word in sentence]
                train_cat_vec(self, sent_vec, cat_vec, sampled, alpha, work, neu1, sent_grad, cat_grad)
        return sent_vec, cat_vec


    def infer(self, sentences, iteration=5, k=1, work=None, neu1=None):
        self.init_pairnorm()
        sent_vec, cat_vec = self.train_single_sent_id(sentences, iteration, work, neu1)
        neighbors = self.most_similar_catsent(sent_vec, cat_vec, k, ident_cat = True)
        cat_ids = []
        sent_ids = []
        similarity = []
        for neighbor in neighbors:
            cat_id = neighbor[2]
            sent_ids.append(neighbor[0])
            cat_ids.append(cat_id)
            similarity.append(neighbor[1])
        sent_vec += cat_vec
        cat_vec = deepcopy(self.cats[self.cat_no_hash[cat_ids[0]]])
        sent_vec -= cat_vec
        return sent_vec, cat_vec, cat_ids, sent_ids, similarity


    def infer_sent(self, sentences, cat_id, iteration=5, k=1, work=None, neu1=None):
        cat_vec = self.cats[cat_id]
        self.cat_learn = 0
        sent_vec, cat_vec = self.train_single_sent_id(sentences, iteration, work, neu1, None, cat_vec)
        neighbors = self.most_similar_sentence(sent_vec, k)
        sent_ids = []
        similarity = []
        for neighbor in neighbors:
            sent_ids.append(neighbor[0])
            similarity.append(neighbor[1])
        return sent_vec, sent_ids, similarity


    def most_similar_sentence(self, vec, num):
        sims = empty(self.sents_len,dtype=REAL)
        sentvec_sim(self,vec,num,sims)
        nearest = []
        topN = argsort(sims)[::-1][0:num]
        for top_sent in topN:
            sent_id = self.sent_id_list[top_sent]
            nearest.append((sent_id,float(sims[top_sent])))
        return nearest


    def most_similar_category(self, vec, num):
        sims = empty(self.cat_len,dtype=REAL)
        catvec_sim(self,vec,num,sims)
        nearest = []
        topN = argsort(sims)[::-1][0:num]
        for top_cand in topN:
            cat_id = self.cat_id_list[top_cand]
            nearest.append((cat_id,float(sims[top_cand])))
        return nearest


    def most_similar_catsent_concat(self, svec, cvec, num, sent2cat):
        self.sent2cat = sent2cat
        sims = zeros(self.sents_len,dtype=REAL)
        catsentvec_sim_concat(self, svec, cvec, num, sims)
        nearest = []
        topN = argsort(sims)[::-1][0:num]
        for top_cand in topN:
            sent_id = self.sent_id_list[top_cand]
            nearest.append((sent_id,float(sims[top_cand])))
        return nearest
    

    def most_similar_catsent(self, svec, cvec, num, ident_cat = False):
        sims = zeros(self.pair_len, dtype=REAL)
        catsentvec_sim_sum(self, svec, cvec, sims)
        nearest = []
        cat_ids = {}
        neighbors = argsort(sims)[::-1]
        for top_cand in neighbors:
            (sent_no, cat_no) = self.sent_cat_pair[top_cand]
            sent_id = self.sent_id_list[sent_no]
            cat_id = self.cat_id_list[cat_no]
            if not ident_cat or not cat_ids.has_key(cat_id):
                cat_ids[cat_id] = 1
                nearest.append((sent_id,float(sims[top_cand]),cat_id))
            if len(nearest) == num: break
        return nearest


    def save(self, fname, separately=None, sep_limit=10 * 1024**2, ignore=["pairnorm"]):
        ignore.append("w2v")
        self.w2v.save(fname+"_w2v", separately, sep_limit)
        super(Category2Vec, self).save(fname, separately, sep_limit, ignore)


    def save_sent2vec_format(self, fname):
        """
        Store sentence vectors

        """
        logger.info("storing %sx%s projection weights into %s" % (self.sents_len, self.layer1_size, fname))
        assert (self.sents_len, self.layer1_size) == self.sents.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("#sents_len: %d\n#size:%d\n" % self.sents.shape))
            fout.write(utils.to_utf8("#sg:%d\n#hs:%d\n#negative:%d\n#cbow_mean:%d\n" % (self.sg,self.hs,self.negative,self.cbow_mean)))
            for sent_id in self.sent_no_hash.keys():
                row = self.sents[self.sent_no_hash[sent_id]]
                fout.write(utils.to_utf8("%s\t%s\n" % (sent_id, ' '.join("%f" % val for val in row))))


    def save_cat2vec_format(self, fname):
        """
        Store cat vectors

        """
        logger.info("storing %sx%s projection weights into %s" % (self.cat_len, self.layer1_size, fname))
        assert (self.cat_len, self.layer1_size) == self.cats.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("#cats_len: %d\n#size:%d\n" % self.cats.shape))
            fout.write(utils.to_utf8("#sg:%d\n#hs:%d\n#negative:%d\n#cbow_mean:%d\n" % (self.sg,self.hs,self.negative,self.cbow_mean)))
            for cat_id in self.cat_no_hash.keys():
                row = self.cats[self.cat_no_hash[cat_id]]
                fout.write(utils.to_utf8("%s\t%s\n" % (cat_id, ' '.join("%f" % val for val in row))))
    

    @classmethod
    def load(cls, fname, mmap=None):
        model = super(Category2Vec, cls).load(fname, mmap)
        if os.path.isfile(fname+"_w2v"):
            model.w2v = Word2Vec.load(fname+"_w2v", mmap)
            model.vocab = model.w2v.vocab
        return model


    @classmethod
    def load_cat2vec_format(cls, cat_model=None, sent_model=None, word_model=None):
        """
        Load sentence vectors
        """
        model = Category2Vec(None)
        count = 0
        if cat_model:
            logger.info("loading %s object(cat) from %s" % (cls.__name__, cat_model))
            for line in open(cat_model,"r"):
                line = line.rstrip()
                if count == 0:
                    info = line.split()
                    model.cat_len = int(info[0])
                    model.layer1_size = int(info[1])
                    model.sg = int(info[2])
                    model.hs = int(info[3])
                    model.negative = int(info[4])
                    model.cbow_mean = int(info[5])
                    model.cats = empty((model.cat_len, model.layer1_size), dtype=REAL)
                    model.cat_no_hash = {}
                    model.cat_id_list = []
                else:
                    idx = count - 1
                    row = line.split("\t")
                    cat_id = utils.to_unicode(row[0])
                    model.cat_no_hash[cat_id] = idx
                    model.cat_id_list.append(cat_id)
                    vals = row[1].split()
                    for j in xrange(model.layer1_size):
                        model.cats[idx][j] = float(vals[j])
                count += 1
        count = 0
        if sent_model:
            logger.info("loading %s object(sentence) from %s" % (cls.__name__, sent_model))
            for line in open(sent_model,"r"):
                line = line.rstrip()
                if count == 0:
                    info = line.split()
                    model.sents_len = int(info[0])
                    model.sents = empty((model.sents_len, model.layer1_size), dtype=REAL)
                    model.sent_no_hash = {}
                    model.sent_id_list = []
                else:
                    idx = count - 1
                    row = line.split("\t")
                    sent_id = utils.to_unicode(row[0])
                    model.sent_no_hash[sent_id] = idx
                    model.sent_id_list.append(sent_id)
                    vals = row[1].split()
                    for j in xrange(model.layer1_size):
                        model.sents[idx][j] = float(vals[j])
                count += 1
        if word_model:
            logger.info("loading word2vec from %s" % word_model)
            model.w2v = Word2Vec.load(word_model)
            model.vocab = model.w2v.vocab
        return model


    @classmethod
    def arg_parser(cls):
        parser = ArgumentParser(description="Category2Vec ver." + CAT2VEC_VERSION)
        parser.set_defaults(model="cb")
        parser.set_defaults(hs=0)
        parser.set_defaults(neg=0)
        parser.set_defaults(sample=0)
        parser.set_defaults(alpha=0.025)
        parser.set_defaults(dim=200)
        parser.set_defaults(iteration=20)
        parser.set_defaults(thread=cpu_count())
        parser.set_defaults(update=0)
        parser.set_defaults(norm=False)
        parser.add_argument("--version", action="version", version="Category2Vec version: " + CAT2VEC_VERSION)
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
        parser.add_argument('-n', "--normalize", dest="norm", action='store_true')
        parser.add_argument("--train", nargs="+", help="training file(s)")
        return parser


    def identifier(self):
        name = "cat%d" % (self.layer1_size)
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


    def cat_vec_similarity(self, cat_id1, cat_id2):
        """
        Compute cosine similarity between two sentences. sent1 and sent2 are
        the indexs in the train file.

        Example::

          >>> trained_model.cat_vec_similarity(cat_id1, cat_id1)
          1.0

          >>> trained_model.cat_vec_similarity(cat_id1, cat_id3)
          0.73

        """
        return dot(matutils.unitvec(self.cats[self.cat_no_hash[cat_id1]]), matutils.unitvec(self.cats[self.cat_no_hash[cat_id2]]))


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

    parser = Category2Vec.arg_parser()
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
    model = Category2Vec(CatSentence(input_file, split=args.split), iteration=args.iteration, model=args.model, hs = args.hs, negative = args.neg, workers = args.thread, alpha=args.alpha, size=args.dim, update_mode = args.update, normalize_each_epoch = args.norm)
    model.save("%s%s_%s.model" % (outdir, basename, model.identifier()))
    
    program = os.path.basename(sys.argv[0])
    logging.info("finished running %s" % program)
