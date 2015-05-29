import utils
import random, glob

class CatSentence(object):
    """format(one line):[Category ID]\t[Sentence ID]\t[sentence split by a tab]"""
    def __init__(self, source, cat_col=0, sent_col=1, cont_col=2, split=False, rand=False):
        self.source = source
        self.cat_col = cat_col
        self.sent_col = sent_col
        self.cont_col = cont_col
        self.split = split
        self.rand = rand
    def __iter__(self):
        if not self.split:
            try:
                self.source.seek(0)
                for line in self.source:
                    k = utils.to_unicode(line.rstrip()).split("\t")
                    yield k[self.cont_col:],k[self.sent_col],k[self.cat_col]
            except AttributeError:
                with utils.smart_open(self.source) as fin:
                    for line in fin:
                        k = utils.to_unicode(line.rstrip()).split("\t")
                        yield k[self.cont_col:],k[self.sent_col],k[self.cat_col]
        else:
            if isinstance(self.source, list):
                split_files = self.source
            else:
                split_files = glob.glob(self.source+".[a-z][a-z]")
            if self.rand: random.shuffle(split_files)
            for source in split_files:
                with utils.smart_open(source) as fin:
                    for line in fin:
                        k = utils.to_unicode(line.rstrip()).split("\t")
                        yield k[self.cont_col:],k[self.sent_col],k[self.cat_col]

class WikiSentence(object):
    def __init__(self, source):
        self.source = source
    
    def __iter__(self):
        try:
            self.source.seek(0)
            for line in self.source:
                k = utils.to_unicode(line.rstrip()).split("\t")
                categories = k[3].split(" ")
                for cat in categories:
                    if "/" in cat: continue
                    yield k[4:], k[1], cat
        except AttributeError:
            with utils.smart_open(self.source) as fin:
                for line in fin:
                    k = utils.to_unicode(line.rstrip()).split("\t")
                    categories = k[3].split(" ")
                    for cat in categories:
                        if "/" in cat: continue
                        yield k[4:], k[1], cat

class SampledWikiSentence(object):
    def __init__(self, source, sample = 0.1, seed = 1120):
        self.source = source
        self.seed = seed
        self.sample = sample
        
    def __iter__(self):
        random.seed(self.seed)
        try:
            self.source.seek(0)
            for line in self.source:
                if random.random() > self.sample: continue
                k = utils.to_unicode(line.rstrip()).split("\t")
                categories = k[3].split(" ")
                for cat in categories:
                    if "/" in cat: continue
                    yield k[4:], k[1], cat
        except AttributeError:
            with utils.smart_open(self.source) as fin:
                for line in fin:
                    if random.random() > self.sample: continue
                    k = utils.to_unicode(line.rstrip()).split("\t")
                    categories = k[3].split(" ")
                    for cat in categories:
                        if "/" in cat: continue
                        yield k[4:], k[1], cat
