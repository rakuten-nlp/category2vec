# Category2Vec
[Japanese README (日本語ドキュメント)] (https://github.com/rakuten-nlp/category2vec/blob/master/README-ja.md)
## Introduction
Category2Vec is an implementation of the category vector models [Marui and Hagiwara 2015], and the paragraph vector models [Le and Mikolov 2014].
These programs are based on word2vec [Mikolov et al. 2013a,b] in gensim project (https://radimrehurek.com/gensim/)[Rahurek 2013].

After training, you can obtain distributed representations for categories, paragraphs, and words. You can also infer a category from a description.

## Demo
There are demo programs of category vector models(Category2Vec) and paragraph vector models(Sentence2Vec).
To execute the demo programs,

```
    make # to build extensions (optional)
    python demo_catvec.py  # for category vector models
    python demo_sentvec.py # for paragraph vector models
```


## Usage
### Download
Clone the git repository as 

```
    git clone https://github.com/rakuten-nlp/category2vec.git
```

or download the zip archive from here: https://github.com/rakuten-nlp/category2vec/archive/master.zip

### Dependency
Category2Vec requires following python modules:

* numpy
* scipy
* six
* cython

For Ubuntu, you can install them as follows:
```
    apt-get install python-dev python-pip python-numpy python-scipy
    pip install six cython
```

To build extensions, it is recommended to get a BLAS library such as:

* ATLAS
* OpenBLAS

For Ubuntu, you can use the provided ATLAS package as follows:
```
    apt-get install libatlas-base-dev
```

There is limitation in OpenBLAS package included in Ubuntu, we recommend you to install it from the source.
```
    apt-get install gfortran
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    #edit Makefile.rule (USE_THREAD=1 is recommended)
    make
    make install PREFIX=your_installation_directory # for example, PREFIX=/usr/local
```


### Build
You don't have to build the extensions to use Category2Vec,
but to pursue good performance, we recommend to build the extensions.
GCC >= 4.6 is required.

Before the compilation, edit `settings.py`.
If you use scipy >= 0.15.1, you should use the option `use_blas = True`, and specify your BLAS library in `blas_include` and `blas_libs`.
If you have an Intel CPU (>= Sandy Bridge), you can enable the option `use_avx = True`.

After you editted the settings, type `make` to build extensions.

#### Ubuntu
This version has been tested under the following environments:

* Ubuntu 12.04 / 14.04
* Python 2.7.3 / 2.7.6 / 2.7.8 (anaconda)
* Numpy 1.8.0 / 1.8.2 / 1.9.0 (anaconda) / 1.10.0
* Scipy 0.9.0 / 0.13.3 / 0.14.0 (anaconda) / 0.15.0 / 0.16.0 (use_blas=True)
* gcc 4.6.3 / 4.8.1 / 4.8.3
* OpenBLAS 0.2.12 / ATLAS 3.10.1-4

And under the following CPUs:

* Haswell CPUs
    * i7-4770K
    * i7-5820K
* Ivy Bridge CPUs
    * Xeon E5-26xx v2 series
* AMD CPUs
    * Opteron 4100 series (use_avx=False)

#### Mac OS X
In some environment, you'll get NaNs with `use_blas = False`. 
You should use BLAS library in this case.
You can install OpenBLAS using the instruction above,
or you can install it from Homebrew as follows:
```
    brew install homebrew/science/openblas
    # if you use brewed openblas, you should add the following in the ~/.bash_profile
    # export LDFLAGS="-L/usr/local/opt/openblas/lib"
    # export CPPFLAGS="-I/usr/local/opt/openblas/include"
    # edit settings.py and specify the followings
    # use_blas = True
    # blas_libs = ["openblas"]
```

If you use `clang`, edit `settings.py` and specify `use_clang = True`.
```
    easy_install pip
    pip install numpy scipy six cython # it's important to build cython using gcc
    make
```

If you want to use gcc, you can edit `settings.py` and specify `force_gcc = True` and `use_clang = False`.
For Homebrew,
```
    brew install gcc49
    brew install python
    source ~/.bash_profile # to use brewed python
    easy_install pip
    pip install numpy scipy six cython # it's important to build cython using gcc
    make
```
If you get errors around AVX (e.g. no such instruction error), replace the native OS X assembler (usr/bin/as) by a script below.

https://gist.github.com/xianyi/2957847

Or you can specify `use_avx = False` to avoid this error.

This version has been tested under Mac OS X 10.9.5, Python 2.7.9 (Numpy 1.9.2, Scipy 0.15.1) and Apple LLVM 6.0 / gcc 4.8.4 (brew gcc48) / gcc 4.9.2 (brew gcc49) with the above modification on the assembler.

### Example
```
    from cat2vec import Category2Vec
    from sentences import CatSentence
    sentences = CatSentence("myfile.txt")
    # CV-DBoW model with hierarchical softmax, 10 iterations, dimension 300
    model = Category2Vec(sentences, model = "dbow", hs = 1, size = 300, iteration = 10)
    # Save a model
    model.save("myfile.model")
    # Load the model
    model = Category2Vec.load("myfile.model")
```

## Terms and Conditions
Distribution, modification, and academic/commercial use of Category2Vec is permitted, provided that
you conform with GNU Lesser General Public License v3.0 https://www.gnu.org/licenses/lgpl.html

If you are using Category2Vec for research purposes, please cite our paper on Category2Vec [Marui and Hagiwara 2015]

Although we are accepting contributions (e.g., issues and pull
requests) to this category2vec projects,
our contribution policy has not been finalized yet.
We will be announcing our official policy in near future, after which
we can fully accept contributions.

## FAQ (Frequently Asked Questions)
Q. Is commercial use permitted?
- A. Yes, as long as you follow the terms and conditions. See "Terms and Conditions" above for the details.

## Acknowledgements
The developers would like to thank Masato Hagiwara and Kaoru Yamada for their contribution to this project.

## References
[Marui and Hagiwara 2015] Junki Marui, and Masato Hagiwara. Category2Vec: 単語・段落・カテゴリに対するベクトル分散表現. 言語処理学会第21回年次大会(NLP2015).

[Le and Mikolov 2014] Quoc Le, and Tomas Mikolov. Distributed Representations of Sentence and Documents. In Proceedings of ICML 2014.

[Mikolov et al. 2013a] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations 
in Vector Space. In Proceedings of Workshop at ICLR, 2013.

[Mikolov et al. 2013b] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations 
of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.

[Rahurek 2013] Radim Rehurek, Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/

---

&copy; 2015 Rakuten NLP Project. All Rights Reserved. / Sponsored by [Rakuten, Inc.](http://global.rakuten.com/corp/) and [Rakuten Institute of Technology](http://rit.rakuten.co.jp/).