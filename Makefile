# build extensions
compile_cmd = python setup.py build_ext --inplace --force
all: word2vec_inner.so cat2vec_bind.so sent2vec_bind.so

word2vec_inner.so: word2vec_inner.pyx options.h options.c settings.py
	$(compile_cmd)
sent2vec_bind.so: sent2vec_bind.pyx sent2vec_calc.h sent2vec_calc.cpp options.h options.c settings.py
	$(compile_cmd)
cat2vec_bind.so: cat2vec_bind.pyx cat2vec_calc.h cat2vec_calc.cpp options.h options.c settings.py
	$(compile_cmd)
