# Category2Vec
[English README](https://github.com/rakuten-nlp/category2vec/blob/master/README.md)
## はじめに
Category2Vecはカテゴリベクトルモデル[Marui and Hagiwara 2015]・段落ベクトルモデル [Le and Mikolov 2014]の実装です。
ソースコードはgensim project (https://radimrehurek.com/gensim/)[Rahurek 2013]のword2vec [Mikolov et al. 2013a,b]実装を基に開発されました。

学習後、カテゴリ・段落・単語の分散表現が得られ、カテゴリ同士の関係をカテゴリベクトルから抽出したり、
カテゴリが分かっていない短文に対してカテゴリを推測することができます。

ここで言う段落とは文やいくつかの文のまとまりを指し、カテゴリとはいくつかの段落が属するまとまり(タグやジャンルも含む)を指しています。

## デモ
カテゴリベクトルモデル(クラス名: Category2Vec)と段落ベクトルモデル(クラス名：Sentence2Vec)のデモプログラムを動かすには、
以下のコマンドを入力します。

```
    make # エクステンションのビルド (推奨)
    python demo_catvec.py  # カテゴリベクトルモデルのデモ
    python demo_sentvec.py # 段落ベクトルモデルのデモ
```
RAMは8GB以上の環境で実行してください。

## 使い方
### ダウンロード
git リポジトリからクローンするか

```
    git clone https://github.com/rakuten-nlp/category2vec.git
```

zip アーカイブをダウンロードしてください: https://github.com/rakuten-nlp/category2vec/archive/master.zip

### 依存モジュール
Category2Vecは以下のモジュールに依存しています。

* numpy
* scipy
* six
* cython

Ubuntu環境では以下のコマンドでインストールできます。

```
    apt-get install python-dev python-pip python-numpy python-scipy
    pip install six cython
```

必須ではありませんが、以下のようなBLASライブラリをインストールすることを推奨しています。

* ATLAS
* OpenBLAS

Ubuntu環境では以下のコマンドでATLASのインストールができます。
```
    apt-get install libatlas-base-dev
```

Ubuntuで配布されているOpenBLASには制限があることから、
以下のようにソースからインストールすることもできます。
```
    apt-get install gfortran
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    #Makefile.rule を編集 (USE_THREAD=1 推奨)
    make
    make install PREFIX=インストールするディレクトリ #例えば PREFIX=/usr/local のようにする
```


### ビルド
Category2Vecを使うためにエクステンションをビルドをすることは必須ではありませんが、
エクステンション無しのバージョンは速度が遅く制限もあるため、ビルドを推奨しています。
ビルドにはGCC のバージョン4.6以上と同等のコンパイラが必要です。

コンパイルの前に`settings.py`を編集してください。
scipy のバージョン0.15.1以上を使う場合には、`use_blas = True`を使い、用いるBLAS ライブラリを`blas_include`と`blas_libs`に指定してください。
Intel のCPU (Sandy Bridge以降) を使う場合には、`use_avx = True`を指定してAVXを使ってより最適化することもできます。

`settings.py`での設定が終わったら`make`でエクステンションをビルドしてください。

#### Ubuntuの場合
現在のバージョンは以下の環境で動作確認しています。

* Ubuntu 12.04 / 14.04
* Python 2.7.3 / 2.7.6 / 2.7.8 (anaconda)
* Numpy 1.8.0 / 1.8.2 / 1.9.0 (anaconda) / 1.10.0
* Scipy 0.9.0 / 0.13.3 / 0.14.0 (anaconda) / 0.15.0 / 0.16.0 (use_blas=True)
* gcc 4.6.3 / 4.8.1 / 4.8.3
* OpenBLAS 0.2.12 / ATLAS 3.10.1-4

CPU 環境は以下の通りです。

* Haswell CPUs
    * i7-4770K
    * i7-5820K
* Ivy Bridge CPUs
    * Xeon E5-26xx v2 series
* AMD CPUs
    * Opteron 4100 series (use_avx=False)

#### Mac OS X
一部の環境では、直接BLASライブラリを用いない設定(`use_blas = False`)で、
ベクトルの要素がNaNになってしまう問題が報告されています。
この場合はBLASライブラリを使うと問題が回避できます。
OpenBLASを上記のようにソースからコンパイルすることもできますが、
Homebrewを用いて以下のようにインストールすることもできます。
```
    brew install homebrew/science/openblas
    # ~/.bash_profile に以下の記述をする
    # export LDFLAGS="-L/usr/local/opt/openblas/lib"
    # export CPPFLAGS="-I/usr/local/opt/openblas/include"
    # settings.py を編集して以下のように設定する
    # use_blas = True
    # blas_libs = ["openblas"]
```

`clang`でコンパイルする場合は`settings.py`を編集して`use_clang = True`としてください。

これらの設定をしたら以下のように依存モジュールをインストールしてエクステンションをビルドしてください。
```
    easy_install pip
    pip install numpy scipy six cython
    make
```

`gcc`を使う場合には`settings.py`を編集して`force_gcc = True`と`use_clang = False`を指定してください。
Homebrewを使う場合には以下のようにすることができます。
```
    brew install gcc49
    brew install python
    source ~/.bash_profile # brewのpythonを使いたいため
    easy_install pip
    pip install numpy scipy six cython # cythonをgccを使ってインストールしないと問題が起きることがある
    make
```
`gcc`を使っていてAVXに起因するエラー (例えば no such instruction error等)が起きる場合には、OS Xに付属しているassembler (usr/bin/as) を以下のスクリプトのように編集します。

https://gist.github.com/xianyi/2957847

または`use_avx = False`としてAVXを無効にしてこのエラーを回避することもできます。

Mac OS X 10.9.5, Python 2.7.9 (Numpy 1.9.2, Scipy 0.15.1), Apple LLVM 6.0 / gcc 4.8.4 (brew gcc48, assembler modified) / gcc 4.9.2 (brew gcc49, assembler modified)の環境で動作確認しています。

### 使用例
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

## 利用規約・ライセンス

Category2Vec はGNU Lesser General Public License v3.0 https://www.gnu.org/licenses/lgpl.html の元で公開されています。
本ライセンスに従う限り、Category2Vec の再配布、変更、研究/商用利用は自由に行っていただいて構いません。

研究目的でCategory2Vec を使用する場合、Category2Vec の論文 [Marui and Hagiwara 2015] を引用してください。

また、現在Category2Vec に対するissueやpull request等の貢献を受け付けていますが、
現時点では貢献ポリシーが最終決定しておりません。
公式ポリシーを近いうちに告知し、その後皆様の貢献を完全な形で受け付ける予定です。

## よくある質問
Q. 商用利用はできますか？
- A. 利用規約・ライセンスに従う限り、商用利用は許可されています。詳細については、上記「利用規約・ライセンス」を参照してください。


## 謝辞
本プロジェクトに対してご協力いただいた、萩原 正人、山田 薫 (敬称略) の各氏に感謝いたします。

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