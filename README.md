# Description

This repo is a playground/testbed for popular Neural Machine Translation algorithms. This code was intended to serve as an exercise in self-study. I began this project with the goal of producing a state-of-the-art translation system, and you will find that system in the **phase 3** directory. It is freely distributed under the MIT license.

Each folder corresponds to a seperate learning objective:

### Phase 1: Implementing Networks:
This section contains a series of deep learning models that work towards the complexity of machine translation: 
  * Feed-forward Neural Net for classification
  * Recurent Neural Nets that read the characters of a book and predict what characters will come next. [Karpathy's RNN blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) inspired these models
  * The word2vec skipgram model for learning word vectors [[Mikolov et al.]](https://arxiv.org/abs/1301.3781)
  * A sequence-to-sequence autoencoder [[Dai et al.]](https://arxiv.org/abs/1511.01432)
  * The encoder-decoder translation model of [[Cho et al.]](https://arxiv.org/abs/1406.1078)
  * An encoder-decoder translation system with the attention mechanism of [[Luong et al.]](https://arxiv.org/abs/1508.04025)

### Phase 2: Language Data Wrangling 
This section contains a grab-bag of tools and algorithms for working with language data. It also contains some scripts for crawling a Japanese-English parallel corpus from the web, but I moved that project to its own repo (https://github.com/rpryzant/japanese_corpus).
  * Spellchecker that reconstructs words based on edit-distance and a bigram+unigram language model.
  * The sentence-similarity algorithms proposed by [[Utiyama et al.]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.585.1364&rep=rep1&type=pdf) and [[Isahara et al]](http://dl.acm.org/citation.cfm?id=1075106)

### Phase 3: Neural Machine Translation Translation System
This is a complete Japanese-English NMT system based on [[Sundermeyer et al.]](https://pdfs.semanticscholar.org/d29c/f0f457ec2089fd4d776ef9a246de810be689.pdf) and [[Luong et al.]](https://arxiv.org/abs/1508.04025). It was designed with ease-of-use and extensibility in mind.

# Requirements

* Python 2.7
* Tensorflow > 0.12
* Numpy
* NLTK
* h5py
* tqdm
* matplotlib





