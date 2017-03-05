"""

=== DESCRIPTION
this file reproduces the sentence alignment scorer presented in 
https://pdfs.semanticscholar.org/d7a4/97cd9de61617ba55002d0db3435f64149ea0.pdf.

given an english and japanese sentence, it gives a *rough* idea as to the
similarity score of that pair, based on the number of words that could be
direct translations of each other


=== USAGE
run unit tsts: 
    python scorer.py

run from somewhere else:
    ps = PairScorer('path/to/en_ja_dictionary/raw_kv_pairs', 'path/to/rakuten_model_ja.min.json', d)
    ps.score(en sentence, ja sentence)
    ...

"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
from rakutenma import RakutenMA
import json
import nltk
from collections import defaultdict
from jpn.deinflect import guess_stem
from tqdm import tqdm


# FROM https://github.com/rakuten-nlp/rakutenma
# NOTE that this is ALL PoS tags, and non-content 
#      tags have been commented out
RAKUTEN_POS_TAGS = {
    'A-c': 'adjective-common',
    'A-dp': 'adjective-dependent',
#    'C': 'conjunction',
    'E': 'english word',
    'F': 'adverb',
#    'I-c': 'interjection-common',
    'J-c': 'adjectival noun-common',
    'J-tari': 'adjective noun-tari',
    'J-xs': 'adjectival noun-AuxVerb stem',
#    'M-aa': 'auxiliary sign-aa',
#    'M-c': 'auxiliary sign-common',
#    'M-cp': 'auxiliary sign - open parenthesis',
#    'M-p': 'auxiliary sign-period',
    'N-n': 'noun-noun',
    'N-nc': 'noun-common noun',
    'N-pn': 'noun-proper noun',
    'N-xs': 'noun-AuxVerb stem',
#    'O': 'others',
#    'P': 'prefix',
#    'P-fj': 'particle-adverbial',
#    'P-jj': 'partical-phrasal',
#    'P-k' 'particle-case making',
#    'P-rj': 'particle-binding',
#    'P-sj': 'particle-conjunctive',
#    'Q-a': 'suffix-adjective',
#    'Q-j': 'suffix-adjectival noun',
#    'Q-n': 'suffix-noun',
#    'Q-v': 'suffix-verb',
    'R': 'adnominal adjective',
#    'S-c': 'sign-common',
#    'S-l': 'sign-letter',
    'U': 'URL',
    'V-c': 'verb-common',
    'V-dp': 'verb-dependent',
#    'W': 'whitespace',
#    'X': 'auxVerb'
}

# from https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# non-content types have been commented out
PENN_POS_TAGS = {
#    'CC': 'COORDINATING CONJUNCTION',
    'CD': 'CARDINAL NUMBER',
#    'DT': 'DETERMINER',
#    'EX': 'EXISTENTIAL THERE',
    'FW': 'FOREIGN WORD',
#    'IN': 'PREPOSITION OR SUBORDINATING CONJUNCTION',
    'JJ': 'ADJECTIVE',
    'JJR': 'ADJECTIVE, COMPARATIVE',
    'JJS': 'ADJECTIVE, SUPERLATIVE',
#    'MD': 'MODAL',
    'NN': 'NOUN, SINGULAR OR MASS',
    'NNS': 'NOUN, PLURAL',
    'NNP': 'PROPER NOUN, SINMGULAR',
    'NNPS': 'PROPER NOUN, PLURAL',
#    'PDT': 'PREDETERMINER',
#    'POS': 'POSSESSIVE ENDING',
#    'PRP': 'PERSONNNAL PRONOUN',
#    'PRP$': 'POSSESSIVE PRONOUN',
    'RB': 'ADVERB',
    'RBR': 'ADVERB, COMPARATIVE',
    'RBS': 'ADVERB, SUPERLATIVE',
#    'RP': 'PARTICLE',
#    'SYM': 'SYMBOL',
#    'TO': 'TO',
#    'UH': 'INTERJECTION',
    'VB': 'VERB BASE',
    'VBD': 'VERB PAST TENSE',
    'VBG': 'VERB GERUND OR PRESENT PARTICIPLE',
    'VBN': 'VERB PAST PARTICIPLE',
    'VBP': 'VERB NON-3RD PERSON SINGULAR PRESENT',
    'VBZ': 'VERB 3RD PERSON SINGULAR PRESENT',
#    'WDT': 'WH-DETERMINER',
#    'WP': 'WP-PRONOUN',
#    'WP$': 'POSSESSIVE WH-PRONOUN',
#    'WRB': 'WH-ADVERB'
}



class Dictionary():
    def __init__(self, kv_filepath):
        print '\t DICT: parsing dictionay..'
        self.ja_to_en = defaultdict(list)
        self.en_to_ja = defaultdict(list)

        # build {word: [possible translations] } mappings
        for l in open(kv_filepath):
            [k, v] = l.strip().split(',')[:2]
            raw = unicode(k, 'utf-8')
            self.ja_to_en[raw].append(v)
            self.en_to_ja[v].append(raw)


    def is_translation_pair(self, en, ja):
        try:
            return 1 if (ja in self.en_to_ja[en] or \
                             en in ' '.join(x for x in self.ja_to_en[ja])) \
                             else 0
        except UnicodeDecodeError:
            # if there's a non-ascii in the en stuff, return 0 (en should be all plaintext)
            return 0


class PairScorer():
    def __init__(self, kv_filepath, model):
        self.rma = RakutenMA(json.loads(open(model).read()))
        self.rma.hash_func = RakutenMA.create_hash_func(self.rma, 15)
        self.dict = Dictionary(kv_filepath)

    def extract_ja_content_lemmas(self, s):
        """ extracxts content words from a japanese sentence
            (nouns, verb (+roots sometimes, TODO MAKE ALWAYS? (HOW?)), 
            adjectives, no okurigana)
        """
        s = unicode(s, 'utf-8')

        out = []
        for [x, y] in self.rma.tokenize(s):
            if y in RAKUTEN_POS_TAGS:
                if y.startswith('V'):                
                    out += [(guess, y) for guess in guess_stem(x)]
                else:
                    out.append( x )
        return out

    def extract_en_content_lemmas(self, s):
        """ extracts content lemmas from english sentences
        """

        def penn_to_wordnet(pos):
            p = pos[0].lower()
            if    p == 'j': return 'a'
            elif  p == 'r': return 'r'
            elif  p == 'v': return 'v'
            else:             return 'n'

        lemmatizer = nltk.stem.WordNetLemmatizer()
        s = unicode(s, 'utf-8')
        
        out = []
        for w, pos in nltk.pos_tag(nltk.word_tokenize(s)):
            if pos in PENN_POS_TAGS:
                out.append( lemmatizer.lemmatize(w, pos=penn_to_wordnet(pos)) )
        return out


    def degree(self, w, s, mode='en'):
        """ compute degree of a word: sum_{w' \in s} is_translation(w, w')
        """
        if mode == 'en':
            return sum(self.dict.is_translation_pair(w, x) for x in s)
        else:
            return sum(self.dict.is_translation_pair(x, w) for x in s)


    def score(self, en_s, ja_s):
        """ compute similarity between en_s and ja_s
            this metric is essentially the number of shared words between the sentences,
            normalized by total number of shared words and length of sentence
        """
        try:
            en_s = self.extract_en_content_lemmas(en_s)
            ja_s = self.extract_ja_content_lemmas(ja_s)

            s = 0
            for en_w in en_s:
                for ja_w in ja_s:
                    s += (self.dict.is_translation_pair(en_w, ja_w) / \
                              (self.degree(en_w, ja_s) * self.degree(ja_w, en_s, mode='ja') or 1))

            s = s * 2.0 / ((len(en_s) + len(ja_s)) or 1)
            return s
        # TODO FIGURE THIS BUG OUT - line 51: " utf8' codec can't decode byte 0x83 "
        except:
            return 0



if __name__ == '__main__':
    # pretty awful unit tests...
    print 'INFO: running sanity checks...'

    print 'INFO: initializing scorer...'
    ps = PairScorer('en_ja_dictionary/raw_kv_pairs', 'rakuten_model_ja.min.json')
    f = open('test_pairs.txt')

    print 'INFO: running tests...'
    for i, l in enumerate(f):
        en = l.strip()
        ja = next(f).strip()

        assert ps.score(en, ja) > 0.25
        print 'SUCESS: sentence %s passed!' % i


