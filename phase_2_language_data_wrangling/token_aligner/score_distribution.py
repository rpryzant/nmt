"""

=== DESCRIPTION
given a subtitle webpage parse (combined.en, combined.jp),
this file computs the score distribution of the matched
sentences.

=== USAGE
python score_distribution.py [combined.en] [combined.ja] [out]
"""
import sys
from scorer import PairScorer
from tqdm import tqdm
import os

combined_en = open(sys.argv[1])
combined_ja = open(sys.argv[2])

ps = PairScorer('en_ja_dictionary/raw_kv_pairs', 'rakuten_model_ja.min.json')

total = int(os.popen('wc -l %s' % sys.argv[1]).read().strip().split()[0])

s = ','.join(str(ps.score(en.strip(), ja.strip())) for en, ja in tqdm(zip(combined_en, combined_ja), total=total))

out = open(sys.argv[3], 'w')
out.write(s)
out.close()
