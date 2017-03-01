"""
=== DESCRIPTION
This script uses scores from score_distribution.py to clean out 
  potentially misaligned sentences from a corpus.

=== USAGE
python corpus_cleaner.py [en corpus] [ja corpus] [distribution file]
"""
import sys

en = open(sys.argv[1])
ja = open(sys.argv[2])
distribution = [float(x) for x in open(sys.argv[3]).read().split(',')]

en_out = ''
ja_out = ''
for (en_s, ja_s, score) in zip(en, ja, distribution):
    if score > 0:
        en_out += en_s
        ja_out += ja_s

f = open(sys.argv[1] + '.cleaned', 'w')
f.write(en_out)
f.close()

f = open(sys.argv[2] + '.cleaned', 'w')
f.write(ja_out)
f.close()