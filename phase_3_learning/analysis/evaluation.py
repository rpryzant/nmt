
from nltk.translate import bleu_score
from nltk.translate import ribes_score



def sentence_bleu(reference, hypothesis):
    if hypothesis == []: return 0
    try:
        return bleu_score.sentence_bleu([reference], hypothesis) * 100
    except:
        return 0.0

def multisentence_bleu(references, hypotheses):
    references = [[x] for x  in references]
    return bleu_score.corpus_bleu(references, hypotheses) * 100


def sentence_ribes(reference, hypothesis):
    if hypothesis == []: return 0
    try:
        return ribes_score.sentence_ribes([reference], hypothesis) * 100
    except:
        return 0.0

def multisentence_ribes(references, hypotheses):
    references = [[x] for x  in references]
    return ribes_score.corpus_ribes(references, hypotheses) * 100


def evaluate(ys, yhats):
    bleu = sum(sentence_bleu(r, h) for (r, h) in zip(ys, yhats)) / len(ys)
    ribes = sum(sentence_ribes(r, h) for (r, h) in zip(ys, yhats)) / len(ys)
    return bleu, ribes



