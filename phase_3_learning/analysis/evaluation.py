
from nltk.translate import bleu_score
from nltk.translate import ribes_score


def depad(s):
    """ removes start, end, and pad characters from sentences
    """



def sentence_bleu(hypothesis, reference):
    return bleu_score.sentence_bleu([reference], hypothesis) * 100


def multisentence_bleu(hypotheses, references):
    return bleu_score.corpus_bleu(references, hypotheses) * 100


def sentence_ribes(hypothesis, reference):
    return ribes_score.sentence_ribes([reference], hypothesis) * 100

def multisentence_ribes(hypothesis, reference):
    return ribes_score.corpus_ribes(reference, hypothesis) * 100
