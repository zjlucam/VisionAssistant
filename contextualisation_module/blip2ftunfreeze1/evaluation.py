from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

def calculate_bleu(references, hypotheses):
    smoothing_function = SmoothingFunction().method4
    return corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses], 
                       smoothing_function=smoothing_function)

def calculate_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return {key: np.mean([score[key].fmeasure for score in scores]) for key in scores[0]}
