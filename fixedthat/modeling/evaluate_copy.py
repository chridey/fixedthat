#calculate BLEU score along with custom metrics for copy span and substitution span (requires path)

from collections import Counter
import sys
import argparse

from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from fixedthat.preprocessing import ftfy_utils as ut

#use gold paths for gold copy span and gold substitution span
#use derived paths (edit distance) between generated sequence and gold target?

gold_file = sys.argv[1]
translations_file = sys.argv[2]

#also just output the percentage of translations that include a word not from the input
#also just output the percentage of translations that include a word from the output that's not in the input

gold_parent = []
gold_ftfy = []
with open(gold_file) as f:
    for line in f:
        parent, ftfy = line.strip().split('\t')
        gold_parent.append(parent.split())
        gold_ftfy.append(map(int, ftfy.split()))
        
translations = [[] for _ in range(len(gold_parent))]
with open(translations_file) as f:
    for line in f:
        translation = line.strip().split()
        index = int(translation[0])
        translations[index] = map(int, translation[1:len(gold_ftfy[index])+1])

y_true = []
y_pred = []
problems = 0

for i in range(len(gold_ftfy)):
    #print('\t'.join([' '.join(gold_parent[i]), ' '.join(map(str, gold_ftfy[i] + ['TAB'] + translations[i]))]))
    
    if len(gold_ftfy[i]) != len(translations[i]):
        problems += 1
        continue
    
    y_true.extend(gold_ftfy[i])
    y_pred.extend(translations[i])
    
print(precision_recall_fscore_support(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print(problems)
print(Counter(y_true))
print(Counter(y_pred))
