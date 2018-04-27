#calculate BLEU score along with custom metrics for copy span and substitution span (requires path)

import sys
import argparse
import collections

from nltk.translate.bleu_score import corpus_bleu

from pycocoevalcap.bleu import bleu
from pycocoevalcap.meteor import meteor
from pycocoevalcap.rouge import rouge

from fixedthat.preprocessing import ftfy_utils as ut

#use gold paths for gold copy span and gold substitution span
#use derived paths (edit distance) between generated sequence and gold target?

gold_file = sys.argv[1]
translations_file = sys.argv[2]
paths_file = sys.argv[3]

#also just output the percentage of translations that include a word not from the input
#also just output the percentage of translations that include a word from the output that's not in the input

gold_parent = []
gold_ftfy = []
reference_lookup = collections.defaultdict(list)
with open(gold_file) as f:
    for index,line in enumerate(f):
        parent, ftfy = line.strip().split('\t')
        gold_parent.append(parent.split())
        gold_ftfy.append(ftfy.replace('<e>', '').split())
        reference_lookup[index].append(index)
        
translations = [[] for _ in range(len(gold_parent))]
with open(translations_file) as f:
    for line in f:
        translation = line.strip().replace('<e>', '').split()
        index = int(translation[0])
        translations[index] = translation[1:]

count_non_input = 0
count_correct_output = 0

invalid = {'<unk>', '<e>'} | ut.stopwords | set('~!@#$%^&*()_+`-={}|[]\\:";\'<>?,./')

for i in range(len(gold_parent)):
    parent = gold_parent[i]
    ftfy = gold_ftfy[i]
    translation = translations[i]

    non_input = set(translation) - set(parent) - set(invalid)
    if len(non_input):
        #print("\t".join([' '.join(gold_parent[i]), ' '.join(gold_ftfy[i]), ' '.join(translations[i])]))
        count_non_input += 1

    if len(non_input & set(ftfy)):
        count_correct_output += 1

print(count_non_input)
print(count_correct_output)
print(corpus_bleu([[i] for i in gold_ftfy], translations))

references = {}
translations_lookup = {}
for i, ftfy in enumerate(gold_ftfy):
    translations_lookup[i] = [' '.join(translations[i])]
    references[i] = [' '.join(gold_ftfy[j]) for j in reference_lookup[i]]

pairs = [translations_lookup, references]
        
print(rouge.Rouge().compute_score(*pairs))
print(meteor.Meteor().compute_score(*pairs))[0]
print(bleu.Bleu().compute_score(*pairs))[0]
                                           
