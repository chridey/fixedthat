import math
import csv
import json
import itertools

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib

import gensim

from scipy.spatial.distance import cosine
import numpy as np

from fixedthat.preprocessing import ftfy_utils as ut

subset_keys = [['parent_length'], ['ftfy_length'], ['char_edit'], ['overlap'], ['non_ascii'], ['copy_pct']]
subset_keys_full = [['full_sim'], ['align_sim'], ['full_parent_vocab'],
                   ['full_ftfy_vocab'], ['align_parent_vocab'], ['align_ftfy_vocab'],
                   list(map(lambda x: 'fp{}'.format(x), range(50))),
                   list(map(lambda x: 'ff{}'.format(x), range(50))),
                   list(map(lambda x: 'ap{}'.format(x), range(50))),
                   list(map(lambda x: 'af{}'.format(x), range(50)))]
feature_keys = list(itertools.chain(*(subset_keys + subset_keys_full)))

def load_word_vectors(filename, small=False):
    print('loading word vectors...')
    print(filename)
    return gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)

def load_training(trainfile):
    data = []
    with open(trainfile) as f:
        reader = csv.DictReader(f, delimiter='\t')
        print(reader.fieldnames)
        for line in reader:
            label = line['claim'] == '1'
            joke = line['joke'] == '1'
            typo = line['typo'] == '1'
            nothing = line['nothing'] == '1'
            more_context = line['more context'] == '1'
            picture = line['picture'] == '1'

            try:
                j = json.loads(line['json'])
                j.update(dict(parent=line['parent'].split()[1:],
                              ftfy=line['ftfy'].split()[1:],
                              label=label,
                              joke=joke,
                              typo=typo,
                              nothing=nothing,
                              more_context=more_context,
                              picture=picture,
                              copy=map(int,line['copy'].split())))
                data.append(j)
            except Exception:
                print(line)

    return pd.DataFrame(data)

def valid_row(row, include_stop=False, ratio=9, max_parent_length=50, max_ftfy_length=50,
              min_parent_length=1, min_ftfy_length=1):
    
    if (not include_stop and ut.has_all_stop_words(row['ftfy'], None, row['ftfy_range'])) \
        or 1. * len(row['parent']) / len(row['ftfy']) > ratio or len(row['parent']) > max_parent_length \
        or len(row['ftfy']) > max_ftfy_length or len(row['parent']) < min_parent_length \
        or len(row['ftfy']) < min_ftfy_length:
        return False
    
    return True

def make_predictions(features, model, include_stop=False, ratio=9, max_parent_length=50, max_ftfy_length=50,
                    min_parent_length=4, min_ftfy_length=2, filter_features=True):

    predictions = model.predict(features[feature_keys])

    if not filter_features:
        return predictions

    for idx, (_,row) in enumerate(features.iterrows()):
        if not valid_row(row, include_stop, ratio, max_parent_length, max_ftfy_length,
                        min_parent_length, min_ftfy_length):
            predictions[idx] = 0

    return predictions
    
def get_features(df, word_vectors, filter_threshold=False):
    features = []

    for idx,row in df.iterrows():
        if filter_threshold and not valid_row(row):
            continue
        feature = dict(parent_length=len(row['parent'])/50.,
                       ftfy_length=len(row['ftfy'])/50.,
                       #ratio=1. * len(row['ftfy']) / len(row['parent']),
                       char_edit=row['max_char_edit'],
                       overlap=row['max_overlap'],
                       non_ascii=row['non_ascii'],
                      copy_pct = 1.*sum(row['copy'])/len(row['parent']))
        
        #also add sum of copy
        #percentage of vocab words? (using word2vec vocab)
        #word2vec similarity (this would be better since the OOV words would not show up)
        #avg overall as well as aligned region
        #avg vector as features - TODO
        parent = ut.join_sentences(row['{}_metadata'.format(row['best'])])
        if filter_threshold and math.isnan(row['ftfy_index']):
            continue
        ftfy = ut.join_sentences(ut.get_search_string(row['ftfy_metadata'], int(row['ftfy_index'])), True)
        parent_start, parent_end = row['parent_range']
        ftfy_start, ftfy_end = row['ftfy_range']

        feature['full_sim'] = 0
        feature['align_sim'] = 0
        feature['full_parent_vocab'] = 0
        feature['full_ftfy_vocab'] = 0
        feature['align_parent_vocab'] = 0 
        feature['align_ftfy_vocab'] = 0
        
        if len(parent) and len(ftfy):
            parent_vectors = np.array([word_vectors[i.lower()] if i.lower() in word_vectors else np.zeros_like(word_vectors['.']) for i in parent])
            ftfy_vectors = np.array([word_vectors[i.lower()] if i.lower() in word_vectors else np.zeros_like(word_vectors['.']) for i in ftfy])

            feature['full_sim'] = cosine(parent_vectors.mean(axis=0), ftfy_vectors.mean(axis=0))
            if parent_end-parent_start != 0 and ftfy_end-ftfy_start != 0:
                feature['align_sim'] = cosine(parent_vectors[parent_start:parent_end].mean(axis=0), 
                                        ftfy_vectors[ftfy_start:ftfy_end].mean(axis=0))
        
            parent_vocab = [1 if i.lower() in word_vectors else 0 for i in parent]
            ftfy_vocab = [1 if i.lower() in word_vectors else 0 for i in ftfy]
        
            feature['full_parent_vocab'] = 1.* sum(parent_vocab)/len(parent_vocab)
            feature['full_ftfy_vocab'] = 1.* sum(ftfy_vocab)/len(ftfy_vocab)
            if parent_end-parent_start != 0:
                feature['align_parent_vocab'] = 1.* sum(parent_vocab[parent_start:parent_end])/(parent_end-parent_start)
            if ftfy_end-ftfy_start != 0:
                feature['align_ftfy_vocab'] = 1.* sum(ftfy_vocab[ftfy_start:ftfy_end])/(ftfy_end-ftfy_start)
        
        feature.update({'fp{}'.format(i):v for i,v in enumerate(parent_vectors.mean(axis=0))})
        feature.update({'ff{}'.format(i):v for i,v in enumerate(ftfy_vectors.mean(axis=0))})
        feature.update({'ap{}'.format(i):v for i,v in enumerate(parent_vectors[parent_start:parent_end].mean(axis=0))})
        feature.update({'af{}'.format(i):v for i,v in enumerate(ftfy_vectors[ftfy_start:ftfy_end].mean(axis=0))})

        feature.update(dict(copy=row['copy'], ftfy=row['ftfy'], parent=row['parent'], label=row['label'],
                            ftfy_range=row['ftfy_range'], parent_range=row['parent_range']))
        
        features.append(feature)

    features = pd.DataFrame(features)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    return features


def train_model(features, labels, params, save_file=None):
    model_type = params.pop('model_type')
    clf = model_type(**params)
    clf.fit(features, labels)

    if save_file is not None:
        joblib.dump(clf, save_file)

    return clf

def load_model(small=False):
    print('loading model...')
    filename = 'data_quality_model'
    if small:
        filename = 'data_quality_model_50'
    return joblib.load(filename)

def validate_model(features, labels):

    split_size = len(labels) // 2

    precision = sum((1.*sum(labels[:split_size])/split_size, 1.*sum(labels[split_size:])/split_size))/2
    print(precision)
    baseline = 2*precision/(1+precision)
    print(baseline)

    penalties = ['l1', 'l2']
    Cs = [10000, 1000, 100, 10, 1, .1, .01, .001]
    kernels = ['linear', 'rbf', 'poly']
    class_weights = [None, 'balanced']
    pls = [4]
    fls = [2]

    '''
    model_types = [(LogisticRegression, {'penalty':'l1'}),
                   (LogisticRegression, {'penalty':'l2'}),
                   (SVC, {'kernel':'linear'}),
                   (SVC, {'kernel':'rbf'}),
                   (SVC, {'kernel':'poly'})]
    '''
    model_types = [(LogisticRegression, {'penalty':'l2'})]
    class_weights = [None]
    Cs = [.01]
    
    best = []
    for model_type, extra_params in model_types:
        for C in Cs:
            for class_weight in class_weights:
                print(model_type, extra_params, C, class_weight)
                #clf = SVC(kernel=kernel, C=C, class_weight=class_weight)
                #clf = SVC(kernel='linear', C=0.01)
                #clf = SVC(kernel=kernel, C=C, class_weight=class_weight)

                clf = model_type(C=C, class_weight=class_weight, **extra_params)
                    
                print('training model...')
                clf.fit(features[feature_keys][:split_size], labels[:split_size])
                print('making predictions...')
                predictions = make_predictions(features[split_size:], clf)
                #predictions = clf.predict(features[split_size:])
                #predictions *= np.array(~((features[split_size:]['parent_length'] < pl/50.) | (features[split_size:]['ftfy_length'] < fl/50.)))
                s1 = sum(predictions)
                p1, r1, f1, s = precision_recall_fscore_support(labels[split_size:], predictions)

                clf.fit(features[feature_keys][split_size:], labels[split_size:])
                predictions = make_predictions(features[:split_size], clf)
                #predictions = clf.predict(features[:split_size])
                #predictions *= np.array(~((features[:split_size]['parent_length'] < pl/50.) | (features[:split_size]['ftfy_length'] < fl/50.)))
                s2 = sum(predictions)
                p2, r2, f2, s = precision_recall_fscore_support(labels[:split_size], predictions)

                params = dict(extra_params)
                params.update({'C': C, 'class_weight': class_weight, 'model_type': model_type})
                                
                best.append([(f1+f2)[1]/2, (p1+p2)[1]/2, (r1+r2)[1]/2, (s1+s2)/2., params])
        
    print('DONE')
    score, precision, recall, predictions, params = sorted(best, key=lambda x:x[0])[-1]
    print(score, precision, recall, predictions, params)

    return params

def main(trainfile, word_vectors_file, save_file=None):
    df = load_training(trainfile)
    word_vectors = load_word_vectors(word_vectors_file)
    features = get_features(df, word_vectors, filter_threshold=True)
        
    print(features.shape)
    print(features[feature_keys].shape)
    print(features['label'].sum())
    params = validate_model(features, features['label'])

    train_model(features[feature_keys], features['label'], params, save_file=save_file)


if __name__ == '__main__':
    import sys

    trainfile = sys.argv[1]
    word_vectors_file = sys.argv[2]
    save_file = None
    if len(sys.argv) > 3:
        save_file = sys.argv[3]
    main(trainfile, word_vectors_file, save_file)
