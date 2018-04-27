import numpy as np
import spacy
import HTMLParser
import re

URL_STRING = 'u__URL__u'
USERNAME_STRING = 'u__USERNAME__u'
SUBREDDIT_STRING = 'u__SUBREDDIT__'
HASHTAG_STRING = 'u__HASHTAG__'

try:
    nlp = spacy.en.English()
except AttributeError:
    nlp = spacy.load('en')
    
stopwords = {word.lower_ for word in nlp.vocab if word.is_stop} | {"'re", "'nt", "'s", "'ll", "'d", "'ve", "'m",
                                                                   'ftfy', 'cum', 'cumming', 'fap',
                                                                   URL_STRING.lower(), USERNAME_STRING.lower()}

SUBS_PENALTY = 0.99
INSERT_PENALTY = 1
DEL_PENALTY = 1
COPY_BONUS = 0

def edit_distance(A, B, fuzzy=True,
                  del_penalty=DEL_PENALTY,
                  insert_penalty=INSERT_PENALTY,
                  subs_penalty=SUBS_PENALTY,
                  copy_bonus=COPY_BONUS):
    # For all i and j, d[i,j] will hold the Levenshtein distance between
    # the first i characters of s and the first j characters of t.
    # Note that d has (m+1) x (n+1) values.
    #let d be a 2-d array of int with dimensions [0..m, 0..n]

    d = np.zeros((len(A)+1, len(B)+1))
    bp = np.empty((len(A)+1, len(B)+1), dtype=tuple)

    for i in range(len(A)+1):
        d[i, 0] = i # the distance of any first string to an empty second string
        # (transforming the string of the first i characters of s into
        # the empty string requires i deletions)
    for j in range(len(B)+1):
        if fuzzy:
            d[0, j] = 0
        else:
            d[0,j] = j # the distance of any second string to an empty first string
            
    for j in range(1, len(B)+1):
        for i in range(1, len(A)+1):
            if A[i-1] == B[j-1]:
                d[i, j] = d[i-1, j-1] - copy_bonus        # no operation required
                bp[i,j] = (i-1, j-1)
            else:
                ret = sorted(((d[i-1, j] + del_penalty, (i-1,j)),  # a deletion
                            (d[i, j-1] + insert_penalty, (i,j-1)),  # an insertion
                            (d[i-1, j-1] + subs_penalty, (i-1,j-1))), # a substitution
                            key = lambda x:x[0])
                d[i,j] = ret[0][0]
                bp[i,j] = ret[0][1]
                
    return d, bp

def backtrack(bp, end, do_full_path=False):
    curr = (-1, end)
    full_path = [curr]
    while True:
        if bp[curr] is None:
            break
        curr = bp[curr]
        full_path.append(curr)
        #print(curr)
        
    if do_full_path:
        return full_path
    
    return curr[1]

def find_range(d, bp):
    r = np.argsort(d[-1])
    i=0
    min_r = d[-1][r[0]]
    #print(r)
    #print(d[-1])
    #while row['parent_body'][r[i]-1].isalnum() and d[-1][r[i]] == min_r:

    max_length = 0
    best = None
    #break ties by taking longest substring
    #further break ties by taking substrings closest to a boundary
    while i < r.shape[0] and d[-1][r[i]] == min_r:
        start = backtrack(bp, r[i])
        #print('s', start, 'e', r[i])
        if r[i] - start > max_length:
            max_length = r[i] - start
            best = (start, r[i])
        i+=1

    if not max_length:
        best = (0, d.shape[1])
        
    return best

def iterMetadata(doc):
    for line in doc.split('\n'):
        if not line.strip():
            continue

        if type(line) != unicode:
            line = unicode(line, encoding='utf-8')
        
        parsed = nlp(line)
        offset = 0
        for sentence in parsed.sents:
            metadata = dict(deps=[],
                            lemmas=[],
                            pos=[],
                            words=[],
                            orig=str(sentence))
        
            for index,word in enumerate(sentence):
                metadata['lemmas'].append(word.lemma_)
                metadata['pos'].append(word.tag_)
                metadata['words'].append(unicode(word))
            
                head = word.head.i-offset
                if word.dep_ == 'ROOT':
                    head = -1
                metadata['deps'].append((head, index, word.dep_))
            
            offset += len(sentence)
            yield metadata

def clean_text(text):
        
    text = HTMLParser.HTMLParser().unescape(text)

    #replace consecutive whitespace with a single space
    text = re.sub('\s{2,}', ' ', text)
    
    #replace bracketed links with the associated text
    text = re.sub('\[(.*?)\]\((.*?)\)', r'\1', text)

    #replace standalone URLs with a URL token
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                  URL_STRING, text)

    #replace usernames with a username token
    text = re.sub('\/u\/(\S*)', USERNAME_STRING, text)
    
    #replace subreddits with a different format
    text = re.sub('\/r\/(\S*)', SUBREDDIT_STRING + r'\1', text)    

    #replace hashtags with a different format
    text = re.sub('#(\S+)', HASHTAG_STRING + r'\1', text)

    #remove reddit formatting
    text = text.replace('*', '')
    text = text.replace('^', '')    

    if len(text):
        if text[0] == '|':
            text = text[1:]
    if len(text):            
        if text[-1] == '|':
            text = text[:-1]

    if len(text):
        if text[0] == '>':
            text = text[1:]
            
    return text.strip()

def tokenize(ftfy):
    #ftfy = ftfy.replace('&gt;', ' ').strip()
    ftfy = clean_text(ftfy)
    
    return list(iterMetadata(ftfy))

def get_ftfy_index(metadata):
    ftfy_index = None
    for index,sent_metadata in enumerate(metadata):
        if 'ftfy' in ' '.join(sent_metadata['words']).lower() and ftfy_index is None:
            return index

def join_sentences(sentences, skip_strikethrough=False):
    text = []
    for index, sentence in enumerate(sentences):
        if skip_strikethrough:
            words = []
            in_skip = False
            curr_tildes = ''
            for word in sentence['words']:
                if not in_skip:
                    if word.startswith('~~'):
                        if not word.endswith('~~'):
                            in_skip = True
                    elif not word.endswith('~~'):
                        words.append(word)
                elif word.endswith('~~'):
                    in_skip = False
            text.extend(words)
        else:
            text.extend(sentence['words'])
    return text

def split_ftfy(tokenized_ftfy):
    
    for index, word in enumerate(tokenized_ftfy['words']):
        if 'ftfy' in word.lower():
            break

    start = 1
    if index == 0:
        if len(tokenized_ftfy['words']) > 1 and tokenized_ftfy['words'][1] in (';', ':', '-', ',', '.'):
            index = 2
        else:
            index = 1
        start = 0
    else:
        if tokenized_ftfy['words'][index-1] in (':', '-', ',', '#', '*', ';'):
            index -= 1
            
    ret = [{}, {}]
    for key in tokenized_ftfy:
        ret[0][key] = tokenized_ftfy[key][:index]
        ret[1][key] = tokenized_ftfy[key][index:]

    return start, ret

def get_search_string(ftfy, ftfy_index):
    if ftfy_index == 0:
        ftfy_index, ftfy_split = split_ftfy(ftfy[0])
        ftfy = ftfy_split + ftfy[1:]
        #print(ftfy_index, ftfy)
        
    if ftfy_index == 0:
        start_range = 1
        end_range = len(ftfy)
    else:
        start_range = 0
        end_range = ftfy_index
        
    return ftfy[start_range:end_range]
    
def get_sentence_alignment(ftfy, ftfy_index, parent):
    
    search = join_sentences(get_search_string(ftfy, ftfy_index))

    best_min = 2**32            
    best_index = None
    best_range = None

    if len(parent) == 0 or len(search) == 0:
        #return best_index, best_min, best_range
        return best_min, best_range

    text = join_sentences(parent)
        
    d, bp = edit_distance(search, text, True)
        
    if min(d[-1]) < best_min:
        return min(d[-1]), find_range(d,bp)
    
    '''
    for index in range(max(1,len(parent)-(end_range-start_range)+1)):
        text = join_sentences(parent[index:index+(end_range-start_range)])
        
        d, bp = edit_distance(search, text, True)
        
        if min(d[-1]) < best_min:
            best_index = index
            best_min = min(d[-1])
            best_range = find_range(d, bp)
         
    return best_index, best_min, best_range
    '''

def get_nearest_boundary(parent, parent_range):
    ctr = 0
    left_boundary = 0
    right_boundary = None
    for i in parent:
        ctr += len(i['words'])
        if ctr <= parent_range[0]:
            left_boundary = ctr
        if ctr >= parent_range[1] and right_boundary is None:
            right_boundary = ctr-1
            
    return left_boundary, right_boundary    
    
def find_exact_ftfy_boundaries(ftfy, ftfy_index, parent, parent_range, pad_to_boundary=True):
    if parent_range is None:
        return (0,0), parent_range

    left_boundary, right_boundary = get_nearest_boundary(parent, parent_range)
    
    ftfy = get_search_string(ftfy, ftfy_index)
    ftfy = join_sentences(ftfy, True)

    if not len(ftfy):
        return (0,0), parent_range
    
    #print(len(ftfy))
    parent = join_sentences(parent)
    #print(len(parent))
    
    ftfy_start = 0
    ftfy_end = len(ftfy)-1

    parent_start, parent_end = parent_range
    parent_end -= 1
    
    #print(ftfy_start, ftfy_end, parent_start, parent_end)    
    while ftfy_start < len(ftfy) and parent_start < len(parent):
        #print(ftfy_start, parent_start)
        #print(ftfy[ftfy_start], parent[parent_start], ftfy[ftfy_start].lower() == parent[parent_start].lower())
        
        if not len(ftfy[ftfy_start].strip()):
            ftfy_start += 1
            
        if not len(parent[parent_start].strip()):
            parent_start += 1

        if ftfy_start < len(ftfy) and parent_start < len(parent) and ftfy[ftfy_start].lower() != parent[parent_start].lower():
            break
        
        ftfy_start += 1
        parent_start += 1

                            
    #print(ftfy_start, parent_start)
    ftfy_end = max(ftfy_start, ftfy_end)
    parent_end = max(parent_start, parent_end)
                
    while ftfy_end >= 0 and parent_end >= 0 and ftfy_end < len(ftfy) and parent_end < len(parent):
        #print(ftfy_end, parent_end)
        #print(ftfy[ftfy_end], parent[parent_end], ftfy[ftfy_end].lower() == parent[parent_end].lower())
        
        if not len(ftfy[ftfy_end].strip()):
            ftfy_end -= 1
            
        if not len(parent[parent_end].strip()):
            parent_end -= 1
                   
        if ftfy_end >= 0 and parent_end >= 0 and ftfy[ftfy_end].lower() != parent[parent_end].lower():
            break
        
        ftfy_end -= 1
        parent_end -= 1
            
    #print(ftfy_end, parent_end)
    ftfy_end = max(0, ftfy_start-1, ftfy_end)
    parent_end = max(0, parent_start-1, parent_end)    
        
    #handle sentence boundaries
    if pad_to_boundary: #TODO- pad to subtree boundary
        #if the start of the ftfy and parent alignment do not match, extend to previous sentence boundary
        while parent_range[0] < len(parent) - 1 and not len(parent[parent_range[0]].strip()):
            parent_range[0] += 1
        idx = 0
        while idx < len(ftfy)-1 and not len(ftfy[idx].strip()):
            idx += 1
        #print(idx, len(ftfy), parent_range[0], len(parent))
        if ftfy[idx].lower() != parent[parent_range[0]].lower():
            parent_start = left_boundary

        if parent_end > parent_range[1]-1:
            parent_range[1] = min(parent_end + 1, len(parent))
        while parent_range[1]-1 > 0 and not len(parent[parent_range[1]-1].strip()):
            parent_range[1] -= 1
        idx = len(ftfy)-1
        #print((ftfy_start,ftfy_end+1), (parent_start,parent_end+1))
        #print(idx, ftfy[idx])
        while idx > 0 and not len(ftfy[idx].strip()):
            idx -= 1
        #print(idx, ftfy[idx], parent_range[1]-1, parent[parent_range[1]-1])        
        if ftfy[idx].lower() != parent[parent_range[1]-1].lower():
            parent_end = right_boundary
    
    return (ftfy_start,ftfy_end+1), (parent_start,parent_end+1)

def get_sentence_labels(parent, parent_range):
    i = 0
    labels = []
    
    for sentence in parent:
        length = len(sentence['words'])
        sentence_labels = []
        for j in range(i, i+length):
            if j >= parent_range[0] and j <= parent_range[1]:
                sentence_labels.append(1)
            else:
                sentence_labels.append(0)
        i += length
        labels.append(sentence_labels)
        
    return labels

def filter_ftfy(ftfy, parent):
    
    MAX_LEN = 50
    if len(parent) > MAX_LEN or len(ftfy) > MAX_LEN:
        return False

    ftfy_chars = ''.join(ftfy)

    #handle foreign language
    #TODO: langid classifier
    #for now, just use if non-ascii > 10% of ftfy
    if sum(filter(lambda x: 32 <= ord(x) <= 126, ftfy_chars)) < .9 * len(ftfy_chars):
        return False

    #handle jokes and typos (also takes care of simple transpositions)
    #do this for all (min(N, len(ftfy)+1) choose 2) character similarity > 66%
    #66% chosen to handle cases like your -> you're or their -> there
    ftfy_set = set(ftfy_chars.lower())
    for i in range(1, len(ftfy)+1):
        for j in range(len(parent)-i):
            parent_set = set(''.join(parent[j:j+i].lower()))
            if len(parent_set & ftfy_set) > .66 * len(ftfy_set):
                return False

    return True

def get_num_ascii(ftfy):
    ftfy_chars = ''.join(ftfy)

    #handle foreign language
    #TODO: langid classifier
    #for now, just use if non-ascii > 10% of ftfy
    return sum(map(lambda x: 32 <= ord(x) <= 126, ftfy_chars))

def get_num_non_ascii(ftfy):
    ftfy_chars = ''.join(ftfy)
    if not len(ftfy_chars):
        return 0
    #handle foreign language
    #TODO: langid classifier
    #for now, just use if non-ascii > 10% of ftfy
    return 1. * sum(map(lambda x: 32 > ord(x) or ord(x) > 126, ftfy_chars)) / len(ftfy_chars)

def get_max_overlap(ftfy, parent):
#handle jokes and typos (also takes care of simple transpositions)
    #do this for all (min(N, len(ftfy)+1) choose 2) character similarity > 66%
    #66% chosen to handle cases like your -> you're or their -> there
    ftfy_set = set(''.join(ftfy).lower())
    if not len(ftfy_set):
        return 0
    max_overlap = 0
    for i in range(1, len(ftfy)+1):
        for j in range(len(parent)-i+1):
            parent_set = set(''.join(parent[j:j+i]).lower())
            if len(parent_set & ftfy_set) > max_overlap:
                max_overlap = len(parent_set & ftfy_set)

    return 1. * max_overlap / len(ftfy_set)        

def get_max_char_edit(ftfy, parent):
    ftfy = ''.join(ftfy).lower()
    if not len(ftfy):
        return 0
    parent = ''.join(parent).lower()

    d, bp = edit_distance(ftfy, parent)
    return 1.*min(d[-1]) / len(ftfy)

def get_all_sentence_labels(ftfy, ftfy_index, parent, parent_range, pad_to_boundary=True):
    if parent_range is None:
        return None, None, None

    left_boundary, right_boundary = get_nearest_boundary(parent, parent_range)
    
    ftfy = get_search_string(ftfy, ftfy_index)
    ftfy = join_sentences(ftfy, True)

    if not len(ftfy):
        return None, None, None
    
    parent = join_sentences(parent)
    
    ftfy_start = 0
    ftfy_end = len(ftfy)-1

    parent_start, parent_end = parent_range
    parent_end -= 1

    #label types
    OTHER = 0
    COPY = 1
    SUB = 2
    INSERT = 3
    DELETE = 4
    PAD = 5
    labels = [OTHER]*(len(parent))#+1)
        
    while ftfy_start < ftfy_end and parent_start < parent_end and ftfy[ftfy_start].lower() == parent[parent_start].lower():
        #if labels[parent_start] != 1:
        #    labels[parent_start] = 3
        #labels[parent_start+1] = 1
        labels[parent_start] = COPY
        ftfy_start += 1
        parent_start += 1
                
    while ftfy_end > ftfy_start and parent_end > parent_start and ftfy_end < len(ftfy) and parent_end < len(parent) and ftfy[ftfy_end].lower() == parent[parent_end].lower():
        #if labels[parent_end] != 1:
        #    labels[parent_end] = 3
        #labels[parent_end+1] = 1
        labels[parent_end] = COPY
        ftfy_end -= 1
        parent_end -= 1
                    
    #handle sentence boundaries
    if pad_to_boundary: #TODO- pad to subtree boundary
        #if the start of the ftfy and parent alignment do not match, extend to previous sentence boundary
        idx = 0
        if ftfy[idx].lower() != parent[parent_range[0]].lower():
            parent_start = left_boundary
            for i in range(left_boundary, parent_range[0]):
                #if labels[i] not in (1,3):
                #    labels[i] = 2
                if labels[i] != COPY:
                    labels[i] = PAD
                    
        idx = len(ftfy)-1
        if ftfy[idx].lower() != parent[parent_range[1]-1].lower():
            parent_end = right_boundary
            for i in range(parent_range[1]+1, right_boundary+1):
                #if labels[i] not in (1,3):
                #    labels[i] = 2
                if labels[i] != COPY:
                    labels[i] = PAD

    #for i in range(parent_start, parent_end+2):
        #if labels[i] not in (1,2):
        #    labels[i] = 3
    for i in range(parent_start, parent_end+1):        
        if labels[i] not in (COPY,PAD):
            labels[i] = SUB

    if parent_start == parent_end+1:
        labels[i] == INSERT

    if ftfy_start==ftfy_end+1:
        for i in range(parent_start, parent_end+1):
            if labels[i] not in (COPY,PAD):
                labels[i] = DELETE
            
    return (ftfy_start,ftfy_end+1), (parent_start,parent_end+1), labels

    
def has_all_stop_words(ftfy, ftfy_index=None, ftfy_range=None):
    if ftfy_index is not None:
        ftfy = map(lambda x:x.lower(),
                join_sentences(get_search_string(ftfy, ftfy_index), True))

    if ftfy_range is not None:
        ftfy = ftfy[ftfy_range[0]:ftfy_range[1]]
        
    for word in ftfy:
        if word not in stopwords and word.isalpha():
            return False

    return True
    
def get_ftfy(metadata):
    ftfy = metadata['ftfy_metadata']
    ftfy_index = metadata['ftfy_index']
    return join_sentences(get_search_string(ftfy, ftfy_index), True)

def get_parent_window(metadata, padded=False):
    parent = metadata['{}_metadata'.format(metadata['best'])]
    if padded:
        left_boundary, right_boundary = get_nearest_boundary(parent, metadata['best_range'])
        best_range = left_boundary, right_boundary+1
    else:
        best_range = metadata['best_range']
    
    return join_sentences(parent)[best_range[0]:best_range[1]]

def get_adjusted_range(metadata):
    parent = metadata['{}_metadata'.format(metadata['best'])]
    left_boundary, right_boundary = get_nearest_boundary(parent, metadata['best_range'])
    
    best_range = metadata['best_range']
    
    return best_range[0]-left_boundary, best_range[1]-left_boundary
    
def count_aligned_words(transitions):
    #if there is at least one aligned word and it's not a stop word
    pass
