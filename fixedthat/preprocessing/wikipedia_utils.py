import collections
import gzip
import csv

def normalize_title(title):
    start = title.find('(')
    end = title.find(')')
    if start != -1 and end != -1:
        title = title[:start] + title[end+1:]
    title = title.replace('__', '_')
    if title[-1:] == '_':
        title = title[:-1]
    return title.lower()

def get_categories():
    categories = {}
    with open('categories_full.tsv') as f:
        for line in f:
            rows = line.strip().split('\t')
            categories[rows[0]] = rows[1:]
    return categories

def get_pairs(link_lookup, terms1, terms2):
    expanded_terms1 = set()
    for term in terms1:
        expanded_terms1.add(term)
        expanded_terms1.update(link_lookup[term])

    expanded_terms2 = set()
    for term in terms2:
        expanded_terms2.add(term)
        expanded_terms2.update(link_lookup[term])

    return expanded_terms1 & expanded_terms2 #len(expanded_terms1 & expanded_terms2) > 0

def build_link_lookup(pagefile, redirectfile, linkfile=None):
    print('reading...')

    lookup = collections.defaultdict(list)
    page_ids = set()
    with gzip.open(pagefile) as f:
        for idx,line in enumerate(f):
            print(idx)
            if not line.startswith("INSERT INTO `page` VALUES"):
                continue
            
            for values in line.split('),('):
                values = values.replace("INSERT INTO `page` VALUES (", '').split(',')
                if int(values[1]) != 0:
                    continue
                title = normalize_title(values[2][1:-1])
                try:
                    title.encode('utf-8')
                except Exception:
                    print('unicode problem')
                    continue
                lookup[title].append(int(values[0]))
                page_ids.add(int(values[0]))
                
    with gzip.open(redirectfile) as f:
        for idx,line in enumerate(f):
            print(idx)
            if not line.startswith("INSERT INTO `redirect` VALUES"):
                continue
            
            for values in line.split('),('):
                values = values.replace("INSERT INTO `redirect` VALUES (", '').split(',')
                if int(values[1]) != 0:
                    continue
                
                title = normalize_title(values[2][1:-1])
                try:
                    title.encode('utf-8')
                except Exception:
                    print('unicode problem')                    
                    continue
                lookup[title].append(int(values[0]))
                page_ids.add(int(values[0]))

    link_lookup = collections.defaultdict(list)
    if linkfile is not None:
        with gzip.open(linkfile) as f:
            for idx,line in enumerate(f):
                print(idx)
                if not line.startswith("INSERT INTO `pagelinks` VALUES"):
                    continue

                for values in csv.reader(line.split('),(')):
                    if len(values) < 3:
                        print(values)
                        continue

                    values[0] = values[0].replace("INSERT INTO `pagelinks` VALUES (", '') #.split(',')
                    if len(values) < 1:
                        print(values)
                        continue
                    if int(values[1]) != 0:
                        continue

                    #print(values)
                    from_index = int(values[0])
                    to_title = normalize_title(values[2][1:-1])
                    if to_title not in lookup:
                        ##UNDO print(to_title)
                        continue
                    if from_index not in page_ids:
                        continue

                    to_index = lookup[to_title]

                    link_lookup[from_index].extend(to_index)
                    for ti in to_index:
                        link_lookup[ti].append(from_index)                

    print(len(lookup))
    print(len(link_lookup))

    return lookup, link_lookup

def segment_collocations(titles, claim, claim_lemmas, claim_pos, collocations_only=False):
        indices = set()
        collocations = {}
        #print('PARENT')                                                                                                                                                                                                                     
        a = tuple(map(lambda x:x.lower(), claim))
        for i in list(range(2,len(a)+1))[::-1]:
            j = 0
            while j <= len(a)-i:
                r = set(range(j, j+i))
                if len(r & indices):
                    j += 1
                    continue
                if '_'.join(a[j:j+i]) in titles:
                    collocations[min(r)] = a[j:j+i]
                    indices.update(r)
                    j += i
                else:
                    j += 1
        
        for ix,(w,l,p) in enumerate(zip(claim,claim_lemmas, claim_pos)):
            #print(p[:2], w.lower(), (w.lower(),) in titles)                                                                                                                                                                                 
            if ix in indices:
                continue
            if p[:2] == 'NN' and w.lower() in titles: #(w.lower(),) in titles:                                                                                                                                                               
                collocations[ix] = [w.lower()]
                indices.add(ix)
                continue

            #print(p[:2], l.lower(), (l.lower(),) in titles)                                                                                                                                                                                 
            if p[:2] in ('NN',) and l.lower() in titles: #(l.lower(),) in titles:                                                                                                                                                            
                collocations[ix] = [l.lower()]
                indices.add(ix)

        ret = []
        ix = 0
        while ix < len(claim):
            if ix in collocations:
                ret.append('_'.join(collocations[ix]))
                ix += len(collocations[ix])
                continue
            elif not collocations_only:
                ret.append(claim[ix])
            ix += 1
            
        return ret

