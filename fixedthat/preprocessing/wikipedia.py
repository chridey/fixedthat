
def get_categories():
    categories = {}
    with open('categories_full.tsv') as f:
        for line in f:
            rows = line.strip().split('\t')
            categories[rows[0]] = rows[1:]
    return categories

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
    
