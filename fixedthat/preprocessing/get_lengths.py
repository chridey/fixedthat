import sys

from fixedthat.preprocessing import ftfy_utils as ut

specials = ut.stopwords | set('~!@#$%^&*()_+`-={}|[]\\:";\'<>?,./')

infile = sys.argv[1]

with open(infile) as f:
    for line in f:
        parent, ftfy = line.strip().split('\t')
        parents = set(parent.split())
        ftfys = ftfy.split()

        count = 0
        for word in ftfys:
            if word not in parents: # and word not in specials and word.isalnum():
                count += 1

        '''
        counts = []
        for word in ftfys:
            counts.append(str(count))            
            if word not in parents:
                count -= 1
        counts.append(str(count))
        ftfy += ' ' + ' '.join(counts)
        '''
        ftfy = str(count)
        
        print("\t".join((parent, ftfy)))

        
            
