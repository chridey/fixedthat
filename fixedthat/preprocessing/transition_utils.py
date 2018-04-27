def path2copy(path, parent, parent_range, ftfy):
    #print(parent, parent_range, ftfy)
    copy = [0] * len(parent)
    
    parent = ['BOS'] + parent[parent_range[0]:parent_range[1]]
    ftfy = ['BOS'] + ftfy

    if path is None or path[-1] is None:
        return copy
    
    path[-1][0] = len(parent)-1
    path[-1][1] = len(ftfy)-1
    
    for idx,node in enumerate(path[1:]):
        #print(idx, node, parent[node[0]].lower(), ftfy[node[1]].lower())
        if node[0] < len(parent) and node[1] < len(ftfy):
            if parent[node[0]].lower() == ftfy[node[1]].lower():
                copy[parent_range[0]+node[0]-1] = 1
            elif copy[parent_range[0]+node[0]-1] != 1:
                copy[parent_range[0]+node[0]-1] = 0

    #print(copy)
    return copy
    
def path2segments(path, parent, parent_range, ftfy, verbose=False):
    parent = ['BOS'] + parent[parent_range[0]:parent_range[1]]
    ftfy = ['BOS'] + ftfy
    
    on_path = []
    off_path = []

    if path is None or path[-1] is None:
        return [[]], [[]]
    
    path[-1][0] = len(parent)-1
    path[-1][1] = len(ftfy)-1

    previous = None
    #TODO
    
    for idx,node in enumerate(path[1:]):
        if parent[node[0]].lower() == ftfy[node[1]].lower():
            on_path[-1].append(ftfy[node[1]].lower())
        else:
            off_path[-1].append(ftfy[node[1]].lower())

    if path[1][1] not in (1,-1):
        off_path.insert(0, ftfy[1:path[1][1]])
        
    return transitions

def path2transitions(path, parent, parent_range, ftfy, verbose=False):
    leftovers = parent[parent_range[1]:]
    parent = ['BOS'] + parent[parent_range[0]:parent_range[1]]
    ftfy = ['BOS'] + ftfy
    
    #shift until we find an aligned word, then generate/delete from the stack, then pop all, then shift, then copy
    transitions = []
    stack = []
    buffer = []

    if path is None or path[-1] is None:
        return []
    
    path[-1][0] = len(parent)-1
    path[-1][1] = len(ftfy)-1
    
    for idx,node in enumerate(path[1:]):
        if verbose:
            print(node, parent[node[0]], ftfy[node[1]])
        if parent[node[0]].lower() == ftfy[node[1]].lower():
            if len(stack) and stack[-1] == node[0]:
                stack.pop()
                transitions.pop()

            if len(buffer) and buffer[-1] == node[1]:
                buffer.pop()
                
            while len(buffer):
                transitions.append(ftfy[buffer.pop(0)].lower())
                
            if len(stack):
                transitions.append('REDUCE')
                stack = []
                
            transitions.append('SHIFT')
            transitions.append('COPY-REDUCE')
        else:
            curr_ftfy = path[idx+1][1]
            #if curr_ftfy == -1:
            #    curr_ftfy = len(ftfy) - 1
            curr_parent = path[idx+1][0]
            #if curr_parent == -1:
            #    curr_parent = len(parent) - 1

            if verbose:
                print(curr_parent, curr_ftfy)
            
            if path[idx][1] != curr_ftfy:
                buffer.append(node[1])
            if path[idx][0] != curr_parent: # and path[idx+2][0] != curr_parent:
                transitions.append('SHIFT')                
                stack.append(node[0])

    #if there is anything left at the end, add it
    if len(buffer):
        transitions += map(lambda x: ftfy[x].lower(), buffer)

    #if path[1][0] == -1:
    #    parent_start = parent_range[0] + len(parent) - 1
    #else:
    parent_start = parent_range[0] + path[1][0]
        
    if parent_start not in (1,-1):
        transitions = ['SHIFT'] * (parent_start-1) + ['REDUCE'] + transitions
            
    #if parent_range[1] > parent_range[0] + len(parent):
    #    transitions += ['SHIFT'] * (parent_range[1] - parent_range[0] + len(parent)) + ['REDUCE']

    if len(leftovers):
        transitions += ['SHIFT'] * len(leftovers) + ['REDUCE']
        
    if path[1][1] not in (1,-1):
        transitions = map(lambda x:x.lower(), ftfy[1:path[1][1]]) + transitions
        
    return transitions

def transitions2ftfy(transitions, parent, verbose=False, ignore_invalid=False):
    output = []
    stack = []

    idx = 0
    for op in transitions:
        if verbose:
            print(op, idx)
        if op == 'SHIFT':
            if ignore_invalid and idx >= len(parent):
                break
            stack.append(parent[idx].lower())
            idx += 1
        elif op == 'REDUCE':
            if not ignore_invalid:
                assert(len(stack))
            stack = []
        elif op == 'COPY-REDUCE':
            if not ignore_invalid:
                assert(len(stack))
            output += stack
            stack = []
        elif op == 'COPY':
            if not ignore_invalid:
                assert(len(stack))
            output.append(stack[-1])
        else:
            output.append(op.lower())

    if not ignore_invalid:
        assert(idx == len(parent))
                 
    return output

def transitions2transducer(transitions, parent, verbose=False, ignore_invalid=False):
    output = []
    stack = []
    buffer = []
    
    idx = 0
    for op in transitions:
        if verbose:
            print(op, idx, stack, buffer, output)
        if op == 'SHIFT':
            if idx < len(parent):
                stack.append(parent[idx].lower())
                idx += 1
        elif op == 'REDUCE' or op == 'COPY-REDUCE':
            output += buffer 
            if op == 'COPY-REDUCE':
                output += stack
            output += ['<e>'] * len(stack)
            stack = []
            buffer = []
        else:
            buffer.append(op.lower())
    
    if len(buffer):
        output += buffer

    if len(stack):
        output += ['<e>'] * len(stack)
            
    return output


