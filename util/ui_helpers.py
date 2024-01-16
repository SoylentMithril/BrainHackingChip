
def traverse_to(from_obj, path, to_val=True):
    if len(path) == 1 and path[0] in from_obj:
        return from_obj[path[0]]['attributes']['value'] if to_val == True else from_obj[path[0]]
    
    newpath = path if path[0] in from_obj else ['sub_attr', *path]
    return traverse_to(from_obj[newpath[0]], newpath[1:], to_val)