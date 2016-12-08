# MODIFIED VERSION FROM https://github.com/vladikk/JSON2CSV/blob/master/json2csv.py
from itertools import chain
import json

def json_to_dicts(json_str):
    objects = json.loads(json_str)
    
    def to_single_dict(lst):
        result = {}
        for d in lst:
            for k in d.keys():
                result[k] = d[k]
        return result;

    to_keyvalue_pairs(objects[0])
    
    return [dict(to_keyvalue_pairs(obj)) for obj in objects]
        
def to_keyvalue_pairs(source, ancestors=[], key_delimeter='_'):
    def is_sequence(arg):
        return (not hasattr(arg, "strip") and hasattr(arg, "__getitem__"))

    def is_dict(arg):
        return hasattr(arg, "keys")

    if is_dict(source):
        result = [to_keyvalue_pairs(source[key], ancestors + [key]) for key in source.keys()]
        return list(chain.from_iterable(result))
    elif is_sequence(source):
        result = [to_keyvalue_pairs(item, ancestors + [str(index)]) for (index, item) in enumerate(source)]
        return list(chain.from_iterable(result))
    else:
        return [(key_delimeter.join(ancestors), source)]   

def dicts_to_csv(source):
    def build_row(dict_obj, keys):
        return [dict_obj.get(k, "") for k in keys]
    
    keys = sorted(set(chain.from_iterable([o.keys() for o in source])))
    rows = [build_row(d, keys) for d in source]
    
    return keys, rows

def write_csv(file, keys=None, rows=[]):
    if keys :
        file.writerow(keys) 
    
    for row in rows:
        file.writerow([str(c)  for c in row])  
