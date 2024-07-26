

def delete(d, del_key='ID'):
    for key in list(d.keys()):
        if del_key in key:
            del d[key]
        elif isinstance(d[key], dict):
            delete(d[key], del_key)

def delete_empty(d):
    for key in list(d.keys()):
        if not d[key]:
            del d[key]
        elif isinstance(d[key], dict):
            delete_empty(d[key])
