def dictset(dictionary, keypath, value):
    for key_indx in range(len(keypath)):
        here = dictionary
        for tmp_indx in range(0, key_indx):
            here = here[keypath[tmp_indx]]
        
        if keypath[key_indx] not in here:
            here[keypath[key_indx]] = {}
            
        if key_indx == len(keypath) - 1:
            here[keypath[key_indx]] = value

def checkkeypath(dictionary, keypath):
    here = dictionary
    for key in keypath:
        if key in here:
            here = here[key]
        else:
            return None
    return here
