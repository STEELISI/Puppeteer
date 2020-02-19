from os import listdir, walk
from os.path import isfile, join

def filenames_from_dirname(dirname):
    
    PATH = dirname
    
    filenames = []
    for path, dirs, files in walk(PATH):
        for filename in files:
            fullpath = join(path, filename)
            filenames.append(fullpath)
    return filenames

def dirnames_from_dirname(dirname):
    
    PATH = dirname
    dirnames = []
    for path, dirs, files in walk(PATH):
        for dir in dirs:    
            fullpath = join(path, dir)
            dirnames.append(fullpath)
    return dirnames