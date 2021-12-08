"""
Various utility functions used for processing and updating JSON definition of networks.
"""

import json
import collections.abc
import os


def LoadJSONFile(fileName):
    #Checking if JSON file is fully defined path or just a file name without path.
    #If just a name, then it will search in default directory.
    if "/" in fileName:
        if ".json" in fileName:
            pass
        else:
            fileName = fileName + ".json"
    else:
        for (dirpath, dirnames, filenames) in os.walk("configs/network"):
            for filename in filenames:
                if fileName in filename:
                    fileName = os.path.join(dirpath,filename)
                    break
        # raise
    with open(fileName) as json_file:
        data = json.load(json_file)
    return data


def UpdateStringValues(d, u):
    """
    Updates values of a nested dictionary/list structure. The method searches
    for a keys provided by the override dictionary and replaces them with
    associated values.
    """
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = UpdateStringValues(d.get(k, {}), u)
        elif isinstance(v,list):
            list_new = []
            for val in v:
                if isinstance(val, collections.abc.Mapping):
                    tmp_dict = UpdateStringValues(val, u)
                    list_new.append(tmp_dict)
                elif isinstance(val, list):
                    pass
                elif val in u.keys():
                    list_new.append(u[val])
                else:
                    list_new.append(val)
            d[k] = list_new
        else:
            for key in u.keys():
                if isinstance(v, str):
                    if key in v:
                        d[k] = EvalString(v,u)
                        break
    return d


def EvalString(string,updateDict):
    for key,value in updateDict.items():
        string = string.replace(key,str(value))
    return eval(string)


def ConvertNetwork(json):
    pass


if __name__ == "__main__":
    test = {"K1":1,
            "K2":2,
            "K3":"V1",
            "K4":2,
            "K5":{"test":["V1","V2"]},
            "K6":"V2",
            "K7":2,
            }
    test2 = {"V1":4,"V2":5}
    dict=UpdateStringValues(test,test2)
    # ReplaceValues(test,test2)
    print(dict)
