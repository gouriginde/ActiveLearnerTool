def content():
    contentDict = {'projectName':[['Project 1','SansDuplicates_Clones.csv'],
                                  ['Project 2','abc.csv'],
                                  ['Project 3','def.csv'],
                                  ['Project 4','ghi.csv'],
                                  ['Project 5','jkl.csv']],
                    'dependencyType':[['Requires',0],
                                      ['Similar',1],
                                      ['AND',2],
                                      ['OR',3],
                                      ['XOR',4],
                                      ['NaN','NaN']],
                    'dependencyFlag':[['Yes',1],
                                      ['No',0]]}
    return contentDict



def getServerDetails(input_file):
    settings_dict = {}
    with open(input_file,mode ='r', errors='ignore') as f:
        content = f.readlines()
        for line in content:
            params = line.split(":")
            if params[0].strip() not in settings_dict:
                settings_dict[params[0].strip()]=params[1].strip()
    return settings_dict

def lookupValue(Key,Target):
    '''
    Looks up for the value in contentDict in content() function, returns the Corresponding value for the Target
    eg. lookup('dependencyType',1) will return 'Similar'
        lookup('dependencyType','Similar') will return 1

    '''
    content_dict = content()
    for pair in content_dict[Key]:
        if Target == pair[0]:
            return pair[1]
        elif Target == pair[1]:
            return pair[0]
        else:
            continue
    return None