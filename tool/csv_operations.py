import random
import numpy as np
from contentManagement import lookupValue

def get_random_req_combination(df):
    '''
    Takes dataframe as input.
    selects a random row from the dataframe
    returns the contents of the row as dictionary, which are displayed in Flask.
    '''    
    
    #Generate Random index value
    random_index = random.randrange(df.shape[0])
    # rqmt_data.shape[0] returns the numbe of rows in the dataframe
    # randrange selects a random number in range zero and number of rows in the data frame
    
    rqmts_dict = {'combo_id': random_index,
                  'rqmt1_id': df.iloc[random_index]['req1_id'],
                  'rqmt1_dt': df.iloc[random_index]['req1'],
                  'rqmt2_id': df.iloc[random_index]['req2_id'],
                  'rqmt2_dt': df.iloc[random_index]['req2'],
                  'similarity': df.iloc[random_index]['similarity'],
                  'cosine': df.iloc[random_index]['cosine'],
                  }
    rqmts_dict['bClass'] = lookupValue('dependencyFlag',df.iloc[random_index]['BinaryClass'])
    rqmts_dict['mClass'] = lookupValue('dependencyType',df.iloc[random_index]['MultiClass'])   
    return rqmts_dict

def verifyChanges(df,userDict,fPath):
    '''
    Takes original dataframe (df), data entered by user in form (userDict),
    filePath to save changes back to csv (fPath).
    '''
    index = int(userDict['combo_id'])   #index of the combination which was visible in GUI.
    originalDict = df.iloc[index]   #returns series of the data from the dataframe for respective index value.
    fileUpdate = False    #Flag to update csv file.
    for key in list(userDict.keys())[1:]:   #ignored the first key combo_id, as it is not part of the series
        if userDict[key] != originalDict[key]:  
            fileUpdate = True
            print ("Value in "+key+" is different, Update DF and csv file.")
            updateValue(df = df,index = index,column = key, value = userDict[key])
    if fileUpdate:
        df.to_csv(fPath)
        print ("CSV File has been updated...")
    else:
        print ("No updates done...")

def updateValue(df,index,column,value):
    '''
    Updates the value at a particular index and column location in the dataframe.
    '''
    print ("Before...")
    print (df.iloc[index])
    df.at[index,column]=value
    print ()
    print ("Updated")
    print (df.iloc[index])
    
