import datetime
import os
import pandas as pd

def createLogs(fPath,args,comments):
    '''
    Create files to push logs and outputs for tracking purpose.
    '''
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    if not os.path.exists(fPath+"/"+current_date):
        os.makedirs(fPath+"/"+current_date)
    global logFilePath,outputFilePath#,resultsPath,annotationsPath
    logFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifiersList']+"-"+comments+".txt"
    outputFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifiersList']+"-"+comments+".csv"
    #resultsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifiersList']+"-RESULTS-"+comments+".csv"
    #annotationsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifiersList']+"-ANNOTATIONS-"+comments+".csv"
    for fPath in [logFilePath,outputFilePath]:
        file = open(fPath,'a')
        file.write("\n"+100*"-"+"\nArguments :- \n")
        for col in args.columns:
            file.write(str(col)+" : "+str(args.loc[0,str(col)])+"\n")
        #for col in args.index.tolist():
        #    file.write(str(col)+" : "+str(args.loc[0,str(col)])+"\n")
        file.write(100*"-")
        file.close()

    return logFilePath,outputFilePath
    
def getArguments(fName):
    '''
    Reads the arguments available in the file and converts them into a data frame.
    '''
    file = open(fName,'r')
    df_args = pd.DataFrame()
    print ("\n"+100*"-"+"\nArguments :- \n")
    for line in file:
        print (line.strip())
        kv_pair = line.split(":")
        df_args.loc[0,str(kv_pair[0]).strip()] = str(kv_pair[1]).strip()
    print (100*"-")
    validateArguments(df_args)
    return df_args


def validateArguments(df_args):
    '''
    Validates the arguments. Raises Error incase any of the arguments are invalid and stops execution.
    '''
    try:
        if not os.path.exists(os.getcwd()+"/static/data/"+df_args.loc[0,'labelledData']) :
            raise OSError("/static/data/"+df_args.loc[0,'labelledData'] + " File Doesn't Exist")

        if not os.path.exists(os.getcwd()+"/static/data/"+df_args.loc[0,'validationData']) :
            raise OSError("/static/data/"+df_args.loc[0,'validationData'] + " File Doesn't Exist") 

        if df_args.loc[0,'tobepredicted'] not in ['b','m']:
            raise ValueError("Invalid value in tobepredicted. Allowed values b/m")

        if df_args.loc[0,'classifiersList']:  #if some value exists
            clfList = df_args.loc[0,'classifiersList'].split(";")
            for clf in clfList:
                if clf not in ['RF','SVM','NB']:
                    raise ValueError("Invalid classifier list. Allowed values RF,SVM,NB")

        if (float(df_args.loc[0,'testsize']) not in [x/10 for x in range(0,11)]):
            raise ValueError("Invalid value in testsize. Allowed values 0,0.1,0.2....1.0")

    except :
        print ("\nERROR! Input Arguments are invalid....\n\nRefer following details for allowed values....\n")
        showExpectedArguments()
        raise
        exit()
           
    return None

def showExpectedArguments():
    '''
    prints the expected arguments 
    '''
    file = open(os.getcwd()+"/clfParams_Desc.txt")
    for line in file:
        print (line)
    print ()


def writeLog(content):
    file = open(logFilePath,"a")
    file.write(content)
    file.close()
    print (content)
    return None

def addOutputToExcel(df,comment):
    file = open(outputFilePath,"a")
    file.write("\n\n"+comment+"\n\n")
    file.close()
    print (comment)
    print (str(df))
    df.to_csv(outputFilePath,mode='a',index=False)
    return None