import datetime
import os
import pandas as pd

def createLogs(fPath,args,comments):
    #archiveFiles()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    #print (current_time)
    if not os.path.exists(fPath+"/"+current_date):
        os.makedirs(fPath+"/"+current_date)
    global logFilePath,outputFilePath,resultsPath,annotationsPath
    logFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-"+comments+".txt"
    #outputFilePath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-"+comments+".csv"
    #resultsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-RESULTS-"+comments+".csv"
    #annotationsPath = fPath+"/"+current_date+"/"+current_time+"-"+args.loc[0,'classifier']+"-ANNOTATIONS-"+comments+".csv"
    #for fPath in [logFilePath,outputFilePath]:
    for fPath in [logFilePath]:
        file = open(fPath,'a')
        file.write("\n"+100*"-"+"\nArguments :- \n")
        for col in args.columns:
            file.write(str(col)+" : "+str(args.loc[0,str(col)])+"\n")
        #for col in args.index.tolist():
        #    file.write(str(col)+" : "+str(args.loc[0,str(col)])+"\n")
        file.write(100*"-")
        file.close()

    #return logFilePath,outputFilePath #OutputFilePath not needed yet...
    return logFilePath

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
    Validates the arguments.
    '''
    try:
        #print ("Validating Arguments....")
        if not os.path.exists(os.getcwd()+"/static/data/"+df_args.loc[0,'labelledData']):  
            raise("")

        elif not os.path.exists(os.getcwd()+"/static/data/"+df_args.loc[0,'validationData']):  
            raise("")

        elif ((df_args.loc[0,'tobepredicted'] not in ['b','m']) or (df_args.loc[0,'classifier'] not in ['RF','NB','SVM'])): #tenfoldvalidation : n
            raise ("")

        elif (float(df_args.loc[0,'testsize']) not in [x/10 for x in range(0,11)]):
            raise ("")
    except :
        print ("\nERROR! Input Arguments are invalid....\nPlease verify your values with following reference.\n\n")
        showExpectedArguments()
        exit()
    return None

def showExpectedArguments():
    '''
    prints the expected arguments 
    '''
    file = open(os.getcwd()+"/clfParams_Desc.txt")
    for line in file:
        print (line)


def writeLog(content):
    file = open(logFilePath,"a")
    file.write(content)
    file.close()
    print (content)
    return None

def addOutputToExcel(df,comment):
    file = open(outputFilePath,"a")
    file.write(comment)
    file.close()
    print (comment)
    print (str(df))
    df.to_csv(outputFilePath,mode='a',index=False)
    return None

def updateResults(df_results,args):
    '''
    Merges the Results data frame with arguments dataframe and stores the results in a csv file. 
    '''
    df_results.reset_index(inplace=True,drop=True)
    args.reset_index(inplace=True,drop=True)
    combined_df = pd.concat([df_results,args],axis=1)
    combined_df.to_csv(resultsPath,mode="a",index=False)
    
    return resultsPath




def createAnnotationsFile(df_rqmts):
    '''
    Dumps the manuall Annotations data into a csv file.
    '''
    df_rqmts.to_csv(annotationsPath,mode="a",index=False,header=False)
    
    return resultsPath