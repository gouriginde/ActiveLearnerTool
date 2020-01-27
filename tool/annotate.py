import pandas as pd
import numpy as np
import logs
from uncertaintySampling import leastConfidenceSampling,minMarginSampling,entropySampling

def analyzePredictions(args,df_predictions,targetLabel,confidence):
    '''
    Calculates the maximum prability value from the prediction probabilites of all the classes
    Labels the combinations as 'I' aka intelligently annotated for which the maximum probability value is greater than the confidence
    Labels doubtful combinations as 'A'
    Calls annotUncertainSamples function, to annotate most uncertain samples manually.
    Returns final dataframe containing all the combinations after marking them 'M','I','A' as the case may be.
    '''

    if targetLabel == "BinaryClass":
        labelColumn = "BLabelled"
    else:
        labelColumn = "MLabelled"

    logs.writeLog("\n\nAnalyzing Predictions... Based on the Maximum Probability Value for each combination.")
    
    if confidence == 0:
        #probabilityBins = df_predictions['maxProb'].value_counts()
        bins = np.arange(0,1.1,0.1)
        probabilityBins = pd.cut(df_predictions['maxProb'],bins=bins).value_counts().sort_index(ascending=False)
        while True:
            try:
                logs.writeLog("\n\nFollowing is the observed maximum probability distribution : \n"+str(probabilityBins))
                logs.writeLog("\n\nPlease select the threshold probability value to mark the predictions confident : ")
                confidence = float(input(""))
                logs.writeLog("User Input : "+str(confidence))
                if confidence>1.0:
                    logs.writeLog("\n\nInvalid Input. Please provide a valid value.\n")
                else:
                    break
            except ValueError:
                logs.writeLog("\n\nVALUE ERROR! ---- Invalid Input. Please provide a valid value.")
    logs.writeLog("\n\nLooking for Confident Predictions.... for which Confidence Value >= "+str(confidence))
    df_ConfidentPredictions = df_predictions[(df_predictions['maxProb']>=float(confidence))]  #Mark the values as confident if the maxProb is greater than confidence value.
    df_ConfidentPredictions[labelColumn]='I'  
    logs.writeLog("\n\nConfident Predictions : "+str(len(df_ConfidentPredictions))+" Rows\n"+str(df_ConfidentPredictions[:10]))
    
    df_ConfidentPredictions=df_ConfidentPredictions[['req1_id','req1','req2_id','req2','BinaryClass','MultiClass','BLabelled','MLabelled']]  #Remove the extra columns.
    
    #Predictions which are not part of the ConfidentPredictions
    logs.writeLog("\n\nSegregating the doubtful predictions...")
    df_doubtfulPredictions = df_predictions[~df_predictions.isin(df_ConfidentPredictions)].dropna()  #Make sure the BinaryClass/MultiClass of To Be labelled are not np.NaN; give it any dummy value
    df_doubtfulPredictions[labelColumn]='A'   #Mark the doubtful Predictions as 'A' - to be Annoatated....
    logs.writeLog ("\n\nDoubtful Predictions : "+str(len(df_doubtfulPredictions))+" Rows\n"+str(df_doubtfulPredictions[:10]))
    
    #Create an empty DataFrame
    df_AnnotatedData = pd.DataFrame()
    df_results = pd.DataFrame()
    
    if df_doubtfulPredictions.shape[0]>0:  #Manual Annotation is needed only if there is a doubtful Prediction.

        #Call annotUncertainSamples Function; Asks user to manually annotate annotCount number of samples where sample are selected with the queryType.
        df_AnnotatedData = annotUncertainSamples(args,df_doubtfulPredictions,targetLabel)       
        logs.writeLog("\n\nAnnotated Data : "+str(len(df_AnnotatedData))+" Rows \n"+str(df_AnnotatedData[:10]))

    
    #Merge the results.
    df_results=pd.concat([df_ConfidentPredictions,df_AnnotatedData],axis=0)
    #resultsDf.rename(columns={'predictedLabel':targetLabel},inplace=True)
    df_results.reset_index(inplace=True,drop=True)
        
    return df_results,confidence
    

def annotUncertainSamples(args,df_uncertainSamples,targetLabel):
    '''
    Based on the queryType, most uncertain samples are sampled and forwarded to manual annotater to get the annotations.
    '''
    df_manuallyAnnotated = pd.DataFrame(columns=df_uncertainSamples.columns)#Create an empty Dataframe to store the manually annotated Results

    queryType = args.loc[0,'samplingType']

    iteration = 0
    while iteration<int(args.loc[0,'manualAnnotationsCount']):  #while iteration is less than number of annotations that need to be done.
        if (len(df_uncertainSamples)>0):
            logs.writeLog("\n\n Iteration : "+str(iteration+1))
            if queryType == 'leastConfidence':
                indexValue = leastConfidenceSampling(df_uncertainSamples)
            elif queryType == 'minMargin':
                indexValue = minMarginSampling(df_uncertainSamples)
            elif queryType == 'entropy':
                indexValue =entropySampling(df_uncertainSamples)
        
            sample = df_uncertainSamples.loc[indexValue,:]
            logs.writeLog("\n\nMost Uncertain Sample : \n"+str(sample))
            
            df_userAnnot = pd.DataFrame(columns = ['req1_id','req2_id','req1','req2','BinaryClass','MultiClass','BLabelled','MLabelled'])
            userAnnot = getManualAnnotation(sample['req1_id'],sample['req2_id'],sample['req1'],sample['req2'],targetLabel)   #Passes the requirements to the user and requests annotation.
            
            if userAnnot == "exit":
                #Dump df_trainingSet into Annotations.csv (These are the manual annotations done before active learning actually starts)
                raise Exception ('\nExited the Program successfully!')
            
            #Remove the selected sample from the original dataframe
            df_uncertainSamples.drop(index=indexValue,inplace=True)   
            df_uncertainSamples.reset_index(inplace=True,drop=True)
            #logs.writeLog(str(df))

            if targetLabel == "BinaryClass":
                #Add the newly annotated combination in the manuallyAnnotatedDf
                df_userAnnot = df_userAnnot.append({'req1_id':sample['req1_id'],'req1':sample['req1'],'req2_id':sample['req2_id'],'req2':sample['req2'],'BinaryClass':userAnnot,'MultiClass':0,'BLabelled':'M','MLabelled':'A'},ignore_index=True)  #Added MultiClass as 0 because when we are learning BinaryClass... MultiClass can contain a dummy value.
                logs.createAnnotationsFile(df_userAnnot)
            
                df_manuallyAnnotated = pd.concat([df_manuallyAnnotated,df_userAnnot])
                #logs.writeLog("Manually Annotated DataFrame : \n"+str(manuallyAnnotatedDf))
            else:
                #Add the newly annotated combination in the manuallyAnnotatedDf
                df_userAnnot = df_userAnnot.append({'req1_id':sample['req1_id'],'req1':sample['req1'],'req2_id':sample['req2_id'],'req2':sample['req2'],'BinaryClass':1,'MultiClass':userAnnot,'BLabelled':sample['BLabelled'],'MLabelled':'M'},ignore_index=True)  #Added MultiClass as 0 because when we are learning BinaryClass... MultiClass can contain a dummy value.
                logs.createAnnotationsFile(df_userAnnot)
            
                df_manuallyAnnotated = pd.concat([df_manuallyAnnotated,df_userAnnot])
                #logs.writeLog("Manually Annotated DataFrame : \n"+str(manuallyAnnotatedDf))

        iteration+=1
    
    #Remove all the extra columns. df now contains only combinations marked 'A'
    df_uncertainSamples=df_uncertainSamples[['req1_id','req1','req2_id','req2','BinaryClass','MultiClass','BLabelled','MLabelled']]
    #logs.writeLog(str(df_uncertainSamples))

    #Remove all the extra columns. df now contains only combinations marked 'M'
    df_manuallyAnnotated=df_manuallyAnnotated[['req1_id','req1','req2_id','req2','BinaryClass','MultiClass','BLabelled','MLabelled']]
    logs.writeLog("\n\nManually Annotated Combinations... "+str(len(df_manuallyAnnotated))+"Rows \n"+str(df_manuallyAnnotated[:10]))
    
    return pd.concat([df_manuallyAnnotated,df_uncertainSamples],axis=0)


def getManualAnnotation(req1_id,req2_id,req1,req2,target):
    '''
    The user get's the two requirements and is expected to provide the annotation for the combination. 
    The target can be BinaryClass or MultiClass based on the arguments provided by the user.
    Returns the annotation.
    '''
    if target =='BinaryClass':
        while True:  #While loop to make sure the user provides proper input. 
    
            logs.writeLog ("\n\nAre the following Requirements Dependent or Not?")
            logs.writeLog ("\n\nRequirement 1 ("+req1_id+") : "+str(req1))
            logs.writeLog ("\nRequirement 2 ("+req2_id+") : "+str(req2))
            logs.writeLog ("\nPlease enter 1 for Dependent, 0 for Independent   :   ")
            userAnnotation = input("")
            if userAnnotation in ['1','0']:
                logs.writeLog ("\nValue provided by the user :- "+str(userAnnotation.lower()))
                return userAnnotation
            elif userAnnotation.lower() == "exit":
                logs.writeLog ("\nValue provided by the user :- "+str(userAnnotation.lower()))
                return "exit"
                #raise Exception ('\nExited the Program successfully!')
            else:
                logs.writeLog ("\nUser Annotation : "+userAnnotation.lower())
                logs.writeLog ("Invalid Input. Allowed Values -- 1 / 0")
                logs.writeLog ("In order to exit from the program. Enter 'exit'")
                continue             
    else:
        while True:  #While loop to make sure the user provides proper input. 
    
            logs.writeLog ("\n\nPlease provide the dependency type for the following requirements.")
            logs.writeLog ("\n\nRequirement 1 ("+req1_id+") : "+str(req1))
            logs.writeLog ("\nRequirement 2 ("+req2_id+") : "+str(req2))
            #logs.writeLog ("\nPlease select one of the following choices. \n1 - AND\n2 - OR \n3 - Requires \n4 - Similar \n5 - Cannot Say \nEnter your Choice here :   ")   #Removed 0 - Independent
            #logs.writeLog ("\nPlease select one of the following choices. \n3 - Requires \n4 - Similar \n6 - Others \nEnter your Choice here :   ")   #Removed 0 - Independent
            logs.writeLog ("\nPlease select one of the following choices. \n1 - Requires \n2 - Reflects \n3 - Conflicts \nEnter your Choice here :   ")   
            
            userAnnotation = input("")
            #if userAnnotation in ['1','2','3','4','5']:
            #if userAnnotation in ['3','4','6']:
            if userAnnotation in ['1','2','3']:             
                logs.writeLog ("\nValue provided by the user :- "+str(userAnnotation.lower()))
                return userAnnotation
            elif userAnnotation.lower() == "exit":
                logs.writeLog ("\nValue provided by the user :- "+str(userAnnotation.lower()))
                return "exit"
                #raise Exception ('\nExited the Program successfully!')
            else:
                logs.writeLog ("\nValue provided by the user :- "+str(userAnnotation.lower()))
                #logs.writeLog ("Invalid Input. Allowed Values -- 1 / 2 / 3 / 4 / 5 ")
                #logs.writeLog ("Invalid Input. Allowed Values -- 3 / 4 / 6 ")
                logs.writeLog ("Invalid Input. Allowed Values -- 1 / 2 / 3 ")
                logs.writeLog ("In order to exit from the program. Enter 'exit'")
                continue
    return None    
