import os
import numpy as np
import pandas as pd
import warnings
import argparse
from sklearn.utils import shuffle
import logs 
from clf_model import createInitialTrainingSet,createClassifier,predictLabels,validateClassifier
from annotate import analyzePredictions 
from activeLearner import get_args,getData
import uncertaintySampling

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)   #To make sure all the columns are visible in the logs.
pd.set_option('display.width', 1000)

def learnTargetLabel(args,df_rqmts,targetLabel):
    #Based on scenario, user can provide the initial manual annotations in the input file and mark them as 'M' in Labelled Column
    #If the user decides to provide the manual annotations on the go. Then inorder to provide the initial Manual Annotations which will form the training set,
    #User can provide the trainingCount number of annotations and the labelled data will be marked as 'M' in Labelled column.
    
    
    if targetLabel == "BinaryClass":
        labelColumn = "BLabelled"
    else:
        labelColumn = "MLabelled"
    
    #LabelledCombinations (Manually Annotated Set)
    df_manuallyAnnotatedSet = shuffle(df_rqmts[df_rqmts[labelColumn]=='M'])

    #Dump the labelled/Manually annotated data combinations in results.csv file
    #logs.createAnnotationsFile(df_manuallyAnnotatedSet)

    if args.loc[0,'initManualAnnotAvail'].lower() == 'n':     #Create Initial Training Set only if the ManualAnnotationAvailabilityFlag is 'n'
        df_rqmts = createInitialTrainingSet(df_rqmts,int(args.loc[0,'trainingCount']),targetLabel)   #User provides manual annotations for trainingCount(default=10) requirement combinations.
        logs.writeLog("Combined Data Set after doing the initial Manual Annotations : \n"+str(df_rqmts))
    
    
    ###############################################################################################################################
    
    df_manuallyAnnotatedSet = shuffle(df_rqmts[df_rqmts[labelColumn]=='M'])
    #Splitting Initially Manually Annotated Data (300 Data points into Train and Validation Test Sets......)  
    validationSetCount = 75
    df_validationSet = df_manuallyAnnotatedSet[:validationSetCount]
    logs.writeLog("\nSeparating Validation Set : "+str(len(df_validationSet))+" Rows\n"+str(df_validationSet[:10]))
    
    df_validationSet.reset_index(drop=True,inplace=True)
    df_rqmts = df_rqmts[~df_rqmts.isin(df_validationSet)].dropna()   #Removed the 75 selected combinations which formed the Validation Set.
    
    ###############################################################################################################################
        
    #Initial Analysis of the data available. 
    iteration = 0 

    manuallyAnnotatedCount = len(df_rqmts[df_rqmts[labelColumn]=='M'])
    intelligentlyAnnotatedCount = len(df_rqmts[df_rqmts[labelColumn]=='I'])
    toBeAnnotatedCount = len(df_rqmts[df_rqmts[labelColumn]=='A'])
    
    if targetLabel=='BinaryClass':
        #Create a dataframe to store the results after each iteration of active Learning.
        df_resultTracker = pd.DataFrame(columns=['Iteration','ManuallyAnnotated','IntelligentlyAnnotated','ToBeAnnotated','TrainingSize','TestSize','ValidationSize','ClassifierTestScore','ClassifierValidationScore','DependentCount','IndependentCount','f1Score','precisionScore','recallScore'])
        
        #Number of combinations which have been manually or intelligently labelled as dependent or independent    
        dependentCount = len(df_rqmts[(df_rqmts['BinaryClass'].astype('int')==1) & (df_rqmts[labelColumn].isin(['M','I']))])
        independentCount = len(df_rqmts[(df_rqmts['BinaryClass'].astype('int')==0) & (df_rqmts[labelColumn].isin(['M','I']))])

        #Add the initial analysis to the analysis dataFrame created.
        df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':'-','TestSize':'-','ValidationSize':validationSetCount,'ClassifierTestScore':'-','ClassifierValidationScore':'-','DependentCount':dependentCount,'IndependentCount':independentCount,'f1Score':'-','precisionScore':'-','recallScore':'-'},ignore_index=True)

    else:
        #Create a dataframe to store the results after each iteration of active Learning...  #Added ORCount,ANDCount etc columns to keep a track of different dependency types
        #df_resultTracker = pd.DataFrame(columns=['Iteration','ManuallyAnnotated','IntelligentlyAnnotated','ToBeAnnotated','TrainingSize','TestSize','ValidationSize','ClassifierTestScore','ClassifierValidationScore','AndCount','ORCount','RequiresCount','SimilarCount','CannotSayCount','f1Score','precisionScore','recallScore'])
        df_resultTracker = pd.DataFrame(columns=['Iteration','ManuallyAnnotated','IntelligentlyAnnotated','ToBeAnnotated','TrainingSize','TestSize','ValidationSize','ClassifierTestScore','ClassifierValidationScore','RequiresCount','SimilarCount','OtherCount','f1Score','precisionScore','recallScore'])
        
        #Number of combinations which have been manually or intelligently labelled for different dependency types   
        #andCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==1) & (df_rqmts[labelColumn].isin(['M','I']))])
        #orCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==2) & (df_rqmts[labelColumn].isin(['M','I']))])
        requiresCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==3) & (df_rqmts[labelColumn].isin(['M','I']))])
        similarCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==4) & (df_rqmts[labelColumn].isin(['M','I']))])
        #cannotSayCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==5) & (df_rqmts[labelColumn].isin(['M','I']))])
        otherCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==6) & (df_rqmts[labelColumn].isin(['M','I']))])

        #Add the initial analysis to the analysis dataFrame created.
        #df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':'-','TestSize':'-','ValidationSize':validationSetCount,'ClassifierTestScore':'-','ClassifierValidationScore':'-','AndCount':andCount,'ORCount':orCount,'RequiresCount':requiresCount,'SimilarCount':similarCount,'CannotSayCount':cannotSayCount,'f1Score':'-','precisionScore':'-','recallScore':'-'},ignore_index=True)
        df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':'-','TestSize':'-','ValidationSize':validationSetCount,'ClassifierTestScore':'-','ClassifierValidationScore':'-','RequiresCount':requiresCount,'SimilarCount':similarCount,'OtherCount':otherCount,'f1Score':'-','precisionScore':'-','recallScore':'-'},ignore_index=True)
    
    logs.writeLog("\n\nInitial Data Analysis : \n"+str(df_resultTracker)+"\n")
    
    confidence = 0  #initial value of confidence; user will be providing value after looking for the probability distribution.
    
    while True:

        iteration+=1

        logs.writeLog("\n\nIteration : "+str(iteration)+"\n")
        logs.writeLog("\nSplitting Labelled Data and Unlabelled Data\n")
        
        df_labelledData = df_rqmts[df_rqmts[labelColumn].isin(['M','I'])] #Training Data
        logs.writeLog('\nLabelled Data : '+str(len(df_labelledData))+' Rows \n'+str(df_labelledData[:10])+"\n")
        
        df_unlabelledData = df_rqmts[df_rqmts[labelColumn]=='A']  #Test Data
        logs.writeLog('\nUnlabelled Data : '+str(len(df_unlabelledData))+' Rows \n'+str(df_unlabelledData[:10])+"\n")
        
        if len(df_labelledData)==0:
            err = "There are no Labelled Data Points to Training the Classifier. Either Manually Annotate them in input file or Mark initManualAnnotAvail flag 'y' in arguments."
            logs.writeLog("Error! : "+str(err))
            raise (err)
        
        if len(df_unlabelledData)==0:   #If there are no more ToBeAnnotated Combinations then Exit..
            logs.writeLog("There are no more Unlabelled Data Points....")
            df_rqmts = pd.concat([df_rqmts,df_validationSet],axis=0,ignore_index=True)
            return df_rqmts,df_resultTracker,confidence  
        
        if iteration >= 11: #After 10 iterations, ask user if he/she wants to continue active learner....
            while True:
                logs.writeLog("Exceeded the iteration limit. Still there are "+str(len(df_unlabelledData))+" combinations to be Labelled. Do you wish to continue Annotating? Enter 'y'/'n'")
                userInput = input()
                logs.writeLog("User's input : "+str(userInput))
                if userInput.lower()=='n':
                    logs.writeLog("Stopping Condition Reached...")
                    df_rqmts = pd.concat([df_rqmts,df_validationSet],axis=0,ignore_index=True)
                    return df_rqmts,df_resultTracker,confidence 
                elif userInput.lower()=='y':
                    logs.writeLog("Continuing with Iteration "+str(iteration))
                    break
                else:
                    logs.writeLog ("Invalid Input. Allowed Values -- y / n")
                    continue

        logs.writeLog("\n"+100*"-")
        logs.writeLog ("\nCreating Classifier....")
        countVectorizer, tfidfTransformer, classifier, classifierTestScore,trainSize,testSize,f1Score,precisionScore,recallScore = createClassifier(args.loc[0,'classifier'],float(args.loc[0,'testsize']),df_labelledData,targetLabel)  
        
        ############################################################################################################################
        logs.writeLog ("\n\nValidating Classifier...")
        classifierValidationScore = validateClassifier(countVectorizer,tfidfTransformer,classifier,df_validationSet,targetLabel)
        logs.writeLog("\n\nClassifier Validation Set Score : "+str(classifierValidationScore))
        ############################################################################################################################
        logs.writeLog("\n"+100*"-")

        input("\n\nHit Enter to Proceed....")
    
        logs.writeLog ("\n\nPredicting Labels....")
        df_predictionResults = predictLabels(countVectorizer,tfidfTransformer,classifier,df_unlabelledData,targetLabel)  
        logs.writeLog('\nPrediction Results :  '+str(len(df_predictionResults))+" Rows \n"+str(df_predictionResults[:10]))
        
        df_finalPredictions,confidence = analyzePredictions(args,df_predictionResults,targetLabel,confidence)
        logs.writeLog("\n\nFinal Predictions : "+str(len(df_finalPredictions))+" Rows \n"+str(df_finalPredictions[:10]))
        
        df_updatedDatabase = pd.concat([df_labelledData,df_finalPredictions],axis=0,ignore_index=True)
        logs.writeLog("\n\nUpdated Database : "+str(len(df_updatedDatabase))+" Rows \n"+str(df_updatedDatabase[:10]))
        df_rqmts = df_updatedDatabase

        #Update the Results and add them to Result Tracker
        
        if targetLabel == 'BinaryClass':
            manuallyAnnotatedCount = len(df_rqmts[df_rqmts['BLabelled']=='M'])
            intelligentlyAnnotatedCount = len(df_rqmts[df_rqmts['BLabelled']=='I'])
            toBeAnnotatedCount = len(df_rqmts[df_rqmts['BLabelled']=='A'])
            
            dependentCount = len(df_rqmts[(df_rqmts['BinaryClass'].astype('int')==1) & (df_rqmts['BLabelled'].isin(['M','I']))])
            independentCount = len(df_rqmts[(df_rqmts['BinaryClass'].astype('int')==0) & (df_rqmts['BLabelled'].isin(['M','I']))])
            
            df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':trainSize,'TestSize':testSize,'ValidationSize':validationSetCount,'ClassifierTestScore':classifierTestScore,'ClassifierValidationScore':classifierValidationScore,'DependentCount':dependentCount,'IndependentCount':independentCount,'f1Score':f1Score,'precisionScore':precisionScore,'recallScore':recallScore},ignore_index=True)
            logs.writeLog("\n\nAnalysis DataFrame : \n"+str(df_resultTracker))  
        
        else:
            manuallyAnnotatedCount = len(df_rqmts[df_rqmts['MLabelled']=='M'])
            intelligentlyAnnotatedCount = len(df_rqmts[df_rqmts['MLabelled']=='I'])
            toBeAnnotatedCount = len(df_rqmts[df_rqmts['MLabelled']=='A'])
        
            #andCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==1) & (df_rqmts['MLabelled'].isin(['M','I']))])
            #orCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==2) & (df_rqmts['MLabelled'].isin(['M','I']))])
            requiresCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==3) & (df_rqmts['MLabelled'].isin(['M','I']))])
            similarCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==4) & (df_rqmts['MLabelled'].isin(['M','I']))])
            #cannotSayCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==5) & (df_rqmts['MLabelled'].isin(['M','I']))])
            otherCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==6) & (df_rqmts['MLabelled'].isin(['M','I']))])

            #df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':trainSize,'TestSize':testSize,'ValidationSize':validationSetCount,'ClassifierTestScore':classifierTestScore,'ClassifierValidationScore':classifierValidationScore,'AndCount':andCount,'ORCount':orCount,'RequiresCount':requiresCount,'SimilarCount':similarCount,'CannotSayCount':cannotSayCount,'f1Score':f1Score,'precisionScore':precisionScore,'recallScore':recallScore},ignore_index=True)
            df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':trainSize,'TestSize':testSize,'ValidationSize':validationSetCount,'ClassifierTestScore':classifierTestScore,'ClassifierValidationScore':classifierValidationScore,'RequiresCount':requiresCount,'SimilarCount':similarCount,'OtherCount':otherCount,'f1Score':f1Score,'precisionScore':precisionScore,'recallScore':recallScore},ignore_index=True)
            
            logs.writeLog("\n\nAnalysis DataFrame : \n"+str(df_resultTracker))
    
    #Merge Validation Set back to the prediction set to ensure all the 19699 combinations are returned.
    df_rqmts = pd.concat([df_rqmts,df_validationSet],axis=0,ignore_index=True)
    return df_rqmts,df_resultTracker,confidence
    
def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)  #Ignore Future warnings if any occur. 
    
    #initialize directory which contains all the data and which will contain logs and outputs
    currentFileDir = os.getcwd()
    args = logs.getArguments(currentFileDir+"/ALParams.txt") 
    
    #args=get_args()  #Get all the command line arguments
    #options = vars(args)  #Stores the arguments as dictionary ; used in logs
    ifileName = args.loc[0,'input']     
    #clf = args.classifier
    comments = args.loc[0,'comments']
    dependencyTypeNeeded = args.loc[0,'dependencyTypeNeeded']
    
    logFilePath,OFilePath = logs.createLogs(currentFileDir+"/static/data/Logs",args,comments)   #Creates the log file, default value is os.getcwd()+"/static/data/logs/" ; user still can provide his own logPath if needed.
    
    df_rqmtData = getData(currentFileDir+"/static/data/"+ifileName)   
    logs.writeLog("\n\nData Fetched from the input file : "+str(len(df_rqmtData))+" Rows \n"+str(df_rqmtData[:10]))
    
    logs.writeLog("\n\nStep 1 :- Learning BinaryClass Label\n")
    df_rqmtComb,df_BinaryAnalysis,thresholdConf = learnTargetLabel(args,df_rqmtData,'BinaryClass')
    
    logs.addOutputToExcel(df_BinaryAnalysis,"\nAnalysis of BinaryClass Label Classification (Threshold Prob "+ str(thresholdConf) +") : \n")
    input("Hit Enter to Proceed....")
    
    if dependencyTypeNeeded == 'y':
        logs.writeLog("\n\nStep 2 :- Learning MultiClass Label\n")
        df_rqmtCombDependent = df_rqmtComb[df_rqmtComb['BinaryClass']==1]   #Filtering only the Dependent Requirement Combinations for low level classification
        df_rqmtCombInDependent = df_rqmtComb[df_rqmtComb['BinaryClass']==0]

        df_rqmtCombDependent['MLabelled'] = df_rqmtCombDependent['MLabelled'].replace(' ','A')   #Mark the intelligently annotated combinations as unlabelled. (But what about the manual annotations done?)
        logs.writeLog("Following is the data set to be used for MultiClass classification : "+str(len(df_rqmtCombDependent))+" Rows\n"+str(df_rqmtCombDependent[:10]))
        
        
        if len(df_rqmtCombDependent[df_rqmtCombDependent['MLabelled']=='A'])>0:
            df_rqmtCombUpdated,df_MultiAnalysis,thresholdConf = learnTargetLabel(args,df_rqmtCombDependent,'MultiClass')
            logs.addOutputToExcel(df_MultiAnalysis,"Analysis of MultiClass Label Classification (Threshold Prob "+ str(thresholdConf) +") : \n")
            df_rqmtComb = pd.concat([df_rqmtCombUpdated,df_rqmtCombInDependent],axis=0)
            #logs.updateResults(df_rqmtResults,args)   #Update Results in excel....
        else:
            logs.writeLog("\n\nThere are no dependent combinations. So its not possible to find the Dependency Types.")
        
        
    logs.updateResults(df_rqmtComb,args)   #Update Results in excel....
    
    logs.writeLog("\nOutput Analysis is available at : "+str(OFilePath))
    logs.writeLog("\nLogs are available at : "+str(logFilePath))
    

if __name__ == '__main__':
    main()