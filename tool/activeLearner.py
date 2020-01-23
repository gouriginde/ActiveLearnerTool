import os
import numpy as np
import pandas as pd
import warnings
import argparse
from sklearn.utils import shuffle
import logs 
from clf_model import createInitialTrainingSet,createClassifier,predictLabels
from annotate import analyzePredictions 
import uncertaintySampling

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)   #To make sure all the columns are visible in the logs.
pd.set_option('display.width', 1000)

def get_args():
    
    parser = argparse.ArgumentParser(description="This script takes the requirement combinations as input, actively learns and predicts the dependency types.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--input","-i",type=str,required = True, help="path to requirement combinations data file")
    #parser.add_argument("--targetLabel","-tl",type=str,required=False,choices=['b','m'],default='b',help="Which label you wish to predict? Enter 'b' for BinaryClass or 'm' for MultiClass")
    parser.add_argument("--dependencyTypeNeeded",'-dt',type=str,required=False,default='y',choices=['y','n'], help = "Dependency Type Classification is needed or not.")
    parser.add_argument("--initManualAnnotAvail","-imma",type=str,required=False, choices = ['y','n'], default='y', help="Initial Manual annotations are available or not. If yes, then those Labelled 'M' will be used for Training Classifier.")
    parser.add_argument("--classifier","-clf",type=str,required=False,choices=['RF','NB','SVM'],default='RF', help="Please provide the classifier you wish to use for the prediction model - 'RF' for Random Forest / 'NB' for Naive Bayes / 'SVM' for Support Vector Machine.")
    parser.add_argument("--testsize","-ts",type=float,required=False, default=0.2,choices=[x/10 for x in range(0,11)], help="Test Split ratio. Allowed value less than 1.0")
    #parser.add_argument("--confidence",'-conf',type=float,required=False, default = 0.8, choices = [x/100 for x in range(0,101)], help = "Threshold maximum probability value for marking the prediction as confident. Allowed value less than 1")
    parser.add_argument("--trainingCount","-tc",type=int,default = 10,required = False, help="Number of manual annotations to be done initially which forms the training set.")
    parser.add_argument("--samplingType","-st",type=str,default = 'leastConfidence',choices= ['leastConfidence','minMargin','entropy'],help="Uncertainity Sampling Type : Allowed values are 'leastConfidence','minMargin','entropy'.")
    parser.add_argument("--manualAnnotationsCount","-mac",type=int,default=1,required=False, help="Number of manual annotations to be done in each iteration of active learning")
    parser.add_argument("--logPath","-lp",type=str,default=os.getcwd()+"/static/data/logs", required=False,help="Path to save the logs and outputs.")
    parser.add_argument("--comments","-c",type=str,required=False,help="Any comments you wish to add in the logs for tracking purpose.")
    
    return parser.parse_args()

def getData(fPath):

    '''
    Fetches data from the file path provided and does the preprocessing.
    1. Fills the NaN values in Labelled column with 'A' in order to mark them as to be annotated combinations.
    2. Fills the NaN values in label [BinaryClass/MultiClass] column with 0 as dummy value.
    3. Shuffles the requirement combinations
    '''

    df_data = pd.read_csv(fPath,',',encoding="utf-8")#ISO-8859-1
    df_data['BLabelled'].fillna('A',inplace=True)
    df_data['MLabelled'].fillna('A',inplace=True)
    df_data['BinaryClass'].fillna(0,inplace=True)    
    df_data['MultiClass'].fillna(0,inplace=True)
    df_data = shuffle(df_data[['req1_id','req1','req2_id','req2','BinaryClass','MultiClass','BLabelled','MLabelled']])   #shuffle's the rows and ignores columns except [req1,req2,Binary,MultiClass,Labelled]
    return df_data

def learnTargetLabel(args,df_rqmts,targetLabel):
    #Based on scenario, user can provide the initial manual annotations in the input file and mark them as 'M' in Labelled Column
    #If the user decides to provide the manual annotations on the go. Then inorder to provide the initial Manual Annotations which will form the training set,
    #User can provide the trainingCount number of annotations and the labelled data will be marked as 'M' in Labelled column.
    if targetLabel == "BinaryClass":
        labelColumn = "BLabelled"
    else:
        labelColumn = "MLabelled"

    if args.initManualAnnotAvail.lower() == 'n':     #Create Initial Training Set only if the ManualAnnotationAvailabilityFlag is 'n'
        df_rqmts = createInitialTrainingSet(df_rqmts,args.trainingCount,targetLabel)   #User provides manual annotations for trainingCount(default=10) requirement combinations.
        logs.writeLog("Combined Data Set after doing the initial Manual Annotations : \n"+str(df_rqmts))
    
    #Initial Analysis of the data available. 
    iteration = 0 

    manuallyAnnotatedCount = len(df_rqmts[df_rqmts[labelColumn]=='M'])
    intelligentlyAnnotatedCount = len(df_rqmts[df_rqmts[labelColumn]=='I'])
    toBeAnnotatedCount = len(df_rqmts[df_rqmts[labelColumn]=='A'])
    
    if targetLabel=='BinaryClass':
        #Create a dataframe to store the results after each iteration of active Learning.
        df_resultTracker = pd.DataFrame(columns=['Iteration','ManuallyAnnotated','IntelligentlyAnnotated','ToBeAnnotated','TrainingSize','ValidationSize','ClassifierTestScore','DependentCount','IndependentCount','f1Score','precisionScore','recallScore'])
        
        #Number of combinations which have been manually or intelligently labelled as dependent or independent    
        dependentCount = len(df_rqmts[(df_rqmts['BinaryClass'].astype('int')==1) & (df_rqmts[labelColumn].isin(['M','I']))])
        independentCount = len(df_rqmts[(df_rqmts['BinaryClass'].astype('int')==0) & (df_rqmts[labelColumn].isin(['M','I']))])

        #Add the initial analysis to the analysis dataFrame created.
        df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':'-','ValidationSize':'-','ClassifierTestScore':'-','DependentCount':dependentCount,'IndependentCount':independentCount,'f1Score':'-','precisionScore':'-','recallScore':'-'},ignore_index=True)

    else:
        #Create a dataframe to store the results after each iteration of active Learning...  #Added ORCount,ANDCount etc columns to keep a track of different dependency types
        df_resultTracker = pd.DataFrame(columns=['Iteration','ManuallyAnnotated','IntelligentlyAnnotated','ToBeAnnotated','TrainingSize','ValidationSize','ClassifierTestScore','AndCount','ORCount','RequiresCount','SimilarCount','CannotSayCount','f1Score','precisionScore','recallScore'])
        
        #Number of combinations which have been manually or intelligently labelled for different dependency types   
        andCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==1) & (df_rqmts[labelColumn].isin(['M','I']))])
        orCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==2) & (df_rqmts[labelColumn].isin(['M','I']))])
        requiresCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==3) & (df_rqmts[labelColumn].isin(['M','I']))])
        similarCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==4) & (df_rqmts[labelColumn].isin(['M','I']))])
        cannotSayCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==5) & (df_rqmts[labelColumn].isin(['M','I']))])

        #Add the initial analysis to the analysis dataFrame created.
        df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':'-','ValidationSize':'-','ClassifierTestScore':'-','AndCount':andCount,'ORCount':orCount,'RequiresCount':requiresCount,'SimilarCount':similarCount,'CannotSayCount':cannotSayCount,'f1Score':'-','precisionScore':'-','recallScore':'-'},ignore_index=True)
    
    logs.writeLog("\n\nInitial Data Analysis : \n"+str(df_resultTracker)+"\n")
    
    confidence = 0  #initial value of confidence; user will be providing value after looking for the probability distribution.
    
    while True:

        iteration+=1
        
        logs.writeLog("\nIteration : "+str(iteration)+"\n")
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
            return df_rqmts,df_resultTracker  
        
        if iteration >= 10: #After 10 iterations, ask user if he/she wants to continue active learner....
            while True:
                logs.writeLog("Exceeded the iteration limit. Still there are "+str(len(df_unlabelledData))+" combinations to be Labelled. Do you wish to continue Annotating? Enter 'y'/'n'")
                userInput = input()
                logs.writeLog("User's input : "+str(userInput))
                if userInput.lower()=='n':
                    logs.writeLog("Stopping Condition Reached...")
                    return df_rqmts,df_resultTracker 
                elif userInput.lower()=='y':
                    logs.writeLog("Continuing with Iteration "+str(iteration))
                    break
                else:
                    logs.writeLog ("Invalid Input. Allowed Values -- y / n")
                    continue

        logs.writeLog ("\nCreating Classifier....")
        countVectorizer, tfidfTransformer, classifier, classifierScore,trainSize,validationSize,f1Score,precisionScore,recallScore = createClassifier(args.classifier,args.testsize,df_labelledData,targetLabel)  
        
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
            
            df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':trainSize,'ValidationSize':validationSize,'ClassifierTestScore':classifierScore,'DependentCount':dependentCount,'IndependentCount':independentCount,'f1Score':f1Score,'precisionScore':precisionScore,'recallScore':recallScore},ignore_index=True)
            logs.writeLog("\nAnalysis DataFrame : \n"+str(df_resultTracker))  
        
        else:
            manuallyAnnotatedCount = len(df_rqmts[df_rqmts['MLabelled']=='M'])
            intelligentlyAnnotatedCount = len(df_rqmts[df_rqmts['MLabelled']=='I'])
            toBeAnnotatedCount = len(df_rqmts[df_rqmts['MLabelled']=='A'])
        
            andCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==1) & (df_rqmts['MLabelled'].isin(['M','I']))])
            orCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==2) & (df_rqmts['MLabelled'].isin(['M','I']))])
            requiresCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==3) & (df_rqmts['MLabelled'].isin(['M','I']))])
            similarCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==4) & (df_rqmts['MLabelled'].isin(['M','I']))])
            cannotSayCount = len(df_rqmts[(df_rqmts['MultiClass'].astype('int')==5) & (df_rqmts['MLabelled'].isin(['M','I']))])

            df_resultTracker = df_resultTracker.append({'Iteration':iteration,'ManuallyAnnotated':manuallyAnnotatedCount,'IntelligentlyAnnotated':intelligentlyAnnotatedCount,'ToBeAnnotated':toBeAnnotatedCount,'TrainingSize':trainSize,'ValidationSize':validationSize,'ClassifierTestScore':classifierScore,'AndCount':andCount,'ORCount':orCount,'RequiresCount':requiresCount,'SimilarCount':similarCount,'CannotSayCount':cannotSayCount,'f1Score':f1Score,'precisionScore':precisionScore,'recallScore':recallScore},ignore_index=True)
            logs.writeLog("\nAnalysis DataFrame : \n"+str(df_resultTracker))
        
        
    return df_rqmts,df_resultTracker

def main():
    
    warnings.simplefilter(action='ignore', category=FutureWarning)  #Ignore Future warnings if any occur. 
    
    #initialize directory which contains all the data and which will contain logs and outputs
    currentFileDir = os.getcwd()+"/static/data/"     
    
    args=get_args()  #Get all the command line arguments
    options = vars(args)  #Stores the arguments as dictionary ; used in logs
    
    ifileName = args.input     
    clf = args.classifier
    logPath = args.logPath
    comments = args.comments
    dependencyTypeNeeded = args.dependencyTypeNeeded

    logFilePath,OFilePath = logs.createLogs(logPath,clf,comments)   #Creates the log file, default value is os.getcwd()+"/static/data/logs/" ; user still can provide his own logPath if needed.
    logs.writeLog("\n"+100*"-")
    logs.writeLog("\nArguments : \n")  #Writes all the arguments provided in command line (including the default values.)
    for key,value in options.items():
        logs.writeLog(str(key)+" : "+str(value)+"\n")
    logs.writeLog(100*"-")
    df_rqmtData = getData(currentFileDir+ifileName)   
    logs.writeLog("\n\nData Fetched from the input file : "+str(len(df_rqmtData))+" Rows \n"+str(df_rqmtData[:10]))
    
    logs.writeLog("\n\nStep 1 :- Learning BinaryClass Label\n")
    df_rqmtComb,df_BinaryAnalysis = learnTargetLabel(args,df_rqmtData,'BinaryClass')
        
    logs.addOutputToExcel(df_BinaryAnalysis,"\nAnalysis of BinaryClass Label Classification : \n")
    input("Hit Enter to Proceed....")
    if dependencyTypeNeeded == 'y':
        logs.writeLog("\n\nStep 2 :- Learning MultiClass Label\n")
        df_rqmtComb = df_rqmtComb[df_rqmtComb['BinaryClass']==1]   #Filtering only the Dependent Requirement Combinations for low level classification
        df_rqmtComb['MLabelled'] = df_rqmtComb['MLabelled'].replace(' ','A')   #Mark the intelligently annotated combinations as unlabelled. (But what about the manual annotations done?)
        logs.writeLog("Following is the data set to be used for MultiClass classification : "+str(len(df_rqmtComb))+" Rows\n"+str(df_rqmtComb[:10]))
        
        if len(df_rqmtComb[df_rqmtComb['MLabelled']=='A'])>0:
            df_rqmtComb,df_MultiAnalysis = learnTargetLabel(args,df_rqmtComb,'MultiClass')
            logs.addOutputToExcel(df_MultiAnalysis,"Analysis of MultiClass Label Classification : \n")
        else:
            logs.writeLog("\n\nThere are no dependent combinations. So its not possible to find the Dependency Types.")

        
    
    logs.writeLog("\nOutput Analysis is available at : "+str(OFilePath))
    logs.writeLog("\nLogs are available at : "+str(logFilePath))
    
if __name__ == '__main__':
    main()