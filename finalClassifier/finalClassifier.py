import os
import numpy as np
import pandas as pd
import warnings
import argparse
import logs
import ast

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target

from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)   #To make sure all the columns are visible in the logs.
pd.set_option('display.width', 1000)

def get_args():
    
    parser = argparse.ArgumentParser(description="This script takes the requirement combinations as input and computes the cross fold validations.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--input","-i",type=str,required = True, help="path to requirement combinations data file")
    parser.add_argument("--app","-a",type=str,required=False,default='AL',choices = ['AL','DD'],help="Enter the application for which you want to use Final Classifier [AL/DD]")
    parser.add_argument("--comments","-c",type=str,required=False,help="Any comments you wish to add in the logs for tracking purpose.")
        
    return parser.parse_args()

def getData(fPath,app,cols):
    '''
    Fetches data from the file path provided
    1. Removes rows with NaN values.
    2. Shuffles the requirement combinations
    '''

    df_data = pd.read_csv(fPath,',',encoding="utf-8")#ISO-8859-1
    df_data = df_data[cols]   #ignores columns except [req1,req2,MultiClass]
        
    if app=='AL':
        df_data = df_data[(df_data['MultiClass']!=0) & (df_data['MLabelled']!='A')]

    df_data.dropna(inplace=True)    
    return df_data

def balanceData(df_labelledData,targetLabel):
    '''
    Balances the data as per MultiClass Values
    
    returns Balanced dataset
    '''
    df_labelledData[targetLabel] = df_labelledData[targetLabel].astype('int')    #Making sure the values are integer only and not float... 
    
    stats = df_labelledData[targetLabel].value_counts()  #Returns a series of number of different types of TargetLabels (values) available with their count.
    min_value_count = stats.min()  #Calculate minimum value count out of all labels.... will extract this number of combinations of each label type.
    #print (stats)
    sample_size = min_value_count
    df_BalancedSet = pd.DataFrame(columns=df_labelledData.columns)
    
    #For each type of label
    for key in stats.keys():
        
        #Sample out some values for Labelled Set
        df_sample = df_labelledData[df_labelledData[targetLabel]==key].sample(sample_size)
        df_labelledData = df_labelledData[~df_labelledData.isin(df_sample)].dropna()  #Remove Sampled Values from original data set.
        df_BalancedSet = pd.concat([df_BalancedSet,df_sample],axis=0)   #Add sampled values into the Test Set
    
    logs.writeLog("\n\nBalanced Set Size : "+str(len(df_BalancedSet)))
    logs.writeLog("\nBalanced Set Value Count : \n"+str(df_BalancedSet[targetLabel].value_counts()))

    return df_BalancedSet

def nlpPipeline(df_data,targetLabel):
    '''
    Passes the given dataset through NLP pipeline and returns features and labels.
    '''
    features = df_data.loc[:,['req1','req2']]
    labels = np.array(df_data.loc[:,targetLabel].astype('int'))
    
    count_vect = CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    features_count = count_vect.fit_transform(np.array(features))

    tfidf_transformer = TfidfTransformer()
    features_tfidf = tfidf_transformer.fit_transform(features_count)

    return features_tfidf,labels

def my_tokenizer(arr):
    req1_list = ast.literal_eval(arr[0])
    req2_list = ast.literal_eval(arr[1])
    return req1_list+req2_list

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)  #Ignore Future warnings if any occur. 
    
    #initialize directory which contains all the data and which will contain logs and outputs
    currentFileDir = os.getcwd()
    
    args=get_args()  #Get all the command line arguments
    options = vars(args)  #Stores the arguments as dictionary ; used in logs
    
    ifileName = args.input     
    app = args.app
    comments = args.comments
 
    if app == "AL":
        targetLabel = 'MultiClass'
        cols = ['req1','req2','MultiClass','MLabelled']
    else:
        targetLabel = 'Dependency'
        cols = ['req1','req2','Dependency']

    logFilePath,OFilePath = logs.createLogs(currentFileDir+"/static/data/Logs","ensemble",comments)   #Creates the log file, default value is os.getcwd()+"/static/data/logs/" 

    #Extract and remove NaN's from the data file
    df_rqmtDataOriginal = getData(currentFileDir+"/static/data/"+ifileName,app,cols)   
    logs.writeLog("\n\nData Fetched from the combinations file : "+str(len(df_rqmtDataOriginal))+" Rows \n"+str(df_rqmtDataOriginal[:10]))
    input (".......")
    #Perform data balancing on complete dataset; pass features through nlp pipeline
    logs.writeLog("\n\nPerforming Balancing of Data....")
    df_rqmtBalanced = balanceData(df_rqmtDataOriginal,targetLabel)
    
    logs.writeLog("\n\nPreparing DataSet for classification....")
    features,labels = nlpPipeline(df_rqmtBalanced,targetLabel)
    
    logs.writeLog("\n\nDefining Ensemble Model...")
    
    rf_model = RandomForestClassifier()
    nb_model = MultinomialNB()
    svm_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
    
    clf_model = VotingClassifier(estimators=[('RF', rf_model), ('NB', nb_model),('SVM',svm_model)], voting='hard')  
    
    logs.writeLog("\n\nPerforming Five Fold Cross Validation")
    crossv_5 = StratifiedKFold(5)
    scores_5 = cross_val_score(clf_model, features, labels, cv=crossv_5) #https://scikit-learn.org/stable/modules/cross_validation.html
    
    scores_5 = str(round(scores_5.mean(),2)) +"(+/- "+str(round(scores_5.std()*2,2))+")"
    
    logs.writeLog("Score : "+scores_5)

    logs.writeLog("\n\nPerforming Ten Fold Cross Validation")
    crossv_10 = StratifiedKFold(10)
    scores_10 = cross_val_score(clf_model,features,labels,cv=crossv_10)
    scores_10 = str(round(scores_10.mean(),2)) +"(+/- "+str(round(scores_10.std()*2,2))+")"
    
    logs.writeLog("Score : "+scores_10)

    logs.writeLog("\n\nPerforming Five Times Five Fold Cross Validation")
    rskf_5 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=36851234)
    scores_r5 = cross_val_score(clf_model,features,labels,cv=rskf_5)
    scores_r5 = str(round(scores_r5.mean(),2)) +"(+/- "+str(round(scores_r5.std()*2,2))+")"
    
    logs.writeLog("Score : "+scores_r5)

    logs.writeLog("\n\nPerforming Ten Times Ten Fold Cross Validation")
    rskf_10 = RepeatedStratifiedKFold(n_splits=10, n_repeats=10,random_state=36851234)
    scores_r10 = cross_val_score(clf_model,features,labels,cv=rskf_10)
    scores_r10 = str(round(scores_r10.mean(),2)) +"(+/- "+str(round(scores_r10.std()*2,2))+")"
    
    logs.writeLog("Score : "+scores_r10)


if __name__ == '__main__':
    main()





'''
1) Remove binary class option - D
2) Read whole dataset and balance data for multiclass - D
4) Process data from NLP pipeline (Combine Train+Test+Validation; no need to split)
3) Add only ensemble model (Hard Voting)
5) Provide model and do cross_validation (5 and 10)
6) Repeate KFold (SKLearn)


Add NLP to Final Classifier
Add Ensemble to Final Classifier

Print Confusion Matrix 
'''