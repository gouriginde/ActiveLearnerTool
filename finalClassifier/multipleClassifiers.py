import os
import numpy as np
import pandas as pd
import warnings
import logs
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,classification_report
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)   #To make sure all the columns are visible in the logs.
pd.set_option('display.width', 1000)

def getData(fPath):
    '''
    Fetches data from the file path provided
    1. Removes rows with NaN values.
    2. Shuffles the requirement combinations
    '''

    df_data = pd.read_csv(fPath,',',encoding="utf-8")#ISO-8859-1
    df_data.dropna(inplace=True)
    df_data = shuffle(df_data[['comb_id','req1','req2','BinaryClass','MultiClass']])   #shuffle's the rows and ignores columns except [req1,req2,Binary,MultiClass,Labelled]
    return df_data

def prepareData(splitratio,df_labelledData,df_validationData,targetLabel):
    '''
    Splits the dataset into train/test set.
    Passes train and test sets through NLP pipeline
    Returns X_train,X_test,y_train,y_test
    '''
    logs.writeLog("\n\nPerforming Balancing of Data....")
    df_labelledData[targetLabel] = df_labelledData[targetLabel].astype('int')    #Making sure the values are integer only and not float... 
    df_LabelledDataOriginal = df_labelledData  #Backup original Data
    #######################################DATA BALANCING########################################################
    #Create empty dataframes to store the Balanced Combinations .... Making sure equal number of combinations corresponding to all label are available in train and test sets.
    df_testSet = pd.DataFrame(columns=df_labelledData.columns)
    df_trainSet = pd.DataFrame(columns=df_labelledData.columns)
    print ("df_labelledData shape :",df_labelledData.shape)
    print ("df_trainSet shape : ",df_trainSet.shape)
    print ("df_testSet shape : ",df_testSet.shape)

    stats = df_labelledData[targetLabel].value_counts()  #Returns a series of number of different types of TargetLabels (values) available with their count.
    min_value_count = stats.min()  #Calculate minimum value count out of all labels.... will extract this number of combinations of each label type.
    
    #Calcalate the Test Size and Train Size... number of combinations to be sampled for each LABEL type.
    test_size = int(min_value_count*splitratio) if (int(min_value_count*splitratio)>=1) else 1  #added if else condition in case test size is less than 1. then minimum size should be 1.
    train_size = min_value_count - test_size
    print ("Train Size : ",train_size)
    print ("Test Size : ",test_size)

    #For each type of label
    for key in stats.keys():
        
        #Sample out some values for Test Set
        df_sampleTest = df_labelledData[df_labelledData[targetLabel]==key].sample(test_size)
        df_labelledData = df_labelledData[~df_labelledData.isin(df_sampleTest)].dropna()  #Remove Sampled Values from original data set.
        df_testSet = pd.concat([df_testSet,df_sampleTest],axis=0)   #Add sampled values into the Test Set
        
        #Sample out some values for Train Set
        df_sampleTrain = df_labelledData[df_labelledData[targetLabel]==key].sample(train_size)
        df_labelledData = df_labelledData[~df_labelledData.isin(df_sampleTrain)].dropna()  #Remove Sampled Values from original data set.
        df_trainSet = pd.concat([df_trainSet,df_sampleTrain],axis=0)  #Add sampled values into the Test Set
        
    #Shuffle both Train and Test Set....
    df_trainSet = shuffle(df_trainSet)
    df_testSet = shuffle(df_testSet)
    
    #Split Train Test Sets into X_train,y_train,X_test,y_test   (Similar to Train Test Split....)
    X_train = df_trainSet.loc[:,['req1','req2']]
    y_train = df_trainSet.loc[:,targetLabel]
    X_test = df_testSet.loc[:,['req1','req2']]
    y_test = df_testSet.loc[:,targetLabel]
    
    all_features = df_LabelledDataOriginal.loc[:,['req1','req2']]
    
    #separate labels and features from validation set.
    val_features = df_validationData.loc[:,['req1','req2']]
    val_labels = df_validationData.loc[:,targetLabel]

    logs.writeLog("\n\nTraining Set Size : "+str(len(X_train)))
    logs.writeLog("\nTrain Set Value Count : \n"+str(df_trainSet[targetLabel].value_counts()))

    logs.writeLog("\n\nTest Set Size : "+str(len(X_test)))
    logs.writeLog("\nTest Set Value Count : \n"+str(df_testSet[targetLabel].value_counts()))

    logs.writeLog("\n\nValidation Set Size : "+str(len(val_features)))
    logs.writeLog("\nValidation Set Value Count : \n"+str(df_validationData[targetLabel].value_counts()))

    logs.writeLog("\n\nPassing features through NLP Pipeline....")
    #Pass the feature sets through nlp pipeline for feature extraction. (Fits on X_train and transforms X_test and val_features)
    allLabelledFeatures_processed,trainFeatures_processed,testFeatures_processed,valFeatures_processed = nlpPipeline(all_features,X_train,X_test,val_features)
    #returned values are numpy arrays (dataframe to array conversion takes place in nlp pipeline)

    trainLabels = np.array(y_train).astype('int')
    testLabels = np.array(y_test).astype('int')
    valLabels = np.array(val_labels).astype('int')
    
    return trainFeatures_processed,trainLabels,testFeatures_processed,testLabels,valFeatures_processed,valLabels,allLabelledFeatures_processed
    
def nlpPipeline(allLabelledFeatures,trainFeatures,testFeatures,valFeatures):
    '''
    Perform Count Vectorizer and Tfidf Transformation
    1. Fit and Transform on train features
    2. Then using vectorizer and transformer transform test and validation features.

    Returns train, test, validation feature sets after transformations.
    '''

    #Initialize Count Vectorizer which in a way performs Bag of Words on train features
    count_vect = CountVectorizer(tokenizer=lambda doc: doc, analyzer=split_into_lemmas, lowercase=False, stop_words='english')
    train_feature_count= count_vect.fit_transform(np.array(trainFeatures))
    
    #Transform test, All Labelled, validation features
    test_feature_count = count_vect.transform(np.array(testFeatures))
    val_feature_count = count_vect.transform(np.array(valFeatures))
    all_feature_count = count_vect.transform(np.array(allLabelledFeatures))

    #Initialize TFIDF Transformer, fit_transform on train feature counts
    tfidf_transformer = TfidfTransformer()
    train_feature_tfidf= tfidf_transformer.fit_transform(train_feature_count)
    
    #Transform test, All Labelled validation features
    test_feature_tfidf = tfidf_transformer.transform(test_feature_count)
    val_feature_tfidf = tfidf_transformer.transform(val_feature_count)
    all_feature_tfidf = tfidf_transformer.transform(all_feature_count)

    #Transform
    return (all_feature_tfidf,train_feature_tfidf,test_feature_tfidf,val_feature_tfidf)

def createClassifier(clf_name,train_features,train_labels) :     
    '''
    Create Classifier (Rf/SVM/NB)
    '''
    #Random Forest Classifier Creation
    if clf_name == "RF" :
        clf_model = RandomForestClassifier().fit(train_features,train_labels)
        
        
    #Naive Bayes Classifier Creation
    elif clf_name == "NB":
        clf_model = MultinomialNB().fit(train_features,train_labels)

    #Support Vector Machine Classifier Creation.
    elif clf_name == "SVM":
        clf_model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False).fit(train_features,train_labels)
    
    logs.writeLog("\n"+clf_name+" Classifier created.")

    return clf_model

def testClassifier(clf_model,clf_name,test_features,test_labels):
    '''
    Classify test features and compare output with actual labels

    print accuracy score, f1, precision, recall and print classification report.
    '''
    #Make predictions
    predict_labels = clf_model.predict(test_features)
    
    actualLabels = test_labels
    labelClasses = list(set(actualLabels))   
    
    #Calculate accuracy
    clf_test_score = clf_model.score(test_features,actualLabels)
    logs.writeLog ("\n"+clf_name+" Classifier Test Score : "+str(clf_test_score))
    
    #Calculate f1, precision, recall scores
    f1 = round(f1_score(actualLabels, predict_labels,average='macro'),2)
    precision = round(precision_score(actualLabels, predict_labels,average='macro'),2)
    recall = round(recall_score(actualLabels, predict_labels,average='macro'),2)
    
    logs.writeLog ("\n\nClassification Report : \n\n"+str(classification_report(actualLabels,predict_labels)))
    cm = confusion_matrix(actualLabels,predict_labels,labels=labelClasses)    
    logs.writeLog ("\n\nConfusion Matrix : \n"+str(cm)+"\n")
    
    #Create a box plot.....
    return clf_test_score,f1,precision,recall

def validateClassifier(clf_model,clf_name,validation_features,validation_labels):
    '''
    Classify validation features and compare output with actual labels

    print accuracy score.
    '''
    #Make predictions
    predict_labels = clf_model.predict(validation_features)
    
    actualLabels = validation_labels
    labelClasses = list(set(actualLabels))   
    
    #Calculate accuracy
    clf_validation_score = clf_model.score(validation_features,actualLabels)
    logs.writeLog ("\n"+clf_name+" Classifier Validation Score : "+str(clf_validation_score))

    return clf_validation_score

def split_into_lemmas(text):
    text = str(text)
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)  #Ignore Future warnings if any occur. 
    
    #initialize directory which contains all the data and which will contain logs and outputs
    currentFileDir = os.getcwd()
    args = logs.getArguments(currentFileDir+"/clfParams.txt") 

    ifileName = args.loc[0,'labelledData']     
    vfileName = args.loc[0,'validationData']
    comments = args.loc[0,'comments']
    targetLabel = ['BinaryClass' if args.loc[0,'tobepredicted']=='b' else 'MultiClass'][0]
    clf_list = args.loc[0,'classifiersList'].split(";")  #ClassifiersList is a string (RF;NB;SVM)
    testsize = float(args.loc[0,'testsize'])
    logFilePath,OFilePath = logs.createLogs(currentFileDir+"/static/data/Logs",args,comments)   #Creates the log file, default value is os.getcwd()+"/static/data/logs/" ; user still can provide his own logPath if needed.
    
    #Extract and remove NaN's from the labelled data file and validation data
    df_rqmtDataOriginal = getData(currentFileDir+"/static/data/"+ifileName)   
    logs.writeLog("\n\nData Fetched from the Labelled combinations file : "+str(len(df_rqmtDataOriginal))+" Rows \n"+str(df_rqmtDataOriginal[:10]))
    
    df_valData = getData(currentFileDir+"/static/data/"+vfileName)   
    logs.writeLog("\n\nData Fetched from the Validation combinations file : "+str(len(df_valData))+" Rows \n"+str(df_valData[:10]))
    
    #DataFrame to store the results
    df_Scores = pd.DataFrame(columns=["Classifier","Test_Accuracy","F1_Score","Precision","Recall","Validation_Accuracy","CV_5_Score","CV_10_Score","TrainCount","TestCount","ValidationCount"])

    count = 0
    while count < 3:
        count +=1
        df_rqmtData = df_rqmtDataOriginal[['comb_id','req1','req2','BinaryClass','MultiClass']]
        
        logs.writeLog("\n\n"+"."*10+"Iteration "+str(count)+"."*10)
        
        #Segregate train,validation and test sets; perform data balancing on train and test set ; pass features through nlp pipeline
        logs.writeLog("\n\nPreparing DataSet for classification....")
        
        validationSetOriginal = df_valData  #Validation set is not being shuffled, it will be same in all iterations..
        
        trainFeatures,trainLabels,testFeatures,testLabels,validationFeatures,validationLabels,allLabelledFeatures = prepareData(testsize,df_rqmtData,df_valData,targetLabel)

        logs.writeLog ("\n\nCreating Classifier....")
        clfName = "RF"
        clfModel = createClassifier(clfName,trainFeatures,trainLabels)         
        logs.writeLog("\n\n"+clfName+" Classifier Created.")
        
        logs.writeLog ("\n\nTest Classifier....")
        clf_test_score,f1_score,precision,recall = testClassifier(clfModel,clfName,testFeatures,testLabels)

        logs.writeLog ("\n\nValidate Classifier....")
        clf_validation_score = validateClassifier(clfModel,clfName,validationFeatures,validationLabels)

        logs.writeLog("\n\nPerforming StratfiedKFold (5) cross validation....")
        crossv_5 = StratifiedKFold(5)
        scores_5 = cross_val_score(clfModel, validationFeatures, validationLabels, cv=crossv_5) #https://scikit-learn.org/stable/modules/cross_validation.html
        accuracy_cv5 = str(round(scores_5.mean(),2))+" +/- "+str(round(scores_5.std(),2) * 2)
        logs.writeLog("\nAccuracy: "+accuracy_cv5)
        
        logs.writeLog("\n\nPerforming StratfiedKFold (10) cross validation....")
        crossv_10 = StratifiedKFold(10)
        scores_10 = cross_val_score(clfModel, validationFeatures, validationLabels, cv=crossv_10) 
        accuracy_cv10 = str(round(scores_10.mean(),2))+" +/- "+str(round(scores_10.std(),2) * 2)
        logs.writeLog("\nAccuracy: "+accuracy_cv10)

        #Update Results in dataFrame
        df_Scores = df_Scores.append({'Classifier':clfName,'Test_Accuracy':clf_test_score,'F1_Score':f1_score,'Precision':precision,'Recall':recall,'Validation_Accuracy':clf_validation_score,'CV_5_Score':accuracy_cv5,'CV_10_Score':accuracy_cv10,'TrainCount':len(trainLabels),'TestCount':len(testLabels),'ValidationCount':len(validationLabels)},ignore_index=True)
 
        #Predicting Probabilities for All Labelled Set (from this set itself train and test set were segregated after data balancing....)
        logs.writeLog("\n\nPredicting Labels...")
        predict_labels_all = clfModel.predict(allLabelledFeatures)
        predict_prob_all = clfModel.predict_proba(allLabelledFeatures)
        
        #Segregating the probability value for each label (0 and 1) will be saving them in different columns
        pp_all_Label_0 = pd.Series(predict_prob_all[:,0])
        pp_all_Label_1 = pd.Series(predict_prob_all[:,1])
        
        #Update predicted values into the original training set
        df_rqmtData.reset_index(inplace=True)
        df_rqmtData['PredictedLabel']=predict_labels_all[:]
        df_rqmtData['Predict_Proba_Class_'+str(clfModel.classes_[0])]=pp_all_Label_0
        df_rqmtData['Predict_Proba_Class_'+str(clfModel.classes_[1])]=pp_all_Label_1
        
        #Predicting Probabilities for Validation Set
        predict_labels_val = clfModel.predict(validationFeatures)
        predict_prob_val = clfModel.predict_proba(validationFeatures)
        
        #Segregating the probability value for each label (0 and 1) will be saving them in different columns
        pp_val_Label_0 = pd.Series(predict_prob_val[:,0])
        pp_val_Label_1 = pd.Series(predict_prob_val[:,1])
        
        #Update predicted values into the original training set
        validationSetOriginal['PredictedLabel']=predict_labels_val[:]
        validationSetOriginal['Predict_Proba_Class_'+str(clfModel.classes_[0])]=pp_val_Label_0
        validationSetOriginal['Predict_Proba_Class_'+str(clfModel.classes_[1])]=pp_val_Label_1
        
        #Save the results in .csv files with respect to the count.
        resultsPath = currentFileDir+"/static/data/results"
        logs.writeLog("\n\nSaving Results")
        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)

        df_rqmtData.to_csv(resultsPath+"/LabelledSet"+str(count)+".csv")
        validationSetOriginal.to_csv(resultsPath+"/ValidationSet"+str(count)+".csv")
    
    #Dump results obtained in csv file for tracking.
    logs.writeLog("\n")
    logs.addOutputToExcel(df_Scores,"Results Obtained") 
    
    #Provide locations of results to the user.
    logs.writeLog("\n\nResults obtained are available at : /static/data/results.")
    logs.writeLog("\n\nLogs are available at : "+str(logFilePath))

    
if __name__ == '__main__':
    main()