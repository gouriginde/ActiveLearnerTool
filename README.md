This is a tool developed to process the requirements of a project.
Input is the file with a pair of requirements and corresponding labels.

Final Classifier Module :- Here, we expect the complete labelled dataset to be divided into two sets (Labelled Data and Validation Data in csv files) containing "req1,req2,BinaryClass,MultiClass" columns.

For the given dataset and label to be predicted (Binary/MultiClass), each of the classifiers provided by the user in classifiersList is created by train data, test and validation data are used to compare the outputs with actual labels and the results are stored for each classifier. 

NLP pipeline -> Stop Words Removal, Count Vectorizer,Tfidf Transformation

Below are the run-time parameters involved for this module.
 
labelledData       :  path to labelled requirement combinations data file 
validationData     :  path to validation requirement combinations data file 
tobepredicted      :  Labels to be predicted binary - 'b' / multiclass - 'm'. {b/m}
classifiersList	   :  List classifiers (semicolon separated) to be used for the prediction model - eg, RF;SVM;NB. Use 'RF' for                          Random Forest / 'NB' for Naive Bayes / 'SVM' for Support Vector Machine.
testsize 	       :  Test Split ratio. Allowed value less than 1.0 {0.1,0.2....0.9,1.0}
comments 	       :  Any comments you wish to add in the logs for tracking purpose
 
Sample input files are available in static/data folder. While output files are available in static/data/logs folder for both binary and multiclass classification.