from flask import Flask, url_for, request, render_template,redirect,flash
from contentManagement import content, getServerDetails,lookupValue
from csv_operations import verifyChanges,get_random_req_combination
import os
import pandas as pd
import numpy as np
srvDict = getServerDetails(input_file = "settings.txt")

contentDict = content()

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    
    try:
        fileName = 'blackLine-combined.csv'  # File name
        fPath = os.getcwd()+"/static/data/"+fileName
        rqmt_df = pd.read_csv(fPath)

        rqmtDict = get_random_req_combination(rqmt_df)
        
        #print()
        if request.method=="POST":
            #print ("Retreieved Values....")
            retrievedValues = { 'combo_id' : request.form['ComboID'],
                                'req1_id' : request.form['rqmt-1'],
                                'req1': request.form['rqmtVal-1'],
                                'req2_id': request.form['rqmt-2'],
                                'req2': request.form['rqmtVal-2'],
                                'BinaryClass': lookupValue('dependencyFlag',request.form['dependency_Flag'])
                                }
            if retrievedValues['BinaryClass'] == 1:
                retrievedValues['MultiClass'] = lookupValue('dependencyType',request.form['dependencyType'])
            else:
                retrievedValues['MultiClass']=np.NaN
            verifyChanges(rqmt_df, retrievedValues,fPath)
    except Exception as e:
        flash(e)
    return render_template("index.html",content_Dict = contentDict,rqmt_Combo = rqmtDict)

if __name__ == '__main__':
    app.run(host=srvDict['host'],port=int(srvDict['port']),debug=srvDict['debug_mode'])