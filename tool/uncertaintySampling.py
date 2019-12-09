import pandas as pd
import logs
from scipy.stats import entropy

def leastConfidenceSampling(df_uncertain):
    df_uncertain['lconf']=1-df_uncertain['maxProb']
    df_uncertain = df_uncertain.sort_values(by=['lconf'],ascending=False)
    logs.writeLog("\n\nLeast Confidence Calculations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #logs.writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex

def minMarginSampling(df_uncertain):
    df_uncertain['Margin'] = [max(x)-min(x) for x in df_uncertain['predictedProb']]
    #logs.writeLog(str(df))
    df_uncertain = df_uncertain.sort_values(by=['Margin'],ascending=True)
    logs.writeLog("\n\nMin Margin Calcuations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #logs.writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex

def entropySampling(df_uncertain):
    df_uncertain['entropy'] = [entropy(x) for x in df_uncertain['predictedProb']]
    #logs.writeLog(str(df))
    df_uncertain = df_uncertain.sort_values(by=['entropy'],ascending=False)
    logs.writeLog("\n\nEntropy Calculations..."+str(len(df_uncertain))+" Rows\n"+str(df_uncertain[:10]))
    #logs.writeLog(str(df.index.values[0]))
    sampleIndex = df_uncertain.index.values[0]
    return sampleIndex