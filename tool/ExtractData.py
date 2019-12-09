'''
This program is to do the following
-- Read the 4 files in a single dataframe
-- Check if they are all aligned
-- in a loop only extract the samples which have depends_on and blocks on type="enhancement" Or "task" 
i.e ignore all the bugs.
-- for every pair create a dependency pair with following fields
req1, req2, req1_id, req2_id, req1_priority, req2_priority, cosise_similarity, semantic similarity, BinaryClass, MultiClass fields

Logic:
Step1: #dependent pairs (positive samples)
For every row: req1 in the dataframe
    extract the dependes_on and blocks fields
     for every entry: req2 
        if it is of the type "enhancement/task"
            create a pair req1, req2 and add to new dataframe

Step2: Independent pairs (negative samples)
For all the pairs which do not exists in the step1 do this comparison based on ids
create pairs and add them as independent pairs in dataframe.
'''

import pandas as pd
def main():
    #1 read the files and create a big file
    df_all_data = pd.read_csv("Data_part1.csv")
    df_all_data.append(pd.read_csv("Data_part2.csv"))
    df_all_data.append(pd.read_csv("Data_part3.csv"))
    df_all_data.append(pd.read_csv("Data_part4.csv"))
    print(df_all_data.len())
    #2 Step1: Dependent pairs

    #3 Step2: Independent pairs

    #Create a network to generated dependency graph

    pass

main()