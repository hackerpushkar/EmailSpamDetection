import pandas as pd 
from sklearn.preprocessing import LabelEncoder

#read csv 
df = pd.read_csv("CSV_Files/spam.csv",on_bad_lines='skip')

# Filter 1 : Remove Unwanted Columns 
filter_df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
print(filter_df)

# Filter 2 : Remove Duplicate 
before_filter2_rows = len(filter_df)
filter_df = filter_df.drop_duplicates(keep="first")
# print(f"Duplicates Rows Drop Successfully\n"
#       f"Removed Rows: {before_filter2_rows - len(filter_df)}\n\n"
#       f"Current Rows: {len(filter_df)}")

#Filter 3 : Decorate Data 
#Rename
filter_df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)


data = filter_df.copy() #copy dataframe 


#Encoding categorical text labels into numeric values

encoder = LabelEncoder()
data['target'] = encoder.fit_transform(data['target'])

data.to_csv("CSV_Files/SPAM_v2.csv",index=False)