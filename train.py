# %%
#@ IMPORTS

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import pickle

# PARAMETERS (normally given through CLI for eg.)

control_var = {'min_samples_leaf': 3,
                'max_depth': 15,
                'n': 100              # number of estimators

}
output_file = 'model.bin'

# %% 
# DATA PREPARATION
data = pd.read_csv('Leads.csv')

data.columns = data.columns.str.lower().str.replace(' ', '_') # Lets make column names a bit more consistent

categorical_columns = list(data.dtypes[data.dtypes == 'object'].index)

# Lets make categorical data in relevant columns consistent as well
for c in categorical_columns:
    data[c] = data[c].str.lower().str.replace(' ', '_')


# categorical variables selected to keep based on EDA for each column
selected_cat_var = {'tags': ['will_revert_after_reading_the_email',
  'ringing',
  'interested_in_other_courses'],
 'lead_source': ['google', 'direct_traffic', 'olark_chat'],
 'specialization': ['select',
  'finance_management',
  'human_resource_management'],
 'last_activity': ['email_opened', 'sms_sent', 'olark_chat_conversation'],
 'last_notable_activity': ['modified', 'email_opened', 'sms_sent'],
 'how_did_you_hear_about_x_education': ['select',
  'online_search',
  'word_of_mouth'],
 'city': ['mumbai', 'select', 'thane_&_outskirts'],
 'what_is_your_current_occupation': ['unemployed',
  'working_professional',
  'student'],
 'lead_profile': ['select', 'potential_lead', 'other_leads'],
 'lead_origin': ['landing_page_submission', 'api', 'lead_add_form'],
 'lead_quality': ['might_be', 'not_sure', 'high_in_relevance']}

def replace_data(data, allowed_data_list):
    if data not in allowed_data_list:
        return 'other'
    return data

# Apply the function to limit the unique values in categorical columns that had more than 3 unique values
columns = [
'tags',
'lead_source'   ,                                   
'specialization'  ,                                   
'last_activity'  ,                                    
'last_notable_activity' ,                             
'how_did_you_hear_about_x_education',                
'city'         ,                                       
'what_is_your_current_occupation',                    
'lead_profile'    ,                                    
'lead_origin' ,                                       
'lead_quality']

for c in columns:
    data[c] = data[c].apply(replace_data, allowed_data_list=selected_cat_var[c])

# Dealing with NaNs
data[categorical_columns]=data[categorical_columns].fillna('unk')

to_drop=['asymmetrique_activity_score','asymmetrique_profile_score', 'lead_number']
data = data.drop(to_drop, axis=1)
data.dropna(axis=0, how='any', inplace=True)

# %% 
# SPLIT THE DATASET
df_full_train, df_test = train_test_split(data, test_size=0.2,  random_state=42)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
# Columns to keep for training
# columns reduced after EDA 
categorical = ['lead_origin', 'lead_source', 'do_not_email', 'last_activity', 'how_did_you_hear_about_x_education', 
 'what_is_your_current_occupation', 'what_matters_most_to_you_in_choosing_a_course', 'tags', 'lead_quality', 
 'lead_profile', 'city', 'asymmetrique_activity_index', 'last_notable_activity']

numerical = ['totalvisits', 'page_views_per_visit']

columns = categorical + numerical

# %% 
# Functions for training and predicting

# %%
def train(df_train, y_train, columns, control_var):
    
    dicts = df_train[columns].to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model = RandomForestClassifier(n_estimators=control_var['n'], 
                                    max_depth=control_var['max_depth'],
                                    min_samples_leaf=control_var['min_samples_leaf'],
                                    random_state=42,
                                    )
 # we use max_iter to avoid warnings
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model, columns):
    dicts = df[columns].to_dict(orient = 'records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1] # probabilities only for positive examples
    
    return y_pred

# %% 
# ### FINAL MODEL

print('Training the final model...')

dv, model = train(df_full_train, df_full_train.converted.values, columns, control_var)

y_test = df_test.converted.values

y_pred = predict(df_test, dv, model, columns)

auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc: .3f}')

# %% 
# SAVE THE MODEL

with open(output_file, 'wb') as f_out:

    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
