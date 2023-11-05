import requests

url = "http://localhost:9696/predict"

prospect_id = "a71fc1e8-9edf-4a7a-9678-1db27cc35924" # this is indicative, ideally it should correspond to the info

# not a hot lead from the test dataset 

# prospect = {'lead_origin': 'landing_page_submission',
#  'lead_source': 'direct_traffic',
#  'do_not_email': 'no',
#  'last_activity': 'other',
#  'how_did_you_hear_about_x_education': 'select',
#  'what_is_your_current_occupation': 'unemployed',
#  'what_matters_most_to_you_in_choosing_a_course': 'better_career_prospects',
#  'tags': 'other',
#  'lead_quality': 'other',
#  'lead_profile': 'other',
#  'city': 'other',
#  'asymmetrique_activity_index': '01.high',
#  'last_notable_activity': 'modified',
#  'totalvisits': 1.0,
#  'page_views_per_visit': 1.0}

# a hot lead from the test dataset
prospect = {'lead_origin': 'landing_page_submission',
 'lead_source': 'direct_traffic',
 'do_not_email': 'no',
 'last_activity': 'email_opened',
 'how_did_you_hear_about_x_education': 'select',
 'what_is_your_current_occupation': 'unemployed',
 'what_matters_most_to_you_in_choosing_a_course': 'better_career_prospects',
 'tags': 'will_revert_after_reading_the_email',
 'lead_quality': 'might_be',
 'lead_profile': 'select',
 'city': 'mumbai',
 'asymmetrique_activity_index': 'unk',
 'last_notable_activity': 'email_opened',
 'totalvisits': 3.0,
 'page_views_per_visit': 3.0}


# %%
response = requests.post(url, json = prospect).json()

print(response)

# %%
if response["lead"] == True:
    print("%s is a hot lead, sending info to sales" %prospect_id)
else:
    print("%s is not a hot lead" %prospect_id)