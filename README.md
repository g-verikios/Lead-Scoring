# Leads Scoring

## Leads Dataset
A lead in marketing terms is: "Simply defined, leads in marketing refer to any individual or organization within your marketing reach who has interacted with your brand way or has the potential to become a future customer. A lead can be someone that sees or downloads your content, signs up for a trial, or visits your store."  
The dataset basically includes various user data (e.g. contact preferences, source of the lead, time spent on website etc.) that can be used to assess whether a lead can be converted to a sale.

The aim is to perform lead scoring, in other words determine if a lead is worth passing from the marketing team on to sales based on the available data. As seen from the image we start from an initial pool of leads which then based on the features describing it can be a hot lead which will become a converted lead or in other words someone who will make a purchase.


![image.jpg](attachment:image.jpg)

Different models will be compared and the one with the best performance will be selected to be used in a lead prediction service so that the sales team knows-based on the data from marketing-which users can be potential customers. 


A description of the variables is in Leads Data Dictionary.xlsx

References:
- https://www.salesforce.com/products/guide/lead-gen/scoring-and-grading/
- https://www.wrike.com/marketing-guide/faq/what-is-a-lead-in-marketing/

## Environent setup

## Contents

## Exploratory Data Analysis
Insigths were found through an EDA on the dataset, in the notebook. In addition, 3 different models are compared to choose the one with the best performance.

## How to use
- Notebook with EDA
After setting up the environemt using pipenv you can open the notebook with: jupyter notebook notebook.ipynb
- Model training
Simply open a terminal in this folder and type: pipenv run python train.py
- Running the service (local)
1. Build the docker image using docker build -t lead-scoring .
2. Run the docker image using docker run -it --rm -p 9696:9696 lead-scoring
3. In a terminal with the pipenv environment activated type and run: python predict_test.py

Note: To stop serving use Crtl+C

## Things that could be done better
- The part of handling the features, for e.g. which ones to drop which ones to keep could be set up perhaps in a better way so that it's customizable in a simpler way. Now the column names are hardcoded and this is probably not the cleanest option if a later EDA on newer datasets brings other insights in terms of feature importance etc.
- Generally wanted to do something more exotic other than the boilerplate used in the course for the scripts deploying the model, but in this case time was really limited, perhaps something to change in the future.