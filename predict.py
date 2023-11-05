# %% IMPORTS
import pickle

from flask import Flask
from flask import request
from flask import jsonify

# %% LOAD MODEL
model_file = 'model.bin'

with open(model_file, 'rb') as f_in: # now we read the file, its important to avoid to overwrite the file creating one with zero bytes

    (dv, model) = pickle.load(f_in)


# %% PREDICT FUNCTION & APP

app = Flask('lead') # create a flask app

@app.route('/predict', methods = ['POST']) 


def predict():
    prospect = request.get_json()

    ### This should be inside a separate function ideally 
    X = dv.transform([prospect]) # remember that DictVectorizer expects a list
    y_pred = model.predict(X)

    ####
    result = {
        'lead': bool(y_pred)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host ='localhost', port=9696) 