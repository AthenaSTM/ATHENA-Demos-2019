# Simple Demo RESTful Interface to the ATHENA classifier
#
# This is a **very** simple API to interface with the ATHENA classifier system
# from:
#
#        Riedel, M. C., Salo, T., Hays, J., Turner, M. D., Sutherland, M. T.,
#        Turner, J. A., and Laird, A. R. (2019). Automated, efficient, and
#        accelerated knowledge modeling of the cognitive neuroimaging literature
#        using the ATHENA toolkit. Front. Neurosci., 13:494.
#
# It is **NOT** useful for actual work, and it will **not** run in an online
# setting. Also note that it can be a little slow and sending a second request
# while running will cause a crash. However, it is implemented in the Python
# libraries Flask and Flask-RestPlus; therefore it can be expanded into a full
# working system and with a proxy server it should be able to be implemented in
# a way to handle multiple parallel requests.
#
# Please note that the system may be tested using the Swagger endpoint. When
# doing so, you MUST manually format the text to be analyzed to be correctly
# quoted, and also have no LF or CRLF's in the text. If you write a client to
# connect to this API, this can be easily handled with any standard JSON
# formatting library.
#
# Matthew D. Turner
# Version 1.0.0
# 2019.07.07 (rev. 5)

# Imports for the server component
from flask import Flask
from flask_restplus import reqparse, abort, Api, Resource, fields

# Imports for the machine learning component
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load file names for the classifers:
filenames = pd.read_csv("api/pickled_filenames.csv", sep = ',')

# Main application follows:
app = Flask(__name__)
api = Api(app)

# This model formats the JSON for the POST request:
upload_model = api.model("upload", {
    "classifier_type": fields.String("Type of classifier: 'abstracts' or 'full_text'."),
    "text": fields.String("Text to be classified.")
})

# Routes are defined here. There is only one route for this demo:
@api.route('/Classifier')
class PredictLabels(Resource):
    @api.expect(upload_model)
    def post(self):
        output_ = []
        file_extension = ""

        # Read the POST request's JSON:
        stuff_to_analyze = api.payload
        text_type = stuff_to_analyze["classifier_type"]
        user_query = stuff_to_analyze["text"]

        # Load each classifier in turn and run the text through it:
        if (text_type == "full_text"):
            col = 0
            file_extension = "api/Full_Text_Classifiers/"
        else:         # NB: defaults to abstracts for errors in classifier name!
            col = 1
            file_extension = "api/Abs_Classifiers/"
        for j in range(1, 87):
            filename = file_extension + filenames.iloc[j-1,col]
            label = filename.replace(".sav", "")
            with open(filename, 'rb') as f:
                clf = pickle.load(f)
                vectorizer = pickle.load(f)
            text_vectorized = vectorizer.transform(np.array([user_query]))
            prediction = clf.predict(text_vectorized)
            if prediction == 0:
                pred_text = '0'
            else:
                pred_text = '1'
            # create list of JSON objects
            output = {'label': label, 'prediction': pred_text}
            output_.append(output)

        return output_, 201

app.run(debug=True)
