from flask import Flask
from flask_restplus import reqparse, abort, Api, Resource
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

filenames = pd.read_csv("pickled_filenames.csv", sep = ',')
app = Flask(__name__)
api = Api(app)
ns = api.namespace('API', description='classifier operations')
parser = reqparse.RequestParser()
parser.add_argument('classifier_type', location='form', dest = "classifier_type" , required=True, choices = ('full text', 'abstracts'), help = "Full Text or Abstract")
parser.add_argument('query', location='form', dest = "query" , required=True, type= str, help = "Please")


@ns.route('/predict')
class Predict(Resource):
    @ns.expect(parser)
    def post(self):
        """
        Produces list of labels the classifier recommends
        """
        output_ = []
        args = parser.parse_args()
        text_type = args['classifier_type']
        user_query = args['query']
        file_extension = ""

        if (text_type == "full text"):
            col = 0
            file_extension = "Full_Text_Classifiers/"
        else:
            col = 1
            file_extension = "Abs_Classifiers/"
        for j in range(1, 87):
            # Unloading pickled classifier model and vectorizer
            filename = file_extension + filenames.iloc[j-1,col]
            label = filename.replace(".sav", "")
            with open(filename, 'rb') as f:
                clf = pickle.load(f)
                vectorizer = pickle.load(f)
            uq_vectorized = vectorizer.transform(np.array([user_query]))
            prediction = clf.predict(uq_vectorized)
            pred_proba = clf.predict_proba(uq_vectorized)

            if prediction == 0:
                pred_text = '0'
            else:
                pred_text = '1'

            # create JSON object
            output = {'label': label, 'prediction': pred_text}
            output_.append(output)

        return output_
app.run(debug = True)
