#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import os
warnings.filterwarnings('ignore')
from feature import FeatureExtraction  # Ensure this file exists and is correct

file = open("pickle/model.pkl", "rb")  # Check if this path is correct
gbc = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)  # Ensure getFeaturesList() returns a list of length 30

        y_pred = gbc.predict(x)[0]
        # 1 is safe       
        # -1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0, 0]  # Probability of phishing
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]  # Probability of non-phishing
        
        # Correcting prediction logic based on your model's output
        pred = "It is {0:.2f} % safe to go".format(y_pro_phishing * 100)  
        
        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)

    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000))
