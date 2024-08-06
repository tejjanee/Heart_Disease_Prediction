from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            sex=int(request.form.get('sex')),
            cp=int(request.form.get('cp')),
            trestbps=int(request.form.get('trestbps')),
            chol=int(request.form.get('chol')),
            fbs=int(request.form.get('fbs')),
            restecg=int(request.form.get('restecg')),
            thalach=int(request.form.get('thalach')),
            exang=int(request.form.get('exang')),
            oldpeak=float(request.form.get('oldpeak')),
            slope=int(request.form.get('slope')),
            ca=int(request.form.get('ca')),
            thal=int(request.form.get('thal'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        if results >= 0.5:
            results = "The person has heart disease."
        else:
            results = "The person does not have heart disease."
        return render_template('home.html', result=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
