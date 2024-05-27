import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.perdiction_pipeline import CustomData , PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Hour=float(request.form.get('Hour')),
            Temperature = float(request.form.get('Temperature')),
            Humidity = float(request.form.get('Humidity')),
            Windspeed = float(request.form.get('Windspeed')),
            Visibility = float(request.form.get('Visibility')),
            Seasons = request.form.get('Seasons'),
            Holiday = request.form.get('Holiday'),
            FunctioningDay = request.form.get('FunctioningDay'),
            Day= request.form.get('Day'),
            Month = request.form.get('Month'),
            Year= request.form.get('Year')
        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictionPipeline()
        pred=predict_pipeline.predict(final_new_data)
        

        return render_template('index.html',final_result=int(pred))
        


if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080)
