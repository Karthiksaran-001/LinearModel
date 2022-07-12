from flask import Flask,request,jsonify,render_template,redirect
from flask_cors import CORS,cross_origin  ## This is for deployment
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import math


app = Flask(__name__ , template_folder= "templates")



def scale(features):
    value = pickle.load(open('Scalar_Model.pickle','rb'))
    values =  value.transform(features)
    values[0][-1] = np.log(values[0][-1])
    return values

"""
def scale(features):
    value = pickle.load(open('Scalar_Model.pickle','rb'))
    return value.transform(features)
"""

def prediction(value):
    model = pickle.load(open('ElasticModel.pickle','rb'))
    return model.predict(value)



@app.route('/' , methods =['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/github' , methods =['GET'])
@cross_origin()
def github():
    return redirect('https://github.com/Karthiksaran-001/IneuronLinearRegression')

@app.route('/visual' , methods =['GET'])
@cross_origin()
def visual():
    return render_template('report.html')

@app.route('/predict' , methods = ["POST"])
@cross_origin()
def predict():
  
    if request.method == 'POST':
        zn = float(request.form['zn'])
        chas = float(request.form['chas'])
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        dis = float(request.form['dis'])
        rad = float(request.form['rad'])
        tax= float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        b = float(request.form['b'])
        lstop = float(request.form['lsotp'])
        crime = float(request.form['crime'])
        final_features = [[zn,chas , nox , rm , dis , rad , tax , ptratio ,b , lstop , crime]]
        value = scale(final_features)

        predict =math.floor(prediction(value)[0])
        calculation = f"Approximate Amount is $: {predict *1000}"
    return render_template('index.html', chance = calculation)





if __name__ == '__main__':
    app.run(debug=True)
