from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load("prediction_model.joblib")
encoder = joblib.load("label_encoder.joblib")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    color = ""
    if request.method == 'POST':
        try:
            inputs = [
                request.form['n1'].strip().upper(),
                request.form['n2'].strip().upper(),
                request.form['n3'].strip().upper(),
                request.form['n4'].strip().upper(),
            ]
            encoded_inputs = encoder.transform(inputs).reshape(1, -1)
            pred_num = model.predict(encoded_inputs)[0]
            prediction = encoder.inverse_transform([pred_num])[0]
            color = "green" if prediction == "BIG" else "red"
        except:
            prediction = "ত্রুটি! শুধু BIG বা SMALL লিখুন।"
            color = "black"
    return render_template('index.html', prediction=prediction, color=color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)