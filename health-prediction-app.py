from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model (make sure the path is correct)
model = pickle.load(open('Healt_insurace_charge_model.sav', 'rb'))

def minmax_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        smoker = int(request.form['smoker'] == 'yes')

        # Scale the inputs using MinMax scaling
        age_scaled = minmax_scale(age, 18, 100)
        bmi_scaled = minmax_scale(bmi, 15, 40)

        features = np.array([[age_scaled, bmi_scaled, smoker, age_scaled * bmi_scaled, age_scaled * smoker, bmi_scaled * smoker]])
        prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
