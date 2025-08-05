from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('randomforest_model.pkl')
except Exception as e:
    print("خطأ في تحميل النموذج:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
     
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        parking = int(request.form['parking'])
        mainroad = int(request.form['mainroad'])
        guestroom = int(request.form['guestroom'])
        basement = int(request.form['basement'])
        hotwaterheating = int(request.form['hotwaterheating'])
        airconditioning = int(request.form['airconditioning'])
        prefarea = int(request.form['prefarea'])
        furnishingstatus = int(request.form['furnishingstatus'])

       
        input_data = np.array([[area, bedrooms, bathrooms, stories,
                                mainroad, guestroom, basement,
                                hotwaterheating, airconditioning,
                                parking, prefarea, furnishingstatus]])

        prediction = model.predict(input_data)[0]

        result = f"السعر المتوقع: {prediction:,.2f} ألف"
        return render_template('index.html', result=result)

    except Exception as e:
        error = f"حدث خطأ: {str(e)}"
        return render_template('index.html', result=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)


