from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# طباعة معلومات مسار العمل والملفات للتأكد
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

# تحديد مسار النموذج بشكل صريح بناءً على مكان هذا الملف
model_path = os.path.join(os.path.dirname(__file__), 'randomforest_model.pkl')

try:
    model = joblib.load(model_path)
    print("تم تحميل النموذج بنجاح من:", model_path)
except Exception as e:
    print("خطأ في تحميل النموذج:", e)
    model = None  # لتفادي الخطأ في حالة عدم التحميل

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        error = "النموذج غير محمل، يرجى التأكد من وجود ملف النموذج."
        return render_template('index.html', result=error)

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
