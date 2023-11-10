from flask import Flask, request, render_template
from prediction import predictDisease  

app = Flask(__name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    symptoms = request.form['symptoms']
    predictions = predictDisease(symptoms)
    return render_template('result.html', symptoms=symptoms, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
