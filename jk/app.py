from flask import Flask, request, render_template
import pickle
import gzip
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the compressed model
with gzip.open('credit_model_compressed.pkl.gz', 'rb') as file:
    model = pickle.load(file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make a prediction
    prediction = model.predict(final_features)
    
    # Return the prediction
    return render_template('index.html', prediction_text='Predicted Balance: ${:.2f}'.format(prediction[0]))


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)
