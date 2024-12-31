from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained RNN model and scaler
model = tf.keras.models.load_model('rnn_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('cancer_mutation.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input values from the form and check for non-numeric input
            int_features = []
            for feature in request.form.values():
                try:
                    # Convert input values to floats
                    int_features.append(float(feature))
                except ValueError:
                    # Handle invalid non-numeric input
                    return render_template('cancer_mutation.html', pred="Error: Please enter valid numeric values for all fields.")
            
            # Print the features for debugging
            print("Features entered: ", int_features)

            # Check if the number of inputs matches the expected count (12)
            if len(int_features) != 12:
                return render_template('cancer_mutation.html', pred="Error: Please provide exactly 12 inputs.")

            # Reshape and scale the input
            final_input = np.array(int_features).reshape(1, -1)
            final_input = scaler.transform(final_input).reshape(1, 1, len(int_features))

            # Predict using the model
            prediction = model.predict(final_input)
            output = '{0:.{1}f}'.format(prediction[0][0], 2)

            # Provide feedback based on the prediction result
            if float(output) > 0.5:
                return render_template('cancer_mutation.html', pred=f"The Forest is in Danger. Probability of fire occurring is {output}", bhai="Act quickly!")
            else:
                return render_template('cancer_mutation.html', pred=f"The Forest is Safe. Probability of fire occurring is {output}", bhai="Your Forest is Safe for now")
        except Exception as e:
            # Catch and handle unexpected errors
            print(f"Unexpected error: {e}")
            return render_template('cancer_mutation.html', pred="Error: An unexpected error occurred. Please check your input values.")
    return render_template('cancer_mutation.html')

if __name__ == "__main__":
    app.run(debug=True)
