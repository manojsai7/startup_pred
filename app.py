from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# loading the saved model
with open('random_forest_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
all_features = model_data['all_features']


@app.route('/')
def home():
    # render the main page
    return render_template('index.html')


@app.route('/features', methods=['GET'])
def get_features():
    # return list of features the model needs
    return jsonify({'features': all_features, 'model_features': features})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # making the input array in correct order
        input_array = np.array([[data.get(f, 0) for f in all_features]])

        # scale the inputs
        input_scaled = scaler.transform(input_array)

        # only use the features the model was trained on
        feature_indices = [all_features.index(f) for f in features if f in all_features]
        input_final = input_scaled[:, feature_indices]

        # get prediction and probability
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]

        if prediction == 1:
            result = "SUCCESS"
            confidence = round(float(probability[1] * 100), 2)
            message = "Your startup looks like it's heading in the right direction. Keep pushing!"
        else:
            result = "FAILURE"
            confidence = round(float(probability[0] * 100), 2)
            message = "The odds seem tough right now. A few key changes could make a real difference."

        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'message': message
        })

    except Exception as e:
        return jsonify({
            'prediction': 'ERROR',
            'message': f'Something went wrong: {str(e)}'
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
