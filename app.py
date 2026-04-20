from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import json

app = Flask(__name__)

# 1. Load the trained model at startup
try:
    model = joblib.load("rf_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. Fingerprint generation function (Must match training exactly)
def fast_fp(smiles, n_bits=1024):
    arr = np.zeros((n_bits,), dtype=np.int8)  
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return None

# 3. Route to serve the frontend UI
@app.route('/')
def home():
    return render_template('index.html')

# 4. Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Machine learning model failed to load on the server.'}), 500

    try:
        data = request.get_json()
        smiles = data.get('drug', '')
        cell_line_str = data.get('cell_line', '')

        # Process Drug Features (SMILES to Fingerprint)
        drug_features = fast_fp(smiles)
        if drug_features is None:
            return jsonify({'error': 'Invalid SMILES string. RDKit could not parse the molecule.'}), 400

        # Process Cell Line Features (String to Numpy Array)
        # Assuming the user pastes an array like: [1.2, 3.4, 5.6]
        try:
            cell_list = json.loads(cell_line_str)
            cell_features = np.array(cell_list, dtype=np.float32)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid cell line format. Please input a valid JSON array, e.g., [1.2, 3.4]'})

        # Combine features (horizontal stack)
        X_input = np.hstack([drug_features, cell_features]).astype(np.float32)
        
        # Reshape for a single sample prediction: (1, n_features)
        X_input = X_input.reshape(1, -1)

        # Make Prediction
        prediction = model.predict(X_input)[0]

        # Return result as JSON
        return jsonify({'sensitivity': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)