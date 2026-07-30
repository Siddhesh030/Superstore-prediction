import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# UPDATED: Matches your unique file signature name
MODEL_PATH = "Gradiantmodel.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Professional interactive interface featuring glassmorphism and keyframe animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Business Analytics & Regression Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: radial-gradient(circle at 0% 0%, #0c1020 0%, #05070f 100%);
            color: #f1f5f9;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2.5rem 1rem;
            overflow-x: hidden;
            position: relative;
        }

        /* Ambient floating dynamic glowing backgrounds */
        .glow-orb {
            position: absolute;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            filter: blur(120px);
            z-index: 1;
            opacity: 0.22;
            animation: pulseGlow 10s ease-in-out infinite alternate;
        }
        .orb-1 { width: 400px; height: 400px; top: -100px; right: -50px; }
        .orb-2 { width: 350px; height: 350px; bottom: -50px; left: -100px; animation-delay: 3s; }

        @keyframes pulseGlow {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.15) translate(20px, 30px); filter: blur(140px); }
        }

        .container {
            background: rgba(13, 18, 36, 0.45);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 28px;
            padding: 3rem;
            width: 100%;
            max-width: 900px;
            position: relative;
            z-index: 10;
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            animation: panelFadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes panelFadeIn {
            from { opacity: 0; transform: translateY(30px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        h2 {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(120deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        .section-title {
            grid-column: 1 / -1;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #3b82f6;
            margin-top: 0.5rem;
            border-bottom: 1px solid rgba(59, 130, 246, 0.2);
            padding-bottom: 0.3rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: 1fr;
            grid-gap: 1.5rem;
        }

        @media (min-width: 640px) {
            .grid-form { grid-template-columns: repeat(2, 1fr); }
        }
        @media (min-width: 850px) {
            .grid-form { grid-template-columns: repeat(3, 1fr); }
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.8rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: #94a3b8;
            transition: color 0.2s;
        }

        input {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 0.8rem 1rem;
            border-radius: 12px;
            color: #fff;
            font-size: 0.95rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            outline: none;
        }

        input:focus {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.05);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
        }

        .btn-container {
            grid-column: 1 / -1;
            text-align: center;
            margin-top: 1.5rem;
        }

        button {
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            border: none;
            padding: 1rem 3.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(37, 99, 235, 0.4);
            filter: brightness(1.1);
        }

        /* Modern Result Modal / Card Animation */
        .result-card {
            grid-column: 1 / -1;
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            display: none;
            opacity: 0;
            transform: translateY(15px);
            transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .result-card.reveal {
            display: block;
            opacity: 1;
            transform: translateY(0);
        }

        .success-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .success-box .val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #34d399;
            margin-top: 0.2rem;
            text-shadow: 0 0 20px rgba(52, 211, 153, 0.2);
        }

        .error-box {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
        }
    </style>
</head>
<body>
    <div class="glow-orb orb-1"></div>
    <div class="glow-orb orb-2"></div>

    <div class="container">
        <header>
            <h2>Gradient Boosting Regressor</h2>
            <p class="subtitle">Enter corporate operational vectors for regression estimation</p>
        </header>

        <form id="regressionForm" class="grid-form">
            
            <div class="section-title">Categorical Labels (Encoded Numbers)</div>
            <div class="input-group">
                <label for="ship_mode">Ship Mode</label>
                <input type="number" id="ship_mode" name="Ship Mode" placeholder="e.g. 0, 1, 2" required>
            </div>
            <div class="input-group">
                <label for="customer_name">Customer Name ID</label>
                <input type="number" id="customer_name" name="Customer Name" placeholder="Numeric Index ID" required>
            </div>
            <div class="input-group">
                <label for="segment">Segment</label>
                <input type="number" id="segment" name="Segment" placeholder="Segment Key" required>
            </div>
            <div class="input-group">
                <label for="country">Country</label>
                <input type="number" id="country" name="Country" placeholder="Country Key" required>
            </div>
            <div class="input-group">
                <label for="city">City</label>
                <input type="number" id="city" name="City" placeholder="City Key" required>
            </div>
            <div class="input-group">
                <label for="state">State</label>
                <input type="number" id="state" name="State" placeholder="State Key" required>
            </div>
            <div class="input-group">
                <label for="region">Region</label>
                <input type="number" id="region" name="Region" placeholder="Region Key" required>
            </div>
            <div class="input-group">
                <label for="category">Category</label>
                <input type="number" id="category" name="Category" placeholder="Category Key" required>
            </div>
            <div class="input-group">
                <label for="sub_category">Sub-Category</label>
                <input type="number" id="sub_category" name="Sub-Category" placeholder="Sub-Category Key" required>
            </div>
            <div class="input-group full-width">
                <label for="product_name">Product Name ID</label>
                <input type="number" id="product_name" name="Product Name" placeholder="Product Numeric ID" required>
            </div>

            <div class="section-title">Quantitative Metrics</div>
            <div class="input-group">
                <label for="sales">Sales Value ($)</label>
                <input type="number" id="sales" name="Sales" step="0.01" placeholder="e.g. 245.50" required>
            </div>
            <div class="input-group">
                <label for="quantity">Quantity</label>
                <input type="number" id="quantity" name="Quantity" placeholder="e.g. 3" required>
            </div>
            <div class="input-group">
                <label for="discount">Discount Ratio</label>
                <input type="number" id="discount" name="Discount" step="0.01" placeholder="e.g. 0.2" required>
            </div>

            <div class="btn-container">
                <button type="submit" id="calcBtn">Execute Estimation</button>
            </div>

            <div id="resultCard" class="result-card"></div>
        </form>
    </div>

    <script>
        document.getElementById('regressionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const card = document.getElementById('resultCard');
            const btn = document.getElementById('calcBtn');
            
            btn.innerText = 'Analyzing Variables...';
            card.className = 'result-card'; 
            card.style.display = 'none';

            const formData = new FormData(this);
            const orderedFeatures = [
                "Ship Mode", "Customer Name", "Segment", "Country", "City", 
                "State", "Region", "Category", "Sub-Category", "Product Name", 
                "Sales", "Quantity", "Discount"
            ];
            
            const featureArray = orderedFeatures.map(name => parseFloat(formData.get(name)));

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: featureArray })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    card.innerHTML = `<div class="label">Estimated Regression Output</div><div class="val">${data.prediction.toFixed(4)}</div>`;
                    card.classList.add('reveal', 'success-box');
                } else {
                    card.innerHTML = `<div><strong>Execution Interrupted:</strong> ${data.error}</div>`;
                    card.classList.add('reveal', 'error-box');
                }
            } catch (err) {
                card.innerHTML = `<div>Failed to map connection to backend array logic.</div>`;
                card.classList.add('reveal', 'error-box');
            } finally {
                card.style.display = 'block';
                btn.innerText = 'Execute Estimation';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": f"The target predictive engine file '{MODEL_PATH}' was not verified inside the directory location."}), 500
    
    try:
        data = request.json
        input_data = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
