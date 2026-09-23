
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Keep the pickle in the same folder as app.py when deploying to Vercel.
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Ship Mode", "Customer Name", "Segment", "Country", "City", "State",
    "Region", "Category", "Sub-Category", "Product Name",
    "Sales", "Quantity", "Discount"
]

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gradient Boosting Predictor</title>
<style>
:root{
  --bg:#07111f; --panel:rgba(15,27,48,.76); --panel2:rgba(255,255,255,.07);
  --text:#f7fbff; --muted:#9fb0c7; --accent:#7c5cff; --accent2:#22d3ee;
  --good:#4ade80; --danger:#fb7185; --border:rgba(255,255,255,.12);
  --shadow:0 24px 70px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  margin:0; min-height:100vh; font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
  color:var(--text); background:
  radial-gradient(circle at 10% 10%, color-mix(in srgb,var(--accent) 30%,transparent), transparent 28%),
  radial-gradient(circle at 90% 15%, color-mix(in srgb,var(--accent2) 25%,transparent), transparent 25%),
  linear-gradient(135deg,#050b14,#0a1424 50%,#07101d);
  overflow-x:hidden; transition:.35s ease;
}
body:before{
  content:""; position:fixed; inset:-20%; pointer-events:none; opacity:.35;
  background:conic-gradient(from 90deg at 50% 50%,transparent,var(--accent),transparent,var(--accent2),transparent);
  filter:blur(90px); animation:spin 18s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.container{width:min(1120px,92%); margin:auto; position:relative; z-index:1}
.topbar{display:flex; justify-content:space-between; align-items:center; padding:26px 0 18px}
.brand{display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:.2px}
.logo{
  width:42px;height:42px;border-radius:13px;display:grid;place-items:center;
  background:linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow:0 12px 30px color-mix(in srgb,var(--accent) 35%,transparent);
}
.logo span{font-size:20px}
.theme-wrap{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}
.theme-btn{
  width:25px;height:25px;border-radius:50%;border:2px solid rgba(255,255,255,.45);
  cursor:pointer; transition:.2s; box-shadow:0 4px 12px rgba(0,0,0,.25)
}
.theme-btn:hover{transform:scale(1.16)}
.hero{padding:30px 0 22px; max-width:800px}
.badge{display:inline-flex;gap:8px;align-items:center;padding:8px 12px;border:1px solid var(--border);
  background:var(--panel2);border-radius:999px;color:var(--muted);font-size:13px}
.badge i{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 12px var(--good)}
h1{font-size:clamp(36px,6vw,68px);line-height:.98;margin:18px 0 16px;letter-spacing:-3px}
.gradient-text{background:linear-gradient(90deg,var(--accent2),var(--accent),#fff);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{font-size:17px;color:var(--muted);line-height:1.7;max-width:680px}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:22px;padding-bottom:50px}
.card{
  background:var(--panel); border:1px solid var(--border); border-radius:26px; padding:25px;
  backdrop-filter:blur(22px); box-shadow:var(--shadow);
}
.card h2{margin:0 0 7px;font-size:21px}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px;line-height:1.5}
.form-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:15px}
.field label{display:block;color:#dce7f5;font-size:13px;font-weight:650;margin:0 0 8px}
input,select{
  width:100%;border:1px solid var(--border);outline:none;border-radius:13px;padding:13px 14px;
  background:rgba(0,0,0,.18);color:var(--text);font-size:14px;transition:.2s;
}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 14%,transparent)}
input::placeholder{color:#6f8098}
select option{background:#101b2c;color:white}
.full{grid-column:1/-1}
.predict{
  margin-top:18px;width:100%;border:0;border-radius:15px;padding:15px 18px;cursor:pointer;
  color:white;font-weight:800;font-size:15px;background:linear-gradient(100deg,var(--accent),var(--accent2));
  box-shadow:0 14px 30px color-mix(in srgb,var(--accent) 25%,transparent);transition:.2s;
}
.predict:hover{transform:translateY(-2px);filter:brightness(1.08)}
.result{
  min-height:100%;display:flex;flex-direction:column;justify-content:space-between;
}
.result-top{color:var(--muted);font-size:13px}
.result-value{font-size:clamp(42px,6vw,66px);font-weight:900;letter-spacing:-2px;margin:10px 0;
  background:linear-gradient(135deg,#fff,var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
.result-note{color:var(--muted);line-height:1.6;font-size:14px}
.stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:22px}
.stat{padding:14px;border:1px solid var(--border);border-radius:15px;background:rgba(255,255,255,.035)}
.stat b{display:block;font-size:17px}.stat span{font-size:11px;color:var(--muted)}
.info{margin-top:18px;padding:14px;border-radius:15px;background:rgba(255,255,255,.045);border:1px solid var(--border);font-size:12px;color:var(--muted);line-height:1.6}
.advanced{margin-top:22px;border-top:1px solid var(--border);padding-top:18px}
.advanced summary{cursor:pointer;color:#dce7f5;font-weight:700;font-size:13px}
.advanced .form-grid{margin-top:15px}
.footer{text-align:center;color:#71829a;font-size:12px;padding:0 0 30px}
.error{color:var(--danger);margin-top:12px;font-size:13px;display:none}
@media(max-width:820px){.grid{grid-template-columns:1fr}.hero{padding-top:15px}.topbar{align-items:flex-start}.theme-wrap{max-width:170px}.form-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
  <div class="topbar">
    <div class="brand"><div class="logo"><span>✦</span></div><div>Gradient Predictor</div></div>
    <div class="theme-wrap" aria-label="Color themes">
      <button class="theme-btn" title="Violet" style="background:#7c5cff" onclick="theme(0)"></button>
      <button class="theme-btn" title="Ocean" style="background:#06b6d4" onclick="theme(1)"></button>
      <button class="theme-btn" title="Emerald" style="background:#10b981" onclick="theme(2)"></button>
      <button class="theme-btn" title="Rose" style="background:#f43f5e" onclick="theme(3)"></button>
      <button class="theme-btn" title="Amber" style="background:#f59e0b" onclick="theme(4)"></button>
      <button class="theme-btn" title="Blue" style="background:#3b82f6" onclick="theme(5)"></button>
      <button class="theme-btn" title="Pink" style="background:#ec4899" onclick="theme(6)"></button>
      <button class="theme-btn" title="Lime" style="background:#84cc16" onclick="theme(7)"></button>
      <button class="theme-btn" title="Indigo" style="background:#6366f1" onclick="theme(8)"></button>
      <button class="theme-btn" title="Orange" style="background:#f97316" onclick="theme(9)"></button>
    </div>
  </div>

  <section class="hero">
    <div class="badge"><i></i> Gradient Boosting Regression • Live Prediction</div>
    <h1>Predict your <span class="gradient-text">business outcome.</span></h1>
    <p>Enter sales, quantity and discount values. The trained Gradient Boosting model returns an instant prediction in a clean, responsive dashboard.</p>
  </section>

  <div class="grid">
    <section class="card">
      <h2>Prediction inputs</h2>
      <div class="sub">The main model drivers are Sales and Discount. Quantity is also included.</div>
      <form id="form">
        <div class="form-grid">
          <div class="field">
            <label>Sales</label>
            <input id="sales" type="number" min="0" step="0.01" placeholder="e.g. 2500" required>
          </div>
          <div class="field">
            <label>Quantity</label>
            <input id="quantity" type="number" min="1" step="1" placeholder="e.g. 3" required>
          </div>
          <div class="field">
            <label>Discount</label>
            <input id="discount" type="number" min="0" max="1" step="0.01" placeholder="e.g. 0.20" required>
          </div>
          <div class="field">
            <label>Ship Mode code</label>
            <select id="ship_mode"><option value="0">0 — Default</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select>
          </div>
        </div>

        <details class="advanced">
          <summary>Advanced categorical inputs (optional)</summary>
          <div class="sub" style="margin-top:8px;margin-bottom:0">Your pickle contains the trained model but not the original label-encoder mappings. These fields therefore accept the numeric codes used during training. Keep 0 if you do not have those mappings.</div>
          <div class="form-grid">
            <div class="field"><label>Customer Name code</label><input id="customer" type="number" value="0" min="0"></div>
            <div class="field"><label>Segment code</label><input id="segment" type="number" value="0" min="0" max="2"></div>
            <div class="field"><label>Country code</label><input id="country" type="number" value="0" min="0"></div>
            <div class="field"><label>City code</label><input id="city" type="number" value="0" min="0"></div>
            <div class="field"><label>State code</label><input id="state" type="number" value="0" min="0"></div>
            <div class="field"><label>Region code</label><input id="region" type="number" value="0" min="0" max="3"></div>
            <div class="field"><label>Category code</label><input id="category" type="number" value="0" min="0" max="2"></div>
            <div class="field"><label>Sub-Category code</label><input id="subcategory" type="number" value="0" min="0"></div>
            <div class="field"><label>Product Name code</label><input id="product" type="number" value="0" min="0"></div>
          </div>
        </details>

        <button class="predict" type="submit">✨ Predict now</button>
        <div id="error" class="error"></div>
      </form>
    </section>

    <section class="card result">
      <div>
        <div class="result-top">MODEL PREDICTION</div>
        <div id="prediction" class="result-value">—</div>
        <div id="note" class="result-note">Enter the inputs and click <b>Predict now</b> to run your Gradient Boosting model.</div>
        <div class="stat-row">
          <div class="stat"><b id="s1">—</b><span>Sales</span></div>
          <div class="stat"><b id="s2">—</b><span>Quantity</span></div>
          <div class="stat"><b id="s3">—</b><span>Discount</span></div>
        </div>
      </div>
      <div class="info">
        <b>Deployment note</b><br>
        This UI is designed for Vercel + Flask. Keep <code>app.py</code>, <code>model.pkl</code> and <code>requirements.txt</code> in the same project folder.
      </div>
    </section>
  </div>
  <div class="footer">Built with Flask • scikit-learn • Gradient Boosting Regression</div>
</div>

<script>
const themes = [
  ["#7c5cff","#22d3ee"],["#06b6d4","#3b82f6"],["#10b981","#22c55e"],
  ["#f43f5e","#fb7185"],["#f59e0b","#f97316"],["#3b82f6","#60a5fa"],
  ["#ec4899","#a855f7"],["#84cc16","#10b981"],["#6366f1","#8b5cf6"],
  ["#f97316","#facc15"]
];
function theme(i){
  document.documentElement.style.setProperty("--accent",themes[i][0]);
  document.documentElement.style.setProperty("--accent2",themes[i][1]);
  localStorage.setItem("gradient-theme",i);
}
theme(Number(localStorage.getItem("gradient-theme") || 0));

const val = id => Number(document.getElementById(id).value || 0);
document.getElementById("form").addEventListener("submit", async (e)=>{
  e.preventDefault();
  const err=document.getElementById("error"); err.style.display="none";
  const payload={
    "Ship Mode":val("ship_mode"), "Customer Name":val("customer"),
    "Segment":val("segment"), "Country":val("country"), "City":val("city"),
    "State":val("state"), "Region":val("region"), "Category":val("category"),
    "Sub-Category":val("subcategory"), "Product Name":val("product"),
    "Sales":val("sales"), "Quantity":val("quantity"), "Discount":val("discount")
  };
  if(payload.Sales<0 || payload.Quantity<=0 || payload.Discount<0 || payload.Discount>1){
    err.textContent="Please enter valid Sales, Quantity and Discount values.";
    err.style.display="block"; return;
  }
  const btn=document.querySelector(".predict"); btn.disabled=true; btn.textContent="⏳ Predicting...";
  try{
    const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const data=await r.json();
    if(!r.ok) throw new Error(data.error || "Prediction failed");
    document.getElementById("prediction").textContent=Number(data.prediction).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
    document.getElementById("note").innerHTML="Prediction generated successfully by the deployed <b>GradientBoostingRegressor</b> model.";
    document.getElementById("s1").textContent=payload.Sales.toLocaleString();
    document.getElementById("s2").textContent=payload.Quantity;
    document.getElementById("s3").textContent=(payload.Discount*100).toFixed(0)+"%";
  }catch(ex){
    err.textContent=ex.message; err.style.display="block";
  }finally{
    btn.disabled=false; btn.textContent="✨ Predict now";
  }
});
</script>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__})

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True) or {}
        row = [[
            float(data.get("Ship Mode", 0)),
            float(data.get("Customer Name", 0)),
            float(data.get("Segment", 0)),
            float(data.get("Country", 0)),
            float(data.get("City", 0)),
            float(data.get("State", 0)),
            float(data.get("Region", 0)),
            float(data.get("Category", 0)),
            float(data.get("Sub-Category", 0)),
            float(data.get("Product Name", 0)),
            float(data.get("Sales", 0)),
            float(data.get("Quantity", 0)),
            float(data.get("Discount", 0)),
        ]]
        prediction = float(model.predict(np.asarray(row))[0])
        return jsonify({"prediction": prediction})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

# Vercel imports "app" automatically.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
