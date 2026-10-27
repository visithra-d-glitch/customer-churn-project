from flask import Flask, request, render_template_string, send_file
import pandas as pd
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
meta = joblib.load("meta.pkl")
defaults, options = meta["defaults"], meta["options"]

CAT = [c for c in ["Contract", "Internet Type", "Payment Method", "Offer"] if c in options]
NUM = [c for c in ["Age", "Tenure in Months", "Monthly Charge", "Number of Referrals"] if c in defaults]

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Predictor</title>
<style>
 body{margin:0;font-family:Arial,sans-serif;background:linear-gradient(135deg,#4f46e5,#06b6d4);min-height:100vh;padding:24px 12px;box-sizing:border-box}
 .card{max-width:520px;margin:0 auto;background:#fff;border-radius:16px;padding:24px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
 h2{margin:0 0 4px;color:#4f46e5}
 .sub{color:#6b7280;margin:0 0 18px;font-size:14px}
 label{font-weight:bold;color:#374151;font-size:14px}
 input,select{width:100%;padding:9px;margin:4px 0 14px;border:2px solid #e5e7eb;border-radius:8px;font-size:15px;box-sizing:border-box}
 input:focus,select:focus{border-color:#4f46e5;outline:none}
 button{width:100%;padding:11px;border:none;border-radius:8px;background:#4f46e5;color:#fff;font-size:16px;font-weight:bold;cursor:pointer}
 button:hover{background:#4338ca}
 .dl{background:#0891b2;margin-top:10px}
 .dl:hover{background:#0e7490}
 .res{margin-top:18px;padding:14px;border-radius:10px;text-align:center;font-size:20px;font-weight:bold}
 .red{background:#fee2e2;color:#b91c1c;border:2px solid #ef4444}
 .green{background:#dcfce7;color:#15803d;border:2px solid #22c55e}
</style>
</head>
<body>
<div class="card">
<h2>Customer Churn Predictor</h2>
<p class="sub">Customer details kudunga, churn risk paarunga</p>
<form method="post" action="/">
{% for c in NUM %}
 <label>{{ c }}</label>
 <input type="number" step="any" name="{{ c }}" value="{{ vals.get(c, defaults[c]) }}" required>
{% endfor %}
{% for c in CAT %}
 <label>{{ c }}</label>
 <select name="{{ c }}">
 {% for o in options[c] %}
   <option {% if vals.get(c, defaults[c]) == o %}selected{% endif %}>{{ o }}</option>
 {% endfor %}
 </select>
{% endfor %}
<button type="submit">Predict</button>
</form>
{% if result %}
<div class="res {{ color }}">{{ result }}</div>
<form method="post" action="/download">
 {% for k, v in vals.items() %}
  <input type="hidden" name="{{ k }}" value="{{ v }}">
 {% endfor %}
 <input type="hidden" name="result" value="{{ result }}">
 <button class="dl">Download PDF</button>
</form>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result, color, vals = None, "red", {}
    if request.method == "POST":
        row = dict(defaults)
        for c in NUM:
            row[c] = float(request.form[c])
        for c in CAT:
            row[c] = request.form[c]
        vals = {**{c: float(request.form[c]) for c in NUM},
                **{c: request.form[c] for c in CAT}}
        X = pd.get_dummies(pd.DataFrame([row]))
        X = X.reindex(columns=columns, fill_value=0).astype(float)
        prob = model.predict_proba(X)[0][1] * 100
        if prob >= 50:
            result, color = f"High risk of churn ({prob:.0f}%)", "red"
        else:
            result, color = f"Low risk of churn ({prob:.0f}%)", "green"
    return render_template_string(PAGE, CAT=CAT, NUM=NUM, options=options,
                                  defaults=defaults, vals=vals,
                                  result=result, color=color)

@app.route("/download", methods=["POST"])
def download():
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFillColorRGB(0.31, 0.27, 0.90)
    c.rect(0, h - 90, w, 90, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, h - 55, "Customer Churn Prediction Report")
    y = h - 130
    c.setFont("Helvetica", 13)
    for k, v in request.form.items():
        if k != "result":
            c.setFillColorRGB(0.35, 0.35, 0.35)
            c.drawString(50, y, f"{k}:")
            c.setFillColorRGB(0, 0, 0)
            c.drawString(230, y, str(v))
            y -= 24
    result = request.form["result"]
    high = "High" in result
    if high:
        c.setFillColorRGB(0.99, 0.89, 0.89)
    else:
        c.setFillColorRGB(0.86, 0.99, 0.91)
    c.rect(40, y - 55, w - 80, 45, fill=1, stroke=0)
    if high:
        c.setFillColorRGB(0.72, 0.11, 0.11)
    else:
        c.setFillColorRGB(0.08, 0.50, 0.24)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(55, y - 38, result)
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="churn_report.pdf",
                     mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)