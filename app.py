from flask import Flask, render_template, url_for, request
from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
from image_similarity_measures.quality_metrics import fsim

app = Flask(__name__)
run_with_ngrok(app)

@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("home.html", score = "00.00%", result = None, flag = None)
    elif request.method == "POST":
        original_img = np.load("img.npy")
        try:
            pred_img = cv2.imdecode(np.frombuffer(request.files["pred_img"].read(), np.uint8), cv2.IMREAD_UNCHANGED)
            pred_img = cv2.resize(pred_img, dsize = (149, 220), interpolation = cv2.INTER_NEAREST)
            score = fsim(original_img, pred_img)
        except Exception as e:
            print(e)
            score = 0
        if score >= 0.50:
            result = "good.jpg"
            flag = "REDACTED"
        else:
            result = "bad.jpg"
            flag = None
        return render_template("home.html", score = f"{score:.02%}".zfill(6), result = result, flag = flag)

if __name__ == "__main__":
    app.run()
