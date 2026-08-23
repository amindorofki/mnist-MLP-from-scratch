"Flask demo: draw a digit on an HTML canvas, get a live prediction"

import base64
import io
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from PIL import Image

from predict import load_trained_model, predict

app = Flask(__name__)
MODEL = load_trained_model("mnist_mlp_weights.npz")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>MNIST Digit Classifier</title></head>
<body style="font-family: sans-serif; text-align: center;">
    <h2>Draw a digit (0-9)</h2>
    <canvas id="canvas" width="280" height="280" style="border:2px solid black; background:black;"></canvas>
    <br><br>
    <button onclick="clearCanvas()">Clear</button>
    <button onclick="predict()">Predict</button>
    <h3 id="result"></h3>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        let drawing = false;

        canvas.addEventListener('mousedown', () => drawing = true);
        canvas.addEventListener('mouseup', () => { drawing = false; ctx.beginPath(); });
        canvas.addEventListener('mousemove', draw);

        function draw(e) {
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').innerText = '';
        }
        clearCanvas();

        function predict() {
            const dataURL = canvas.toDataURL('image/png');
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: dataURL})
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('result').innerText =
                    'Prediction: ' + data.digit + '  (' + data.confidence.toFixed(1) + '%)';
            });
        }
    </script>
</body>
</html>
"""


def center_digit(arr, size=28, padding=4):
    "crops to the digit's bounding box, then centers it on a size x size canvas — mimics MNIST preprocessing"
    rows = np.any(arr > 0.05, axis=1)
    cols = np.any(arr > 0.05, axis=0)

    if not rows.any() or not cols.any():
        return arr  # blank canvas, nothing to center

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    digit = arr[rmin:rmax + 1, cmin:cmax + 1]

    # resize the cropped digit to fit inside (size - 2*padding), preserving aspect ratio
    h, w = digit.shape
    target = size - 2 * padding
    scale = target / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    digit_img = Image.fromarray((digit * 255).astype("uint8"))
    digit_img = digit_img.resize((new_w, new_h))
    digit_resized = np.array(digit_img).astype("float32") / 255.0

    canvas = np.zeros((size, size), dtype="float32")
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit_resized

    return canvas


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict_route():
    data_url = request.json["image"]
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize((28, 28))

    arr = np.array(img).astype("float32") / 255.0
    arr = center_digit(arr)

    X = arr.reshape(1, 784)

    preds, confidence, probs = predict(X, MODEL)

    return jsonify(digit=int(preds[0]), confidence=float(confidence[0]))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)