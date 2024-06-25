from flask import Flask, render_template, request
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
model = YOLO('last.pt')
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        if imagefile:
            

            # Perform inference on the uploaded image
            results = model('')
            names_dict = results[0].names
            probs = results[0].probs.data.tolist()
            prediction = names_dict[np.argmax(probs)]

            return render_template('results.html', prediction=prediction)

    return render_template('indexx.html')

if __name__ == '__main__':
    app.run(port=5500, debug=True)
