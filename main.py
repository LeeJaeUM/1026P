from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from YoloV4 import YoloV4
import cv2
import numpy as np
import collections

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_top_colors(image, num_colors=3):
    # 이미지에서 주요 색상을 추출
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = num_colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    top_colors = [(tuple(palette[i]), count) for i, count in enumerate(counts)]
    top_colors = sorted(top_colors, key=lambda x: -x[1])
    return top_colors


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            yolo = YoloV4()
            img = cv2.imread(file_path)
            detected_img = yolo.detect(img)
            result_filename = os.path.splitext(filename)[0] + '_detected' + os.path.splitext(filename)[1]
            result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_file_path, detected_img)
            
            result_img_url = url_for('send_file', filename=result_filename)

            image_for_color = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            top_colors = get_top_colors(image_for_color)

            return render_template('index.html', result_img=result_img_url, top_colors=top_colors)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
