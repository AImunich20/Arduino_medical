import base64
import os
from io import BytesIO
from flask import Flask, request, render_template, send_file, url_for
import cv2
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def detect_colors(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not loaded correctly.")
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return None

def adjust_brightness_contrast(reference_img, target_img):
    ref = reference_img.astype(np.float32) / 255.0
    target = target_img.astype(np.float32) / 255.0
    ref_avg = np.mean(ref)
    target_avg = np.mean(target)
    
    brightness_diff = ref_avg - target_avg
    target += brightness_diff
    target = np.clip(target, 0, 1)
    return (target * 255).astype(np.uint8), brightness_diff, 0

def adjust_tone(reference_img, target_img, step=0.000001):
    ref_avg_color = np.mean(reference_img, axis=(0, 1))
    target_avg_color = np.mean(target_img, axis=(0, 1))
    color_diff = ref_avg_color - target_avg_color
    target_img = target_img.astype(np.float32) / 255.0
    target_img += color_diff * step
    target_img = np.clip(target_img, 0, 1)
    return (target_img * 255).astype(np.uint8), color_diff

def setpath(refpath, tarpath):
    reference_img = cv2.imread(refpath)
    target_img = cv2.imread(tarpath)
    cropped_img_rgb = detect_colors(tarpath)

    if reference_img is None or target_img is None or cropped_img_rgb is None:
        return None, "Error loading images"

    target_img = cv2.resize(target_img, (reference_img.shape[1], reference_img.shape[0]))
    brightness_changes, contrast_changes, color_changes = [], [], []
    
    while True:
        corrected_img, brightness_diff, contrast_diff = adjust_brightness_contrast(reference_img, target_img)
        warm_corrected_img, color_diff = adjust_tone(reference_img, corrected_img, step=0.000001)
        brightness_changes.append(brightness_diff)
        contrast_changes.append(contrast_diff)
        color_changes.append(np.mean(np.abs(color_diff)))
        
        if np.abs(np.mean(reference_img) - np.mean(corrected_img)) < 0.01:
            break
        target_img = warm_corrected_img
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Corrected Image")
    axes[0].axis("off")
    
    axes[1].plot(brightness_changes, label="Brightness")
    axes[1].plot(contrast_changes, label="Contrast")
    axes[1].plot(color_changes, label="Color Change")
    axes[1].set_title("Adjustment Changes")
    axes[1].legend()
    
    plt.tight_layout()
    graph_path = os.path.join(app.config['RESULT_FOLDER'], 'graph.png')
    plt.savefig(graph_path)
    plt.close()
    
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'corrected_image.jpg')
    cv2.imwrite(result_path, corrected_img)
    
    return result_path, graph_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        reference = request.files['reference']
        if file.filename == '' or reference.filename == '':
            return 'No selected file'
        filename = secure_filename(file.filename)
        reference_filename = secure_filename(reference.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        reference_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
        file.save(file_path)
        reference.save(reference_path)
        result_path, graph_path = setpath(reference_path, file_path)
        return render_template('result.html', result_image=url_for('static', filename='results/corrected_image.jpg'), graph_image=url_for('static', filename='results/graph.png'))
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
