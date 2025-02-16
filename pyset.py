from flask import Flask, request, render_template, send_file
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def setpath(refpath, targetpath):
    reference_img = cv2.imread(refpath)  # ภาพต้นฉบับ
    target_img = cv2.imread(targetpath)  # ภาพที่ต้องการวิเคราะห์
    
    if reference_img is None or target_img is None:
        print("Error: ไม่สามารถโหลดภาพได้")
        return None
    
    # แปลงเป็น RGB
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    # ปรับขนาดภาพเป้าหมายให้เท่ากับภาพต้นฉบับ
    target_img = cv2.resize(target_img, (reference_img.shape[1], reference_img.shape[0]))
    
    # สร้างภาพที่ผ่านการแก้ไข (ตัวอย่าง: เปลี่ยนเป็นโทนสีเทา)
    processed_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    
    output_path = os.path.join('uploads', 'processed.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
    return output_path

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if file1 and file2:
            path1 = os.path.join(UPLOAD_FOLDER, file1.filename)
            path2 = os.path.join(UPLOAD_FOLDER, file2.filename)
            file1.save(path1)
            file2.save(path2)
            
            processed_path = setpath(path1, path2)
            
            return render_template('display.html', img1=file1.filename, img2=file2.filename, processed_img='processed.jpg')
    
    return '''
    <!doctype html>
    <title>Upload two images</title>
    <h1>Upload two images</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file1>
      <input type=file name=file2>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/display/<filename>')
def display_image(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
