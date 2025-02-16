from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_colors(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not loaded correctly.")
        return

    # ลด noise ก่อนแปลงภาพ
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'Red': [(0, 100, 100), (10, 255, 255)],  
        'Red2': [(160, 100, 100), (180, 255, 255)],  
        'White': [(0, 0, 150), (180, 80, 255)]  # ปรับช่วงของสีขาว
    }


    color_positions = []
    combined_bbox = None

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # ลองเพิ่มการ Morphological Transform เพื่อลบ noise ออกจาก mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        matching_pixels = np.sum(mask > 0)
        print(f"Pixels matching {color}: {matching_pixels}")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # ลดเงื่อนไขลงเพื่อให้หาเจอง่ายขึ้น
                x, y, w, h = cv2.boundingRect(cnt)
                color_positions.append((color, x, y, w, h))

                if combined_bbox is None:
                    combined_bbox = (x, y, x + w, y + h)
                else:
                    combined_bbox = (
                        min(combined_bbox[0], x),
                        min(combined_bbox[1], y),
                        max(combined_bbox[2], x + w),
                        max(combined_bbox[3], y + h)
                    )

    print("Detected Color Positions:")
    for color, x, y, w, h in color_positions:
        print(f"{color}: {(x, y, w, h)}")

    if combined_bbox:
        cv2.rectangle(image, (combined_bbox[0], combined_bbox[1]), (combined_bbox[2], combined_bbox[3]), (0, 255, 0), 2)
        print(f"Combined Bounding Box: {combined_bbox}")
        cropped_img = image[combined_bbox[1]:combined_bbox[3], combined_bbox[0]:combined_bbox[2]]
        
        # Convert cropped image to RGB
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        return cropped_img_rgb  # Return cropped image for further use
    else:
        print("No valid colors detected!")
        return None

def adjust_brightness_contrast(reference_img, target_img):
    ref = reference_img.astype(np.float32) / 255.0
    target = target_img.astype(np.float32) / 255.0
    ref_avg_brightness = np.mean(ref)
    target_avg_brightness = np.mean(target)

    brightness_diff = ref_avg_brightness - target_avg_brightness
    contrast_diff = np.std(ref) - np.std(target)
    
    adjusted_target = target.copy()

    adjusted_target += brightness_diff
    adjusted_target = (adjusted_target - 0.00001) * (1 + contrast_diff) + 0.001
    adjusted_target = np.clip(adjusted_target, 0, 1)
    
    corrected_img = (adjusted_target * 255).astype(np.uint8)
    return corrected_img, brightness_diff, contrast_diff

def adjust_tone(reference_img, target_img, step=0.000001):
    reference_img = reference_img.astype(np.float32) / 255.0
    target_img = target_img.astype(np.float32) / 255.0
    ref_avg_color = np.mean(reference_img, axis=(0, 1))
    target_avg_color = np.mean(target_img, axis=(0, 1))

    color_diff = ref_avg_color - target_avg_color

    # Scale the color difference to avoid overcorrection
    scale_factor = 0.1  # Adjust this as needed
    color_diff *= scale_factor

    adjusted_target = target_img.copy()
    adjusted_target += color_diff  # Directly add the difference

    adjusted_target[:, :, 0] *= (1 + color_diff[0] * step)
    adjusted_target[:, :, 1] *= (1 + color_diff[1] * step)
    adjusted_target[:, :, 2] *= (1 + color_diff[2] * step)

    adjusted_target = np.clip(adjusted_target, 0, 1)
    return (adjusted_target * 255).astype(np.uint8), color_diff

def calculate_color_difference(reference_img, corrected_img):
    color_diff = np.abs(reference_img.astype(np.float32) - corrected_img.astype(np.float32))
    return color_diff

def apply_adjustments(target_img, brightness_diff, contrast_diff, color_diff):
    # แปลงภาพเป็น float32 เพื่อการคำนวณที่แม่นยำ
    target_img = target_img.astype(np.float32) / 255.0

    # ปรับค่าความสว่าง
    target_img += brightness_diff

    # ปรับคอนทราสต์
    target_img = (target_img - 0.5) * (1 + contrast_diff) + 0.5

    # ปรับโทนสี (R, G, B)
    target_img[:, :, 0] += color_diff[0]
    target_img[:, :, 1] += color_diff[1]
    target_img[:, :, 2] += color_diff[2]

    # คำนวณภาพสุดท้ายหลังการปรับ
    target_img = np.clip(target_img, 0, 1)

    # แปลงกลับเป็น uint8 และ return
    return (target_img * 255).astype(np.uint8)

def setpath(refpath , targetpath):
    reference_img = cv2.imread(refpath)  # ภาพต้นฉบับ
    target_img = cv2.imread(targetpath)  # ภาพที่ต้องการวิเคราะห์

    cropped_img_rgb = detect_colors(targetpath)  # Get cropped RGB image from color detection

    if reference_img is None or target_img is None or cropped_img_rgb is None:
        print("Error: ไม่สามารถโหลดภาพได้")
        return

    target_img_ts = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(cropped_img_rgb, cv2.COLOR_BGR2RGB)
    target_img_as = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    target_img = cv2.resize(target_img, (reference_img.shape[1], reference_img.shape[0]))

    adjusted_img = cv2.imread(targetpath)

    previous_avg_diff = np.inf  # Start with the highest possible difference
    tolerance = 0.01  # Acceptable difference tolerance

    brightness_changes = []
    contrast_changes = []
    color_changes = []

    total_brightness_change = 0
    total_contrast_change = 0
    total_color_change = 0

    while True:
        corrected_img, brightness_diff, contrast_diff = adjust_brightness_contrast(reference_img, target_img)
        warm_corrected_img, color_diff = adjust_tone(reference_img, corrected_img, step=0.000001)

        brightness_changes.append(brightness_diff)
        contrast_changes.append(contrast_diff)
        color_changes.append(color_diff)

        total_brightness_change += brightness_diff
        total_contrast_change += contrast_diff
        total_color_change += np.mean(np.abs(color_diff))  # Average color difference

        corrected_avg_color = np.mean(corrected_img, axis=(0, 1))
        warm_corrected_avg_color = np.mean(warm_corrected_img, axis=(0, 1))
        current_avg_diff = np.abs(np.mean(reference_img, axis=(0, 1)) - corrected_avg_color).sum()

        if current_avg_diff < tolerance:
            break

        target_img = warm_corrected_img

        if abs(previous_avg_diff - current_avg_diff) < tolerance:
            break
        previous_avg_diff = current_avg_diff

    color_diff_result = calculate_color_difference(reference_img, corrected_img)
    
    reference_img_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    corrected_img_rgb = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
    warm_corrected_img_rgb = cv2.cvtColor(warm_corrected_img, cv2.COLOR_BGR2RGB)
    color_diff_rgb = np.clip(color_diff_result, 0, 255).astype(np.uint8)

    # ใช้ apply_adjustments สำหรับการปรับภาพใหม่
    true_corrected_img = apply_adjustments(
        adjusted_img, 
        total_brightness_change, 
        total_contrast_change, 
        np.array([total_color_change] * 3)  # ใช้ค่าเฉลี่ยของ total_color_change สำหรับ RGB
    )
    
    true_corrected_img_rgb = cv2.cvtColor(true_corrected_img, cv2.COLOR_BGR2RGB)
    
    # ปรับ true_corrected_img_rgb ให้มีสีเหมือน corrected_img_rgb
    corrected_avg_color = np.mean(corrected_img_rgb, axis=(0, 1))
    true_corrected_avg_color = np.mean(true_corrected_img_rgb, axis=(0, 1))
    color_adjustment = corrected_avg_color - true_corrected_avg_color

    adjusted_true_corrected_img = true_corrected_img_rgb.astype(np.float32) + color_adjustment
    adjusted_true_corrected_img = np.clip(adjusted_true_corrected_img, 0, 255).astype(np.uint8)
    true_corrected_img_rgb = adjusted_true_corrected_img
    
    # fig, axes = plt.subplots(1, 8, figsize=(18, 5))
    # axes[0].imshow(reference_img_rgb)
    # axes[0].set_title("Reference Image")
    # axes[0].axis("off")
    
    # axes[1].imshow(target_img_ts)
    # axes[1].set_title("Original Target Image")
    # axes[1].axis("off")
    
    # axes[2].imshow(warm_corrected_img_rgb)
    # axes[2].set_title("Warm Corrected Image")
    # axes[2].axis("off")
    
    # axes[3].imshow(color_diff_rgb)
    # axes[3].set_title("Color Difference")
    # axes[3].axis("off")
    
    # axes[4].imshow(corrected_img_rgb)
    # axes[4].set_title("Corrected Image")
    # axes[4].axis("off")
    
    # axes[5].imshow(cropped_img_rgb)
    # axes[5].set_title("Cropped Image")
    # axes[5].axis("off")

    # axes[6].imshow(true_corrected_img_rgb)
    # axes[6].set_title("True Corrected Image")
    # axes[6].axis("off")

    # axes[7].plot(brightness_changes, label="Brightness Change")
    # axes[7].plot(contrast_changes, label="Contrast Change")
    # axes[7].plot([np.mean(diff) for diff in color_changes], label="Color Change")
    # axes[7].set_title("Changes Over Iterations")
    # axes[7].set_xlabel("Iterations")
    # axes[7].set_ylabel("Difference Value")
    # axes[7].legend()

    ref_avg_color = np.mean(reference_img, axis=(0, 1))
    corrected_avg_color = np.mean(corrected_img, axis=(0, 1))
    warm_corrected_avg_color = np.mean(warm_corrected_img, axis=(0, 1))

    print(f"Average color of reference image (RGB): {ref_avg_color}")
    print(f"Average color of corrected image (RGB): {corrected_avg_color}")
    print(f"Average color of warm corrected image (RGB): {warm_corrected_avg_color}")
    
    print(f"Total Brightness Adjustment: {total_brightness_change}")
    print(f"Total Contrast Adjustment: {total_contrast_change}")
    print(f"Total Color Difference: {total_color_change}")
    
    cv2.imwrite("true_corrected_img_rgb1.jpg", cv2.cvtColor(true_corrected_img_rgb, cv2.COLOR_RGB2BGR))
    # plt.tight_layout()
    # plt.show()
    return true_corrected_img_rgb

# if __name__ == "__main__":
#     setpath('wtest2.jpg' , 'wtest8.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        
        if file1 and file2:
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

            print("Saving files to:", file1_path, file2_path)  # Debugging

            file1.save(file1_path)
            file2.save(file2_path)

            ans = setpath(filename1 , filename2)
            # ans_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_image.jpg")
            # cv2.imwrite(ans_path, cv2.cvtColor(ans, cv2.COLOR_RGB2BGR))  # Convert back to BGR and save
            # ans.save(ans_path)


            return render_template('index.html', 
                       file1=url_for('static', filename=f'uploads/{filename1}'),
                       file2=url_for('static', filename=f'uploads/{filename2}'),
                       result = url_for(filename=r'C:\Users\User\CEDT Hack Med\true_corrected_img_rgb1.jpg'))
    
    return render_template('index.html', file1=None, file2=None)

if __name__ == '__main__':
    app.run(debug=True)
