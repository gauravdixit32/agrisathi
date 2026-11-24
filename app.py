from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db_connection
from functools import wraps
import os
from datetime import timedelta
from dotenv import load_dotenv
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

load_dotenv()

# Load model once at startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model_path = os.path.join(os.getcwd(), "leaf_disease_unet_best.pth")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

IMG_SIZE = 256
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

def predict_disease_mask(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = transform(image=image)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8)
    severity = mask.mean() * 100.0
    return mask, float(severity)
# From Colab training: class_names = ['Apple___Apple_scab', 'Apple___Black_rot', ...]
# Add this list manually or load from a file if you saved it
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___healthy"
]

# Load classifier
classifier = models.efficientnet_b0(pretrained=False)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, len(class_names))
classifier_path = os.path.join(os.getcwd(), "leaf_disease_classifier_best.pth")
classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()

classifier_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def predict_disease_name(image_path):
    image = Image.open(image_path).convert("RGB")
    x = classifier_tfms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, dim=1)
        disease_name = class_names[idx.item()]
        return disease_name, confidence.item()


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")
app.permanent_session_lifetime = timedelta(days=1)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        mobile = request.form.get('mobile')
        password = request.form.get('password')

        if not mobile or not password:
            flash('Please fill in all fields', 'danger')
            return redirect(url_for('login'))

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            cursor.execute("SELECT * FROM users WHERE mobile = %s", (mobile,))
            user = cursor.fetchone()

            if user and check_password_hash(user['password_hash'], password):
                session.permanent = True
                session['user_id'] = user['user_id']
                session['user_name'] = user['name']
                session['role'] = user['role']

                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid mobile number or password', 'danger')

        except Exception as e:
            print("Login Error:", e)
            flash('An error occurred. Please try again.', 'danger')

        finally:
            cursor.close()
            conn.close()

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        name = request.form.get('name')
        mobile = request.form.get('mobile')
        password = request.form.get('password')

        if not all([name, mobile, password]):
            flash('Please fill in all fields', 'danger')
            return redirect(url_for('register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO users (name, mobile, password_hash, role)
                VALUES (%s, %s, %s, %s)
            """, (name, mobile, hashed_password, 'farmer'))

            conn.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            conn.rollback()
            print("Registration Error:", e)
            flash('Mobile already registered or database error.', 'danger')
            return redirect(url_for('register'))

        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')



@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=session.get('user_name'))




# Protected routes
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('home'))

    user_id = session.get('user_id')
    cursor = conn.cursor(dictionary=True)

    try:
        if request.method == 'POST':
            age = request.form.get('age')
            land_size = request.form.get('land_size')
            land_unit = request.form.get('land_unit')
            state = request.form.get('state')
            district = request.form.get('district')
            village = request.form.get('village')
            language = request.form.get('language')

            cursor.execute("SELECT profile_id FROM farmer_profile WHERE user_id = %s", (user_id,))
            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    """
                    UPDATE farmer_profile
                    SET age=%s, land_size=%s, land_unit=%s, state=%s, district=%s, village=%s, language=%s
                    WHERE user_id=%s
                    """,
                    (age, land_size, land_unit, state, district, village, language, user_id),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO farmer_profile (user_id, age, land_size, land_unit, state, district, village, language)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, age, land_size, land_unit, state, district, village, language),
                )

            conn.commit()
            flash('Profile saved successfully.', 'success')

        cursor.execute("SELECT * FROM farmer_profile WHERE user_id = %s", (user_id,))
        profile_data = cursor.fetchone()

    except Exception as e:
        conn.rollback()
        print("Profile Error:", e)
        flash('An error occurred. Please try again.', 'danger')
        profile_data = None

    finally:
        cursor.close()
        conn.close()

    return render_template('profile.html', profile=profile_data)

@app.route('/crop-register', methods=['GET', 'POST'])
@login_required
def crop_register():
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('home'))

    user_id = session.get('user_id')
    cursor = conn.cursor(dictionary=True)

    try:
        if request.method == 'POST':
            crop_id = request.form.get('crop_id')
            sowing_date = request.form.get('sowing_date')

            if not crop_id:
                flash('Please select a crop.', 'danger')
            else:
                cursor.execute(
                    "INSERT INTO farmer_crops (user_id, crop_id, sowing_date) VALUES (%s, %s, %s)",
                    (user_id, crop_id, sowing_date),
                )
                conn.commit()
                flash('Crop registered successfully.', 'success')

        cursor.execute("SELECT crop_id, crop_name, season FROM crops ORDER BY crop_name")
        crops_list = cursor.fetchall()

        cursor.execute(
            """
            SELECT fc.id, c.crop_name, c.season, fc.sowing_date
            FROM farmer_crops fc
            JOIN crops c ON c.crop_id = fc.crop_id
            WHERE fc.user_id = %s
            ORDER BY fc.id DESC
            """,
            (user_id,),
        )
        farmer_crops_list = cursor.fetchall()

    except Exception as e:
        conn.rollback()
        print("Crop Register Error:", e)
        flash('An error occurred. Please try again.', 'danger')
        crops_list = []
        farmer_crops_list = []

    finally:
        cursor.close()
        conn.close()

    return render_template('crop_register.html', crops=crops_list, farmer_crops=farmer_crops_list)

@app.route('/farm-location', methods=['GET', 'POST'])
@login_required
def farm_location():
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('home'))

    user_id = session.get('user_id')
    cursor = conn.cursor(dictionary=True)

    try:
        if request.method == 'POST':
            state = request.form.get('state')
            district = request.form.get('district')
            village = request.form.get('village')

            cursor.execute("SELECT profile_id FROM farmer_profile WHERE user_id = %s", (user_id,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    "UPDATE farmer_profile SET state=%s, district=%s, village=%s WHERE user_id=%s",
                    (state, district, village, user_id),
                )
            else:
                cursor.execute(
                    "INSERT INTO farmer_profile (user_id, state, district, village) VALUES (%s, %s, %s, %s)",
                    (user_id, state, district, village),
                )

            conn.commit()
            flash('Location saved successfully.', 'success')

        cursor.execute("SELECT state, district, village FROM farmer_profile WHERE user_id = %s", (user_id,))
        location_data = cursor.fetchone()

    except Exception as e:
        conn.rollback()
        print("Farm Location Error:", e)
        flash('An error occurred. Please try again.', 'danger')
        location_data = None

    finally:
        cursor.close()
        conn.close()

    return render_template('farm_location.html', location=location_data)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file:
            # Save uploaded image
            filename = f"{session['user_id']}_{file.filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Run segmentation model
            mask, severity = predict_disease_mask(image_path)

            # Run classifier on the original image
            disease_name, disease_conf = predict_disease_name(image_path)

            # Save mask image (optional, for UI)
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"mask_{filename}.png")
            cv2.imwrite(mask_path, mask * 255)

            # Store in database
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                # Insert into disease_images
                cursor.execute(
                    """
                    INSERT INTO disease_images (user_id, crop_id, image_path, uploaded_at)
                    VALUES (%s, %s, %s, NOW())
                    """,
                    (session['user_id'], None, filename),
                )
                image_id = cursor.lastrowid

                # Insert into disease_results
                cursor.execute(
                    """
                    INSERT INTO disease_results (image_id, disease_name, confidence, predicted_at)
                    VALUES (%s, %s, %s, NOW())
                    """,
                    (image_id, disease_name, disease_conf),
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                print("Upload DB error:", e)
                flash('Database error. Please try again.', 'danger')
                return redirect(request.url)
            finally:
                cursor.close()
                conn.close()

            # Fetch remedies for this disease
            conn2 = get_db_connection()
            cursor2 = conn2.cursor(dictionary=True)

            # Strip crop prefix (e.g., "Tomato___" -> "")
            clean_name = disease_name.split("___")[-1]
            cursor2.execute("SELECT * FROM remedies WHERE disease_name = %s", (clean_name,))
            remedies = cursor2.fetchall()
            cursor2.close()
            conn2.close()

            # If no remedies found, provide default
            if not remedies:
                remedies = [{
                    'disease_name': clean_name,
                    'remedy_type': 'General',
                    'description': 'Apply broad-spectrum fungicide and improve airflow. Remove affected leaves.'
                }]

            # Render result
            prediction = {
                'disease': disease_name,
                'confidence': f"{disease_conf*100:.2f}%",
                'severity': f"{severity:.2f}%",
                'description': f'Segmentation severity: {severity:.2f}%. Classifier prediction: {disease_name}.',
                'treatment': 'See remedies below.',
                'remedies': remedies,
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'mask_url': url_for('static', filename=f'uploads/mask_{filename}.png'),
            }
            return render_template('upload.html', prediction=prediction)

    return render_template('upload.html', prediction=None)

@app.route('/schemes')
@login_required
def schemes():
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('home'))
        
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT scheme_id, scheme_name, description, min_age, max_age, min_land, max_land, state
        FROM govt_schemes
        ORDER BY scheme_name
        """
    )
    schemes_list = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('schemes.html', schemes=schemes_list)

@app.route('/msp')
@login_required
def msp():
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('home'))

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT m.msp_id, c.crop_name, m.state, m.year, m.price_per_quintal
            FROM msp_data m
            LEFT JOIN crops c ON c.crop_id = m.crop_id
            ORDER BY m.year DESC, c.crop_name, m.state
            """
        )
        msp_list = cursor.fetchall()
    except Exception as e:
        print("MSP Error:", e)
        flash('An error occurred. Please try again.', 'danger')
        msp_list = []
    finally:
        cursor.close()
        conn.close()

    return render_template('msp.html', msp_data=msp_list)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save temporarily
    filename = f"api_{file.filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Run models
    mask, severity = predict_disease_mask(image_path)
    disease_name, disease_conf = predict_disease_name(image_path)

    # Save mask image
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"mask_{filename}.png")
    cv2.imwrite(mask_path, mask * 255)

    # Fetch remedies
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    clean_name = disease_name.split("___")[-1]
    cursor.execute("SELECT * FROM remedies WHERE disease_name = %s", (clean_name,))
    remedies = cursor.fetchall()
    cursor.close()
    conn.close()

    if not remedies:
        remedies = [{
            'disease_name': clean_name,
            'remedy_type': 'General',
            'description': 'Apply broad-spectrum fungicide and improve airflow. Remove affected leaves.'
        }]

    return jsonify({
        'disease': disease_name,
        'confidence': f"{disease_conf*100:.2f}%",
        'severity': f"{severity:.2f}%",
        'remedies': remedies,
        'image_url': url_for('static', filename=f'uploads/{filename}'),
        'mask_url': url_for('static', filename=f'uploads/mask_{filename}.png')
    })

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been successfully logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
