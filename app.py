import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image, ImageOps
from datetime import datetime

from health import health_recommendations_dict
from food import food_recommendations_dict

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Load Keras model
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Create a new table for analysis history
class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(150), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('analyses', lazy=True))

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home Route
@app.route("/")
def home():
    return render_template('index.html')

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials!", "danger")
    return render_template("login.html")

# Signup Route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["fullname"]
        email = request.form["email"]
        password = request.form["password"]
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash("Email already registered!", "warning")
        else:
            new_user = User(name=name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash("Signup successful! Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("signup.html")

# Logout Route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            flash("Invalid file type!", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform prediction
        image = Image.open(filepath).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = round(float(prediction[0][index]) * 100, 2)

        # Save prediction result to the database
        new_analysis = AnalysisHistory(
            user_id=current_user.id,
            filename=filename,
            result=class_name,
            confidence=confidence_score,
        )
        db.session.add(new_analysis)
        db.session.commit()

        flash(f"Prediction: {class_name} | Confidence: {confidence_score}%", "info")

    # Retrieve analysis history for the logged-in user
    user_analyses = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(AnalysisHistory.timestamp.desc()).all()

    return render_template("dashboard.html", user_analyses=user_analyses)

# Function to fetch health recommendations
def get_health_recommendations(disease):
    return health_recommendations_dict.get(disease)

# Function to fetch food recommendations
def get_food_recommendations(disease):
    return food_recommendations_dict.get(disease)

# Route for health recommendations
@app.route("/food_recommendation/<disease>", methods=["GET"])
@login_required  
def food_recommendation(disease):
    disease_key = disease.replace("_", " ").title()  # Convert to Title Case
    recommendations = food_recommendations_dict.get(disease_key, {"error": "No data found"})

    return render_template("food.html", disease=disease_key, data=recommendations)

@app.route("/health_recommendation/<disease>", methods=["GET"])
@login_required  
def health_recommendation(disease):
    disease_key = disease.replace("_", " ").title()  
    recommendations = health_recommendations_dict.get(disease_key, {"error": "No data found"})

    return render_template("health.html", disease=disease_key, data=recommendations)


# Run Flask app
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(port=5000, host="0.0.0.0", debug=True)