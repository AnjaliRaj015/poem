from flask import Flask, render_template, request,url_for,redirect, current_app
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin,login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask_wtf.file import FileField, FileAllowed
import os
from datetime import datetime

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pytesseract
from PIL import Image


app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///new.db'
app.config['SECRET_KEY']='thisisasecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    pic = db.Column(db.String(120), nullable=True)
    name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    poem_history = db.relationship('PoemHistory', backref='user', lazy=True)
    
class PoemHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    poem = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class RegisterForm(FlaskForm):
    name = StringField(validators=[InputRequired(), Length(min=3, max=120)], render_kw={"placeholder": "Name"})
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    pic = FileField('Profile Image', validators=[FileAllowed(['jpg', 'jpeg', 'png', 'gif', 'webp'])])
    submit = SubmitField('SignUp')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')
class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')

# @app.route('/history')
# @login_required
# def history():
#     poem_history = PoemHistory.query.filter_by(user_id=current_user.id).order_by(PoemHistory.timestamp.desc()).all()
#     return render_template('history.html', poem_history=poem_history)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

def allowed_file(filename):
    # Check if the file has an allowed extension
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/register', methods=['GET','POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # Get the uploaded image file
        pic_file = form.pic.data
        if pic_file and allowed_file(pic_file.filename):
            # Securely save the file to the app's static folder
            filename = secure_filename(pic_file.filename)
            pic_path = os.path.join(current_app.static_folder, 'profile_pics', filename)
            pic_file.save(pic_path)
            pic_url = f'../static/profile_pics/{filename}'
        else:
            # Use a default image if no valid file was uploaded
            pic_url = '../static/profile_pics/p1.webp'

        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(pic=pic_url,name=form.name.data,username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

# Load the dataset
df = pd.read_csv("../final_trial.csv")

# Preprocess the text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['Poem'].apply(preprocess)

# Split the poem column into individual stanzas
df['stanzas'] = df['Poem'].apply(sent_tokenize)

# Convert emotions column to numerical labels
emotions = ['love', 'sad', 'anger', 'hate', 'fear', 'surprise', 'courage', 'joy', 'peace',"hope","care"]
df['emotion_label'] = df['Emotion'].apply(emotions.index)

# Extract features from the stanzas using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform([' '.join(stanza) for stanza in df['stanzas']])
y_train = df['emotion_label']

# Train a passive aggressive classifier model
model = PassiveAggressiveClassifier(random_state=42,max_iter=100)
model.fit(X_train_tfidf, y_train)

# Define a function to predict the emotion of each stanza in a new poem
def predict_emotions(new_poem):
    new_stanzas = sent_tokenize(new_poem)
    new_stanzas_tfidf = vectorizer.transform(new_stanzas)
    new_stanzas_emotions = model.predict(new_stanzas_tfidf)
    new_stanzas_emotions = [emotions[label] for label in new_stanzas_emotions]
    return list(zip(new_stanzas, new_stanzas_emotions))
@app.route('/history')
@login_required
def history():
    poem_history = PoemHistory.query.filter_by(user_id=current_user.id).order_by(PoemHistory.timestamp.desc()).all()
    poem_emotions = []
    for history in poem_history:
        emotions = predict_emotions(history.poem)
        poem_emotions.append((history, emotions))
    return render_template('history.html', poem_emotions=poem_emotions)

# Define the routes for the web application
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def predict():
    input_type = request.form['input_type']
    if input_type == 'text_input':
        new_poem = request.form['poem_text']
    elif input_type == 'file_input':
        file = request.files['file_input']
        if file.filename.endswith('.txt'):
            new_poem = file.read().decode('utf-8')
        elif file.filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image file
            img = Image.open(file)
            # Extract text from the image
            pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
            extracted_text = pytesseract.image_to_string(img)
            new_poem = extracted_text
        else:
            return render_template('index.html', error_message="Invalid file format. Please upload a text file or an image.")
    else:
        return render_template('index.html', error_message="Invalid input type. Please select either text input or file input.")

    predicted_emotions = predict_emotions(new_poem)
    # Create a new PoemHistory object for the current user
    new_poem_history = PoemHistory(user_id=current_user.id, poem=new_poem)
    db.session.add(new_poem_history)
    db.session.commit()
    return render_template('index.html', predicted_emotions=predicted_emotions)

# Run the web application
if __name__ == '__main__':
    app.run(debug=True)
