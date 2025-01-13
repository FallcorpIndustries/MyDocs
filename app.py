import os
import uuid
import glob
import json
import logging
import io
import whisper  # Added for Whisper integration
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from langdetect import detect
from pydub import AudioSegment
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration  # Removed Wav2Vec2 imports
from TTS.api import TTS
import nltk
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from docx import Document
import inflect

import pytesseract
from PIL import Image
import numpy as np
import re
import unicodedata
import subprocess

from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length
from flask_sqlalchemy import SQLAlchemy

from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo

from werkzeug.exceptions import RequestEntityTooLarge

from celery import Celery
from flask_caching import Cache

import soundfile as sf
import librosa

from sqlalchemy.orm import relationship

from pyhanko.sign import signers
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pytesseract
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Email credentials
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_password"
RECEIVER_EMAIL = "fallcorpsn@gmail.com"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CACHE_TYPE'] = 'simple'
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB


db = SQLAlchemy(app)
cache = Cache(app)

model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


PLAN_LIMITS = {
    'discovery': {
        'max_summaries_per_month': 5,
        'max_audio_generations_per_month': 5,
        'max_uploads_per_month': 5
    },
    'pro': {
        'max_summaries_per_month': 999999,
        'max_audio_generations_per_month': 999999,
        'max_uploads_per_month': 999999
    },
    'education': {
        'max_summaries_per_month': 200,
        'max_audio_generations_per_month': 200,
        'max_uploads_per_month': 200
    }
}



class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    hashed_password = db.Column(db.String(255), nullable=False)
    
    # Plan: can be "discovery", "pro", or "education"
    plan = db.Column(db.String(50), default='discovery', nullable=False)

    # Example usage counters
    monthly_uploads = db.Column(db.Integer, default=0)
    monthly_summaries = db.Column(db.Integer, default=0)
    monthly_audio_generations = db.Column(db.Integer, default=0)
    
    # relationships, etc.

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)


    
class RegisterForm(FlaskForm):
    username = StringField('Username',
        validators=[DataRequired(), Length(min=4, max=150)]
    )
    email = StringField('Email',
        validators=[DataRequired(), Email()]
    )
    password = PasswordField('Password',
        validators=[DataRequired(), Length(min=6)]
    )
    confirm_password = PasswordField('Confirm Password',
        validators=[DataRequired(), EqualTo('password', message="Passwords must match")]
    )
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')



class HistoryItem(db.Model):
    __tablename__ = 'history_item'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    text = db.Column(db.Text)
    filename = db.Column(db.String(200))
    description = db.Column(db.Text)
    needs_signature = db.Column(db.Boolean, default=False)
    signed = db.Column(db.Boolean, default=False)
    signed_filename = db.Column(db.String(200))
    # New field to store document language
    language = db.Column(db.String(10))  
    audios = db.relationship('Audio', backref='history_item', lazy=True)
    structures = db.relationship('DocStructure', backref='history_item', lazy=True)

class Audio(db.Model):
    __tablename__ = 'audio'
    id = db.Column(db.Integer, primary_key=True)
    history_item_id = db.Column(db.Integer, db.ForeignKey('history_item.id'), nullable=False)
    filename = db.Column(db.String(200))
    description = db.Column(db.String(200))

class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class DocStructure(db.Model):
    __tablename__ = 'doc_structure'
    id = db.Column(db.Integer, primary_key=True)
    history_item_id = db.Column(db.Integer, db.ForeignKey('history_item.id'), nullable=False)
    section_type = db.Column(db.String(50))
    content = db.Column(db.Text)
    order_index = db.Column(db.Integer)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


class EditTextForm(FlaskForm):
    text = TextAreaField('Text', validators=[DataRequired()])
    submit = SubmitField('Save Changes')

class FeedbackForm(FlaskForm):
    comment = TextAreaField('Your feedback', validators=[DataRequired()])
    submit = SubmitField('Submit Feedback')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DOCUMENTS_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'documents')
IMAGES_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
AUDIO_FOLDER = os.path.join('static', 'audio')
SUMMARIES_FOLDER = os.path.join(BASE_DIR, 'summaries')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
CERT_FOLDER = os.path.join(BASE_DIR, 'certificates')

os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(SUMMARIES_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(CERT_FOLDER, exist_ok=True)

jsignpdf_jar = os.path.join(CERT_FOLDER, 'JSignPdf.jar')
keystore_path = os.path.join(CERT_FOLDER, 'keystore.p12')
keystore_password = os.getenv('KEYSTORE_PASSWORD')
alias = 'myalias'

# Load TTS models
tts_en = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=torch.cuda.is_available())
tts_fr = TTS(model_name="tts_models/fr/css10/vits", progress_bar=False, gpu=torch.cuda.is_available())

loaded_models = {"en": tts_en, "fr": tts_fr}

print("Loading image captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Image captioning model loaded.")

p = inflect.engine()
# Specify the tessdata directory (optional if TESSDATA_PREFIX is set)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path as needed


print("Loading summarization model...")
pipeline_device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", device=pipeline_device)
print("Summarization model loaded.")

notification_messages = {
    "analyzing_document": {
        "text": "Analyzing Document",
        "filename": "analyzing_document.wav",
        "language": "en"
    },
    "analyzing_done": {
        "text": "Analyzing done, you can press play to read the document",
        "filename": "analyzing_done.wav",
        "language": "en"
    },
    "summarizing_document": {
        "text": "Summarizing Document",
        "filename": "summarizing_document.wav",
        "language": "en"
    },
    "summarizing_done": {
        "text": "Summarizing done, you can press play to read the summary",
        "filename": "summarizing_done.wav",
        "language": "en"
    },
    "item_added": {
        "text": "Item added successfully",
        "filename": "item_added.wav",
        "language": "en"
    },
    "item_deleted": {
        "text": "Item deleted successfully",
        "filename": "item_deleted.wav",
        "language": "en"
    },
    "audio_generated": {
        "text": "Audio generated successfully",
        "filename": "audio_generated.wav",
        "language": "en"
    },
    "error_notification": {
        "text": "An error occurred during audio generation",
        "filename": "error_notification.wav",
        "language": "en"
    }
}

def check_gpu_availability():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

check_gpu_availability()

def generate_notification_audios():
    for key, message in notification_messages.items():
        audio_path = os.path.join(AUDIO_FOLDER, message["filename"])
        if not os.path.exists(audio_path):
            tts_model = loaded_models.get(message["language"], tts_en)
            speaker = "p225" if message["language"] == "en" else None
            if speaker and hasattr(tts_model, 'speakers') and speaker in tts_model.speakers:
                tts_model.tts_to_file(text=message["text"], speaker=speaker, file_path=audio_path)
            else:
                tts_model.tts_to_file(text=message["text"], file_path=audio_path)
    print("Notification audios generated or already exist.")

generate_notification_audios()

def get_unique_id():
    return uuid.uuid4().hex

def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    replacements = {
        '●': ' bullet ',
        '•': ' bullet ',
        '/': ' slash ',
        '&': ' and ',
        '%': ' percent ',
        '$': ' dollars ',
        '@': ' at ',
        '#': ' number ',
        '*': ' asterisk ',
        '€': ' euros ',
        '£': ' pounds ',
        '¥': ' yen ',
    }
    for symbol, word in replacements.items():
        text = text.replace(symbol, word)
    text = text.replace("'", '')
    text = text.replace('"', '')
    text = text.replace('\n', '. ')
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_abbreviations(text):
    abbreviations = {
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Mrs.": "Misses",
        "St.": "Saint",
        "etc.": "et cetera",
        "e.g.": "for example",
    }
    regex = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b')
    result = regex.sub(lambda x: abbreviations[x.group()], text)
    return result

def normalize_numbers(text):
    words = text.split()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def split_text_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    return sentences

def split_text_into_chunks(text, max_chunk_size=500):
    sentences = split_text_into_sentences(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def extract_text_from_scanned_pdf(file_path):
    try:
        pages = convert_from_path(file_path)
        text = ""
        for page_data in pages:
            page_text = pytesseract.image_to_string(page_data)
            text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Failed to extract text from scanned PDF: {e}")
        return ""

def extract_text_with_headings(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            span_text = span["text"]
                            if font_size >= 12:
                                text += span_text.upper() + "\n"
                            else:
                                text += span_text + " "
            text += "\n"
        return text
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return ""
    

def can_upload(user):
    plan_limits = PLAN_LIMITS.get(user.plan, PLAN_LIMITS['discovery'])
    return user.monthly_uploads < plan_limits['max_uploads_per_month']



def can_summarize(user):
    """
    Return True if this user can still summarize (under plan limits).
    """
    plan = user.plan
    user_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS['discovery'])
    
    if user.monthly_summaries < user_limit['max_summaries_per_month']:
        return True
    return False

def can_generate_audio(user):
    """
    Return True if this user can still generate audio (under plan limits).
    """
    plan = user.plan
    user_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS['discovery'])
    
    if user.monthly_audio_generations < user_limit['max_audio_generations_per_month']:
        return True
    return False


@app.route('/upgrade', methods=['GET'])
@login_required
def upgrade_plans():
    # Show plan options
    return render_template('upgrade_plans.html')


def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.style and paragraph.style.name and paragraph.style.name.startswith('Heading'):
                full_text.append(paragraph.text.upper())
            else:
                full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Failed to extract text from DOCX: {e}")
        return ""

def parse_document_structure(text):
    lines = text.split('\n')
    structures = []
    idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isupper() and len(line.split()) <= 10:
            structures.append(("heading", line))
        elif line.lower().startswith("bullet") or re.match(r'^\*|^-|•', line):
            structures.append(("bullet", line))
        else:
            structures.append(("paragraph", line))
        idx += 1
    return structures

def detect_language_of_text(text):
    try:
        # langdetect returns 'fr' for French, 'en' for English, etc.
        lang = detect(text)
        if lang not in ['en', 'fr']:
            # Default to English if unknown
            lang = 'en'
        return lang
    except Exception:
        return 'en'

def extract_text(file_path):
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        elif file_path.endswith(".pdf"):
            text = extract_text_with_headings(file_path)
            if not text.strip():
                print("Performing OCR on scanned PDF.")
                text = extract_text_from_scanned_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            print("Unsupported file format.")
            return ""

        text = preprocess_text(text)
        text = expand_abbreviations(text)
        text = normalize_numbers(text)
        return text
    except Exception as e:
        print(f"Failed to extract text: {e}")
        return ""

def summarize_text(text, max_length=150, min_length=40):
    try:
        max_chunk_size = 1024
        chunks = split_text_into_chunks(text, max_chunk_size)
        summaries = []
        for chunk in chunks:
            summary = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summaries.append(summary[0]['summary_text'])
        final_summary = ' '.join(summaries)
        return final_summary
    except Exception as e:
        print(f"Summarization failed: {e}")
        return "Failed to generate summary."

def generate_image_description(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        device_for_blip = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_model.to(device_for_blip)
        inputs = blip_processor(image, return_tensors="pt").to(device_for_blip)
        generation_kwargs = {
            "max_new_tokens": 150,
            "num_beams": 5,
            "length_penalty": 1.2,
            "no_repeat_ngram_size": 2,
        }
        with torch.no_grad():
            out = blip_model.generate(**inputs, **generation_kwargs)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        return description
    except Exception as e:
        print(f"Error generating image description: {e}")
        return "Failed to generate image description."

# Modified: Remove existing ASR models (Wav2Vec2)
# Removed the following lines:
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# asr_processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# asr_model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", ignore_mismatched_sizes=True).to(model_device)
# asr_processor_fr = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")
# asr_model_fr = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french", ignore_mismatched_sizes=True).to(model_device)

def generate_audio_task(text, language, audio_filename):
    try:
        tts_model = loaded_models.get(language, loaded_models['en'])
        max_chunk_size = 500
        # Just ensure language is 'en' or 'fr'
        if language not in ['en', 'fr']:
            language = 'en'
        text_chunks = split_text_into_chunks(text, max_chunk_size)
        audio_segments = []
        for chunk in text_chunks:
            if hasattr(tts_model, 'speakers') and tts_model.speakers and language == 'en':
                # Use English speaker p225
                speaker = "p225"
                wav = tts_model.tts(chunk, speaker=speaker, progress_bar=False)
            else:
                wav = tts_model.tts(chunk, progress_bar=False)

            if isinstance(wav, (list, np.ndarray)):
                wav = np.array(wav)
                wav = np.clip(wav, -1.0, 1.0)
                wav = (wav * 32767).astype(np.int16)
                wav_bytes = wav.tobytes()
            elif isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
                wav = np.clip(wav, -1.0, 1.0)
                wav = (wav * 32767).astype(np.int16)
                wav_bytes = wav.tobytes()
            elif isinstance(wav, bytes):
                wav_bytes = wav
            else:
                wav_bytes = str(wav).encode('utf-8')

            audio_segment = AudioSegment(
                wav_bytes,
                frame_rate=int(tts_model.synthesizer.output_sample_rate),
                sample_width=2,
                channels=1
            )
            audio_segments.append(audio_segment)
        combined_audio = sum(audio_segments)
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        combined_audio.export(audio_path, format="wav")
        return audio_filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def allowed_file(filename, filetype):
    if filetype == 'document':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx'}
    elif filetype == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    return False

def document_needs_signature(text):
    signature_keywords = ['sign here', 'signature', 'please sign', 'signatory', 'authorized signature', 'sign and date']
    text_lower = text.lower()
    for keyword in signature_keywords:
        if keyword in text_lower:
            return True
    return False

def generate_tts_audio(text, language='en'):
    try:
        tts_model = loaded_models.get(language, loaded_models['en'])
        audio_id = get_unique_id()
        audio_filename = f"{audio_id}.wav"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        if hasattr(tts_model, 'speakers') and tts_model.speakers and language == 'en':
            # Use English speaker p225
            tts_model.tts_to_file(text=text, speaker="p225", file_path=audio_path)
        else:
            tts_model.tts_to_file(text=text, file_path=audio_path)
        return audio_filename
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return None

def generate_self_signed_cert(cert_file_path, key_file_path):
    if os.path.exists(cert_file_path) and os.path.exists(key_file_path):
        return
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    with open(key_file_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Your State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Your City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Your Organization"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"yourdomain.com"),
    ])
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow() - datetime.timedelta(days=1)
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=3650)
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
        critical=False,
    ).sign(key, hashes.SHA256(), default_backend())
    with open(cert_file_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

def apply_digital_signature(input_pdf_path, output_pdf_path, keystore_path, keystore_password, alias, jsignpdf_jar_path):
    try:
        if not all([input_pdf_path, output_pdf_path, keystore_path, alias, jsignpdf_jar_path]):
            logger.error("Missing parameters for signing.")
            return False
        if keystore_password is None:
            keystore_password = "123"
        if not os.path.exists(input_pdf_path):
            logger.error(f"Input PDF not found: {input_pdf_path}")
            return False
        if not os.path.exists(keystore_path):
            logger.error(f"Keystore not found: {keystore_path}")
            return False
        if not os.path.exists(jsignpdf_jar_path):
            logger.error(f"JSignPdf.jar not found: {jsignpdf_jar_path}")
            return False

        command = [
            'java',
            '-jar',
            jsignpdf_jar_path,
            input_pdf_path,
            '-d', os.path.dirname(output_pdf_path),
            '-os', os.path.basename(output_pdf_path),
            '-ksf', keystore_path,
            '-ksp', keystore_password,
            '-ka', alias,
            '-r', 'Document signing',
            '-l', 'MTL',
            '-V',
            '-pg', '1',
            '-llx', '100',
            '-lly', '100',
            '-urx', '200',
            '-ury', '150'
        ]

        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"jsignpdf error: {result.stderr}")
            return False

        logger.info("PDF signed successfully.")
        return True
    except Exception as e:
        logger.exception(f"Exception during PDF signing: {e}")
        return False

def rename_file_to_clean_pattern(directory):
    signed_files = glob.glob(os.path.join(directory, "*signed*"))
    if not signed_files:
        raise FileNotFoundError(f"No file containing 'signed' found in directory: {directory}")
    original_filepath = signed_files[0]
    base_name, extension = os.path.splitext(os.path.basename(original_filepath))
    if extension == ".pdf" and base_name.endswith(".pdf"):
        base_name = base_name[:-4]
    cleaned_filename = f"signed_{base_name.replace('signed_', '').strip('_')}{extension}"
    new_filepath = os.path.join(directory, cleaned_filename)
    print(f"Renaming file:\n  From: {original_filepath}\n  To: {new_filepath}")
    os.rename(original_filepath, new_filepath)
    return new_filepath

def send_feedback_email(username, comment):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = "New Feedback Received"
    body = f"User: {username}\nFeedback:\n{comment}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Feedback email sent successfully.")
    except Exception as e:
        print(f"Failed to send feedback email: {e}")

@app.route('/')
@login_required
def index():
    history = HistoryItem.query.filter_by(user_id=current_user.id).order_by(HistoryItem.id.desc()).all()
    feedbacks = Feedback.query.order_by(Feedback.id.desc()).all()
    feedback_form = FeedbackForm()
    return render_template('index.html', history=history, notification_messages=notification_messages, feedback_form=feedback_form, feedbacks=feedbacks)

@app.route('/login', methods=['GET', 'POST'])

def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        # e.g. find user by username, then check their password
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash("Logged in successfully!")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.")
    
    return render_template('login.html', form=form)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload_file', methods=['POST'])
@login_required
def upload_file():
    # Check monthly upload limit first
    if not can_upload(current_user):
        flash("Monthly upload limit reached! Please consider upgrading.")
        session['show_upgrade_modal'] = True
        return redirect(url_for('index'))

    # Get file from the request
    file_type = request.form.get('file_type')
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('index'))

    # Define maximum file size (e.g., 16 MB)
    max_file_size = 16 * 1024 * 1024  # 16 MB

    # Check if the file size is available and exceeds our limit
    if file.content_length and file.content_length > max_file_size:
        flash("The file is too large to upload. Please select a smaller file.")
        return redirect(url_for('index'))

    # Process document uploads
    if file_type == 'document' and file and allowed_file(file.filename, 'document'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(DOCUMENTS_FOLDER, filename)
        file.save(file_path)

        # Verify file size after saving as an extra precaution
        if os.path.getsize(file_path) > max_file_size:
            os.remove(file_path)
            flash("The file is too large to upload. Please select a smaller file.")
            return redirect(url_for('index'))

        text = extract_text(file_path)
        if not text:
            flash('Failed to extract text from the document.')
            return redirect(url_for('index'))

        needs_signature = document_needs_signature(text)
        doc_lang = detect_language_of_text(text)

        new_document = HistoryItem(
            user_id=current_user.id,
            type='Document',
            text=text,
            filename=filename,
            needs_signature=needs_signature,
            language=doc_lang
        )
        db.session.add(new_document)
        db.session.commit()

        structures = parse_document_structure(text)
        order_idx = 0
        for st_type, st_content in structures:
            ds = DocStructure(
                history_item_id=new_document.id,
                section_type=st_type,
                content=st_content,
                order_index=order_idx
            )
            db.session.add(ds)
            order_idx += 1
        db.session.commit()

        current_user.monthly_uploads += 1
        db.session.commit()

        flash('item_added')
        return redirect(url_for('index'))

    # Process image uploads
    elif file_type == 'image' and file and allowed_file(file.filename, 'image'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(IMAGES_FOLDER, filename)
        file.save(file_path)

        # Verify file size after saving as an extra precaution
        if os.path.getsize(file_path) > max_file_size:
            os.remove(file_path)
            flash("The file is too large to upload. Please select a smaller file.")
            return redirect(url_for('index'))

        description = generate_image_description(file_path)
        img_lang = detect_language_of_text(description)

        new_image = HistoryItem(
            user_id=current_user.id,
            type='Image',
            description=description,
            filename=filename,
            language=img_lang
        )
        db.session.add(new_image)
        db.session.commit()

        current_user.monthly_uploads += 1
        db.session.commit()

        flash('item_added')
        return redirect(url_for('index'))

    else:
        flash('Unsupported file format or invalid file type.')
        return redirect(url_for('index'))

@app.route('/edit_text/<int:item_id>', methods=['GET', 'POST'])
@login_required
def edit_text(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Invalid history item.')
        return redirect(url_for('index'))
    if item.type != 'Document':
        flash('Selected item is not a document.')
        return redirect(url_for('index'))
    form = EditTextForm()
    if form.validate_on_submit():
        edited_text = form.text.data
        item.text = edited_text
        item.language = detect_language_of_text(edited_text)
        db.session.commit()
        DocStructure.query.filter_by(history_item_id=item.id).delete()
        structures = parse_document_structure(edited_text)
        order_idx = 0
        for st_type, st_content in structures:
            ds = DocStructure(
                history_item_id=item.id,
                section_type=st_type,
                content=st_content,
                order_index=order_idx
            )
            db.session.add(ds)
            order_idx += 1
        db.session.commit()

        flash('Text updated successfully.')
        return redirect(url_for('index'))
    form.text.data = item.text
    return render_template('edit_text.html', form=form, item_id=item_id)

@app.route('/view_text/<int:item_id>')
@login_required
def view_text(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item or item.type != 'Document':
        flash('Invalid item or not a document.')
        return redirect(url_for('index'))
    return render_template('view_text.html', item=item)

@app.route('/summarize_document/<int:item_id>', methods=['POST'])
@login_required
def summarize_document_route(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        return jsonify({'success': False, 'message': 'Invalid history item.'}), 400
    if item.type != "Document":
        return jsonify({'success': False, 'message': 'Selected item is not a document.'}), 400

    # (NEW) Check if user can still summarize
    if not can_summarize(current_user):
        return jsonify({'success': False, 'message': 'You have reached your monthly summary limit. Upgrade to Pro plan!'}), 403

    text = item.text
    if not text:
        return jsonify({'success': False, 'message': 'No text available to summarize.'}), 400

    existing_summary = HistoryItem.query.filter_by(
        user_id=current_user.id,
        type='Summary',
        description=f"Summary of {item.filename}"
    ).first()

    if existing_summary:
        return jsonify({'success': True, 'message': 'Summary already exists.'}), 200

    summary = summarize_text(text)
    if summary:
        # (NEW) If success, increment user's monthly_summaries
        current_user.monthly_summaries += 1
        db.session.add(current_user)  # to commit the usage increment

        new_summary = HistoryItem(
            user_id=current_user.id,
            type='Summary',
            text=summary,
            description=f"Summary of {item.filename}",
            language=item.language
        )
        db.session.add(new_summary)
        db.session.commit()

        language = new_summary.language if new_summary.language in ['en','fr'] else 'en'
        summary_audio_filename = get_unique_id() + '.wav'
        generate_audio_task(summary, language, summary_audio_filename)
        summary_audio = Audio(
            history_item_id=new_summary.id,
            filename=summary_audio_filename,
            description=f"Audio of Summary of {item.filename}"
        )
        db.session.add(summary_audio)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Summarization done.'}), 200
    else:
        return jsonify({'success': False, 'message': 'Failed to generate summary.'}), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    flash("The file is too large to upload. Please select a smaller file.")
    return redirect(request.url)


@app.route('/generate_audio/<int:item_id>', methods=['POST'])
@login_required
def generate_audio_route(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        return jsonify({'success': False, 'message': 'Invalid history item.'}), 400

    # (NEW) Check plan limit
    if not can_generate_audio(current_user):
        return jsonify({'success': False, 'message': 'You have reached your monthly audio generation limit. Upgrade to Pro!'}), 403

    existing_audio = Audio.query.filter_by(
        history_item_id=item.id,
        description=f"Audio of {item.filename or item.description}"
    ).first()

    if existing_audio:
        return jsonify({'success': True, 'message': 'Audio already exists.'}), 200

    text = item.text or item.description
    language = item.language if item.language in ['en','fr'] else 'en'
    audio_filename = get_unique_id() + '.wav'
    result = generate_audio_task(text, language, audio_filename)
    if result:
        # (NEW) increment usage
        current_user.monthly_audio_generations += 1
        db.session.add(current_user)

        audio_entry = Audio(
            history_item_id=item.id,
            filename=audio_filename,
            description=f"Audio of {item.filename or item.description}"
        )
        db.session.add(audio_entry)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Audio generation started.'}), 200
    else:
        return jsonify({'success': False, 'message': 'Failed to generate audio.'}), 500


@app.route('/delete_history_item/<int:item_id>', methods=['POST'])
@login_required
def delete_history_item(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Invalid history item.')
        return redirect(url_for('index'))

    # If deleting a document, also delete related summaries
    if item.type == "Document":
        # Delete the document file
        document_path = os.path.join(DOCUMENTS_FOLDER, item.filename)
        if os.path.exists(document_path):
            os.remove(document_path)
        if item.signed_filename:
            signed_document_path = os.path.join(DOCUMENTS_FOLDER, item.signed_filename)
            if os.path.exists(signed_document_path):
                os.remove(signed_document_path)

        # Find and delete summaries associated with this document
        related_summaries = HistoryItem.query.filter_by(
            user_id=current_user.id,
            type='Summary',
            description=f"Summary of {item.filename}"
        ).all()
        for summary in related_summaries:
            # Delete audio files related to the summary
            for audio in summary.audios:
                audio_path = os.path.join(AUDIO_FOLDER, audio.filename)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                db.session.delete(audio)
            # Delete document structure entries related to summary if needed
            DocStructure.query.filter_by(history_item_id=summary.id).delete()
            db.session.delete(summary)

    elif item.type == "Image":
        image_path = os.path.join(IMAGES_FOLDER, item.filename)
        if os.path.exists(image_path):
            os.remove(image_path)

    # Delete audios associated with the item
    for audio in item.audios:
        audio_path = os.path.join(AUDIO_FOLDER, audio.filename)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        db.session.delete(audio)

    # Delete document structure entries for this item
    DocStructure.query.filter_by(history_item_id=item.id).delete()

    # Finally, delete the main item
    db.session.delete(item)
    db.session.commit()
    flash('item_deleted')
    return redirect(url_for('index'))


@app.route('/help')
def help_route():
    return render_template('help.html')

@app.route('/about')
def about_route():
    return render_template('about.html')

@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    form = FeedbackForm()
    if form.validate_on_submit():
        new_feedback = Feedback(
            user_id=current_user.id,
            comment=form.comment.data
        )
        db.session.add(new_feedback)
        db.session.commit()
        flash('Feedback submitted.')
        send_feedback_email(current_user.username, form.comment.data)
    else:
        flash('Failed to submit feedback.')
    return redirect(url_for('index'))

# Removed the get_asr_models_for_language function
# def get_asr_models_for_language(lang):
#     if lang == 'fr':
#         return asr_processor_fr, asr_model_fr
#     else:
#         # default to English
#         return asr_processor_en, asr_model_en

# Added Whisper model loading
whisper_model = whisper.load_model("large")  # You can choose "tiny", "base", "small", "medium", "large"

@app.route('/process_voice_command_enhanced/<int:item_id>', methods=['POST'])
@login_required
def process_voice_command_enhanced(item_id):
    if 'audio_data' not in request.files:
        return jsonify({'success': False, 'message': 'No audio data provided.'}), 400

    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item or item.type != "Document":
        return jsonify({'success': False, 'message': 'Invalid document.'}), 400

    lang = item.language if item.language in ['en','fr'] else 'en'

    audio_file = request.files['audio_data']
    audio_data = audio_file.read()
    mime_type = request.form.get('mime_type', 'audio/webm')

    try:
        # Convert incoming audio to WAV format using pydub
        format = mime_type.split('/')[1].split(';')[0]
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=format)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)

        # Save the WAV audio to a temporary file for Whisper
        temp_filename = f"temp_{uuid.uuid4().hex}.wav"
        temp_filepath = os.path.join(TEMP_FOLDER, temp_filename)  # Use appropriate temp directory
        with open(temp_filepath, 'wb') as f:
            f.write(wav_io.read())

    except Exception as e:
        print(f"Error processing audio data: {e}")
        return jsonify({'success': False, 'message': 'Failed to process audio.'}), 500

    try:
        # Use Whisper to transcribe the audio
        lang_map = {'en': 'english', 'fr': 'french'}
        whisper_lang = lang_map.get(lang, 'english')  # Default to English if not specified

        result = whisper_model.transcribe(temp_filepath, language=whisper_lang)
        command = result['text'].lower().strip()
        print(f"Enhanced command: {command}")

        # Delete the temporary file after transcription
        os.remove(temp_filepath)

    except Exception as e:
        print(f"Error during speech recognition with Whisper: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return jsonify({'success': False, 'message': 'Failed to process audio.'}), 500

    doc_structs = DocStructure.query.filter_by(history_item_id=item.id).order_by(DocStructure.order_index).all()

    def speak(t):
        filename = get_unique_id() + ".wav"
        generate_audio_task(t, lang, filename)
        return filename

    # Define indicative words with weights for enhanced scoring
    if lang == 'en':
        command_keywords = {
            'ocr': {"ocr": 3, "extract": 2, "text": 1, "page": 1, "read": 1, "scan": 2},
            'heading': {"heading": 3, "entitle": 1, "go": 1, "move": 1, "jump": 1},
            'summarize': {"summarize": 3, "summary": 2, "sum up": 2, "short version": 1},
            'mentions': {"mentions": 3, "find": 1, "references": 2, "keyword": 1},
            'compare': {"compare": 3, "differences": 2, "document": 1},
            'bullet': {"bullet": 3, "list": 1, "points": 1},
            'skip_refs': {"skip": 2, "ignore": 2, "references": 2},
            'author': {"author": 3, "who wrote": 2, "writer": 2},
            'title': {"title": 3, "name of document": 2}
        }
    else:
        # French equivalent with weights
        command_keywords = {
            'ocr': {"ocr": 3, "extrait": 2, "texte": 1, "page": 1, "lire": 1, "scannez": 2},
            'heading': {"entête": 3, "entete": 3, "aller": 1, "section": 1, "titre": 1},
            'summarize': {"résumer": 3, "résumé": 3, "raccourcir": 2, "summarise": 2},
            'mentions': {"mentions": 3, "références": 2, "trouver": 1, "mots-clés": 2},
            'compare': {"comparer": 3, "differences": 2, "document": 1},
            'bullet': {"bullet": 3, "liste": 1, "points": 1},
            'skip_refs': {"ignorer": 2, "références": 2, "pas references": 1},
            'author': {"auteur": 3, "qui a écrit": 2, "auteur du document": 2},
            'title': {"titre": 3, "nom du document": 2}
        }

    # Function to calculate weighted score
    def calculate_weighted_score(cmd, keywords_dict):
        score = {}
        for command_type, keywords in keywords_dict.items():
            cmd_score = 0
            for kw, weight in keywords.items():
                if kw in cmd:
                    cmd_score += weight
            score[command_type] = cmd_score
        return score

    # Calculate scores for each command type
    scores = calculate_weighted_score(command, command_keywords)

    # Determine the best command based on highest score
    best_command = max(scores, key=scores.get)
    best_score = scores[best_command]

    # Define a threshold to decide whether to act on the command
    threshold = 3

    if best_score >= threshold:
        # Execute the command based on best_command
        if best_command == 'ocr':
            # Extract page number
            page_num_match = re.search(r'page (\d+)', command)
            if page_num_match:
                page_num = int(page_num_match.group(1))
                filepath = os.path.join(DOCUMENTS_FOLDER, item.filename)
                try:
                    pages = convert_from_path(filepath)
                except Exception as e:
                    print(f"Error converting PDF to images: {e}")
                    resp_text = "Failed to convert document to images for OCR." if lang == 'en' else "Échec de la conversion du document en images pour l'OCR."
                    audio_file = speak(resp_text)
                    return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})
    
                if 1 <= page_num <= len(pages):
                    page_image = pages[page_num-1]
                    page_text = pytesseract.image_to_string(page_image, lang=lang)
                    item.text += "\n" + page_text
                    db.session.commit()
    
                    # Update document structure
                    DocStructure.query.filter_by(history_item_id=item.id).delete()
                    structures = parse_document_structure(item.text)
                    order_idx = 0
                    for st_type, st_content in structures:
                        ds = DocStructure(
                            history_item_id=item.id,
                            section_type=st_type,
                            content=st_content,
                            order_index=order_idx
                        )
                        db.session.add(ds)
                        order_idx += 1
                    db.session.commit()
    
                    resp_text = f"OCR performed on page {page_num}. Text updated." if lang == 'en' else f"OCR effectuée sur la page {page_num}. Texte mis à jour."
                    audio_file = speak(resp_text)
                    return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
                else:
                    resp_text = "Invalid page number." if lang == 'en' else "Numéro de page invalide."
                    audio_file = speak(resp_text)
                    return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})
            else:
                resp_text = "Please specify the page number for OCR." if lang == 'en' else "Veuillez préciser le numéro de page pour l'OCR."
                audio_file = speak(resp_text)
                return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'heading':
            # Extract heading number
            heading_match = re.search(r'heading (\d+)|entête (\d+)|entete (\d+)', command)
            if heading_match:
                heading_index = int(heading_match.group(1) or heading_match.group(2) or heading_match.group(3)) - 1
                headings = [s for s in doc_structs if s.section_type == 'heading']
                if 0 <= heading_index < len(headings):
                    content = headings[heading_index].content
                    audio_file = speak(content)
                    return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
                else:
                    resp_text = "Heading not found." if lang == 'en' else "En-tête introuvable."
                    audio_file = speak(resp_text)
                    return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})
            else:
                resp_text = "Please specify the heading number." if lang == 'en' else "Veuillez préciser le numéro de l'entête."
                audio_file = speak(resp_text)
                return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'summarize':
            # Attempt to extract section name
            if lang == 'en':
                after_word = re.sub(r'^.*summarize', '', command).strip()
            else:
                after_word = re.sub(r'^.*résumer', '', command).strip()
                after_word = re.sub(r'^.*resume', '', after_word).strip()

            if after_word:
                # Summarize specific section
                target_heading = None
                for s in doc_structs:
                    if s.section_type == 'heading' and after_word in s.content.lower():
                        target_heading = s.content
                        break
                if target_heading:
                    start_index = next((i for i, v in enumerate(doc_structs) if v.content == target_heading), None)
                    if start_index is not None:
                        texts_to_summarize = []
                        for v in doc_structs[start_index+1:]:
                            if v.section_type == 'heading':
                                break
                            texts_to_summarize.append(v.content)
                        full_text = ' '.join(texts_to_summarize)
                        if full_text.strip():
                            sum_text = summarize_text(full_text)
                            audio_file = speak(sum_text)
                            return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
                        else:
                            resp_text = "No text found to summarize in that section." if lang == 'en' else "Aucun texte à résumer dans cette section."
                            audio_file = speak(resp_text)
                            return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})
                # If heading not found, summarize entire document
                sum_text = summarize_text(item.text)
                audio_file = speak(sum_text)
                return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
            else:
                # Summarize entire document
                sum_text = summarize_text(item.text)
                audio_file = speak(sum_text)
                return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'mentions':
            # Extract keyword
            mentions_match = re.search(r'(?:mentions? of|mentions? de) (.+)', command)
            if mentions_match:
                keyword = mentions_match.group(1).strip()
                matches = [v.content for v in doc_structs if keyword.lower() in v.content.lower()]
                if matches:
                    result_text = " ".join(matches[:5])
                    resp_text = f"Found the following references: {result_text}" if lang == 'en' else f"Mentions trouvées: {result_text}"
                    audio_file = speak(resp_text)
                    return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
                else:
                    resp_text = "No mentions found." if lang == 'en' else "Aucune mention trouvée."
                    audio_file = speak(resp_text)
                    return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})
            else:
                resp_text = "Please specify what to find mentions of." if lang == 'en' else "Veuillez préciser ce dont vous voulez trouver des mentions."
                audio_file = speak(resp_text)
                return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'compare':
            # Extract document ID
            compare_match = re.search(r'document (\d+)', command)
            if compare_match:
                compare_id = int(compare_match.group(1))
                other_item = HistoryItem.query.filter_by(id=compare_id, user_id=current_user.id).first()
                if other_item and other_item.type == "Document":
                    old_lines = set(item.text.split('\n'))
                    new_lines = set(other_item.text.split('\n'))
                    added = list(new_lines - old_lines)
                    removed = list(old_lines - new_lines)
                    resp_text = "Changes found. Added: " if lang == 'en' else "Modifications détectées. Ajouté: "
                    resp_text += ' '.join(added[:3]) + " "
                    resp_text += "Removed: " if lang == 'en' else "Supprimé: "
                    resp_text += ' '.join(removed[:3])
                    audio_file = speak(resp_text)
                    return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
                else:
                    resp_text = "Document to compare not found or not a document." if lang == 'en' else "Document à comparer introuvable ou non valide."
                    audio_file = speak(resp_text)
                    return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})
            else:
                resp_text = "Please specify a document ID to compare." if lang == 'en' else "Veuillez préciser l'ID du document à comparer."
                audio_file = speak(resp_text)
                return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'bullet':
            bullets = [v.content for v in doc_structs if v.section_type == 'bullet']
            if bullets:
                bullets_text = " ".join(bullets)
                audio_file = speak(bullets_text)
                return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
            else:
                resp_text = "No bullet points found." if lang == 'en' else "Aucun point de liste trouvé."
                audio_file = speak(resp_text)
                return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'skip_refs':
            filtered = [v.content for v in doc_structs if "reference" not in v.content.lower()]
            filtered_text = " ".join(filtered[:10])
            audio_file = speak(filtered_text)
            return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'author':
            author = None
            if item.filename.lower().endswith('.pdf'):
                try:
                    doc = fitz.open(os.path.join(DOCUMENTS_FOLDER, item.filename))
                    meta = doc.metadata
                    author = meta.get('author', None)
                    doc.close()
                except Exception as e:
                    print(f"Error extracting author from PDF: {e}")
            if author:
                resp_text = f"The author is {author}" if lang == 'en' else f"L'auteur est {author}"
            else:
                resp_text = "Author not found." if lang == 'en' else "Auteur introuvable."
            audio_file = speak(resp_text)
            return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})

        elif best_command == 'title':
            headings = [s for s in doc_structs if s.section_type == 'heading']
            if headings:
                title = headings[0].content
                resp_text = f"The title is {title}" if lang == 'en' else f"Le titre est {title}"
                audio_file = speak(resp_text)
                return jsonify({'success': True, 'audio': url_for('static', filename='audio/' + audio_file)})
            else:
                resp_text = "No title found." if lang == 'en' else "Aucun titre trouvé."
                audio_file = speak(resp_text)
                return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})

    else:
        # If no command matches the threshold, respond accordingly
        resp_text = "Sorry, I didn't understand the command." if lang == 'en' else "Désolé, je n'ai pas compris la commande."
        audio_file = speak(resp_text)
        return jsonify({'success': False, 'audio': url_for('static', filename='audio/' + audio_file)})

@app.route('/voice_sign_document/<int:item_id>', methods=['GET'])
@login_required
def voice_sign_document(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Invalid history item.')
        return redirect(url_for('index'))
    if item.type != "Document":
        flash('Selected item is not a document.')
        return redirect(url_for('index'))
    text = item.text
    if not text:
        flash('No text available.')
        return redirect(url_for('index'))

    needs_signature = document_needs_signature(text)
    if not needs_signature:
        flash('This document does not require a signature.')
        return redirect(url_for('index'))

    return render_template('voice_sign_document.html', item=item)


@app.route('/change_plan', methods=['GET', 'POST'])
def change_plan():
    
    chosen_plan = request.form.get('plan')
    if chosen_plan not in ['discovery', 'pro', 'education']:
        flash("Invalid plan selected.")
        return redirect(url_for('index'))  # or maybe back to the modal

    # If you want a payment flow, you'd handle that here for 'pro' or 'education'
    # For now, we just set the plan
    current_user.plan = chosen_plan
    # Optionally reset counters so they start fresh:
    current_user.monthly_uploads = 0
    current_user.monthly_summaries = 0
    current_user.monthly_audio_generations = 0

    db.session.commit()

    flash(f"You have switched to the {chosen_plan.capitalize()} plan!")
    return redirect(url_for('index'))


@app.route('/start_voice_signing/<int:item_id>', methods=['POST'])
@login_required
def start_voice_signing(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        return jsonify({'success': False, 'message': 'Invalid history item.'}), 400
    if item.type != "Document":
        return jsonify({'success': False, 'message': 'Selected item is not a document.'}), 400
    text = item.text
    if not text:
        return jsonify({'success': False, 'message': 'No text available.'}), 400

    prompt_text = "Do you wish to sign the document? Say 'Yes' to confirm or 'No' to cancel."
    audio_filename = generate_tts_audio(prompt_text, language=item.language)
    if not audio_filename:
        return jsonify({'success': False, 'message': 'Failed to generate audio prompt.'}), 500

    audio_url = url_for('static', filename='audio/' + audio_filename)
    session['voice_signing'] = {
        'item_id': item_id,
        'state': 'ask_sign'
    }
    return jsonify({'success': True, 'audio_url': audio_url})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash("This email is already in use. Please log in or use another email.")
            return redirect(url_for('register'))

        new_user = User(
            username=form.username.data,
            email=form.email.data,
            plan='discovery'  # Default to free plan
        )
        new_user.set_password(form.password.data)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        flash("Registration successful! (Discovery Plan)")
        return redirect(url_for('index'))

    return render_template('register.html', form=form)


def process_voice_command_signing(item, response_text):
    document_filename = item.filename
    document_path = os.path.join(DOCUMENTS_FOLDER, document_filename)
    if not os.path.exists(document_path):
        return False, "Document not found."

    if not document_path.lower().endswith('.pdf'):
        return False, "Only PDF documents can be signed."

    cert_file_path = os.path.join(CERT_FOLDER, 'cert.pem')
    key_file_path = os.path.join(CERT_FOLDER, 'key.pem')
    generate_self_signed_cert(cert_file_path, key_file_path)

    base_name, extension = document_filename.rsplit('.', 1)
    if not base_name.startswith("signed_"):
        signed_document_filename = f"signed_{base_name}.{extension}"
    else:
        signed_document_filename = f"{base_name}.{extension}"
    signed_document_path = os.path.join(DOCUMENTS_FOLDER, signed_document_filename)

    success = apply_digital_signature(
        input_pdf_path=document_path,
        output_pdf_path=signed_document_path,
        keystore_path=keystore_path,
        keystore_password=keystore_password,
        alias=alias,
        jsignpdf_jar_path=jsignpdf_jar
    )

    if success:
        try:
            new_signed_filepath = rename_file_to_clean_pattern(DOCUMENTS_FOLDER)
            new_signed_filename = os.path.basename(new_signed_filepath)
        except FileNotFoundError as e:
            logger.error(f"Error during renaming: {e}")
            return False, str(e)

        item.signed = True
        item.signed_filename = new_signed_filename
        db.session.commit()

        unique_code = get_unique_id()
        doc = fitz.open(new_signed_filepath)
        meta = doc.metadata
        meta['keywords'] = unique_code
        doc.set_metadata(meta)

        temp_filepath = new_signed_filepath + "_temp.pdf"
        doc.save(temp_filepath)  # Save to temp
        doc.close()

        os.remove(new_signed_filepath)
        os.rename(temp_filepath, new_signed_filepath)
        return True, "The document has been signed."
    else:
        return False, "Failed to sign document."

@app.route('/process_voice_command/<int:item_id>', methods=['POST'])
@login_required
def process_voice_command(item_id):
    voice_signing = session.get('voice_signing', {})
    if not voice_signing or voice_signing.get('item_id') != item_id:
        return jsonify({'success': False, 'message': 'Voice signing session not found.'}), 400
    if 'audio_data' not in request.files:
        return jsonify({'success': False, 'message': 'No audio data provided.'}), 400

    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        return jsonify({'success': False, 'message': 'Invalid history item.'}), 400

    lang = item.language if item.language in ['en','fr'] else 'en'

    audio_file = request.files['audio_data']
    audio_data = audio_file.read()
    mime_type = request.form.get('mime_type', 'audio/webm')

    try:
        # Convert incoming audio to WAV format using pydub
        format = mime_type.split('/')[1].split(';')[0]
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=format)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)

        # Save the WAV audio to a temporary file for Whisper
        temp_filename = f"temp_{uuid.uuid4().hex}.wav"
        temp_filepath = os.path.join(TEMP_FOLDER, temp_filename)  # Use appropriate temp directory
        with open(temp_filepath, 'wb') as f:
            f.write(wav_io.read())

    except Exception as e:
        print(f"Error processing audio data: {e}")
        return jsonify({'success': False, 'message': 'Failed to process audio.'}), 500

    try:
        # Use Whisper to transcribe the audio
        lang_map = {'en': 'english', 'fr': 'french'}
        whisper_lang = lang_map.get(lang, 'english')  # Default to English if not specified

        result = whisper_model.transcribe(temp_filepath, language=whisper_lang)
        response_text = result['text'].lower().strip()
        print(f"Recognized speech: {response_text}")

        # Delete the temporary file after transcription
        os.remove(temp_filepath)

    except Exception as e:
        print(f"Error during speech recognition with Whisper: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return jsonify({'success': False, 'message': 'Failed to process audio.'}), 500

    current_state = voice_signing.get('state')
    if current_state == 'ask_sign':
        if 'yes' in response_text:
            success, msg = process_voice_command_signing(item, response_text)
            if success:
                audio_filename = generate_tts_audio(msg, language=lang)
                audio_url = url_for('static', filename='audio/' + audio_filename)
                voice_signing.clear()
                session['voice_signing'] = voice_signing
                return jsonify({'success': True, 'action': 'signed', 'audio_url': audio_url})
            else:
                audio_filename = generate_tts_audio(msg, language=lang)
                audio_url = url_for('static', filename='audio/' + audio_filename)
                return jsonify({'success': True, 'action': 'error', 'audio_url': audio_url})
        elif 'no' in response_text:
            voice_signing.clear()
            session['voice_signing'] = voice_signing
            msg = "Signing cancelled." if lang == 'en' else "Signature annulée."
            audio_filename = generate_tts_audio(msg, language=lang)
            audio_url = url_for('static', filename='audio/' + audio_filename)
            return jsonify({'success': True, 'action': 'cancelled', 'audio_url': audio_url})
        else:
            msg = "Please say 'Yes' to sign or 'No' to cancel." if lang == 'en' else "Veuillez dire 'Oui' pour signer ou 'Non' pour annuler."
            audio_filename = generate_tts_audio(msg, language=lang)
            audio_url = url_for('static', filename='audio/' + audio_filename)
            return jsonify({'success': True, 'action': 'ask_sign', 'audio_url': audio_url})
    else:
        voice_signing.clear()
        session['voice_signing'] = voice_signing
        return jsonify({'success': False, 'message': 'Invalid state.'}), 500

@app.route('/manual_sign_document/<int:item_id>', methods=['POST'])
@login_required
def manual_sign_document(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item or item.type != "Document":
        flash('Invalid or non-document item.')
        return redirect(url_for('index'))
    if not item.needs_signature:
        flash('This document does not require a signature.')
        return redirect(url_for('index'))

    success, msg = process_voice_command_signing(item, "manual")
    if success:
        flash('Document signed successfully.')
    else:
        flash(msg)
    return redirect(url_for('index'))

@app.route('/download_signed_document/<int:item_id>', methods=['GET'])
@login_required
def download_signed_document(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Invalid history item.')
        return redirect(url_for('index'))
    if not item.signed:
        flash('Document is not signed.')
        return redirect(url_for('index'))
    signed_filename = item.signed_filename
    if not signed_filename:
        flash('Signed document not found.')
        return redirect(url_for('index'))
    return send_from_directory(directory=DOCUMENTS_FOLDER, path=signed_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
