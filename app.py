# app.py

import os
import uuid
import glob  # Ensure this import is present
import json
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from langdetect import detect
from pydub import AudioSegment
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, Wav2Vec2ForCTC, Wav2Vec2Processor
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
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length
from flask_sqlalchemy import SQLAlchemy

from celery import Celery
from flask_caching import Cache

import soundfile as sf
import librosa

from sqlalchemy.orm import relationship

# New imports for digital signing
from pyhanko.sign import signers
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID
import datetime





from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
import logging






# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize NLTK
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key in production

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CACHE_TYPE'] = 'simple'
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # Redis configuration
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

db = SQLAlchemy(app)
cache = Cache(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize Celery
def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    # Use Flask app context in Celery tasks
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User Authentication
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    # Add relationship to HistoryItems
    history_items = db.relationship('HistoryItem', backref='user', lazy=True)

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
    # Relationship to audios
    audios = db.relationship('Audio', backref='history_item', lazy=True)

class Audio(db.Model):
    __tablename__ = 'audio'
    id = db.Column(db.Integer, primary_key=True)
    history_item_id = db.Column(db.Integer, db.ForeignKey('history_item.id'), nullable=False)
    filename = db.Column(db.String(200))
    description = db.Column(db.String(200))

# Create the database
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    submit = SubmitField('Login')

class EditTextForm(FlaskForm):
    text = TextAreaField('Text', validators=[DataRequired()])
    submit = SubmitField('Save Changes')

# Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DOCUMENTS_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'documents')
IMAGES_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
AUDIO_FOLDER = os.path.join('static', 'audio')
SUMMARIES_FOLDER = os.path.join(BASE_DIR, 'summaries')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
CERT_FOLDER = os.path.join(BASE_DIR, 'certificates')  # New directory for certificates



# Create necessary directories
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(SUMMARIES_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(CERT_FOLDER, exist_ok=True)  # Create certificates directory

# Paths to jsignpdf and keystore
jsignpdf_jar = os.path.join(CERT_FOLDER, 'JSignPdf.jar')  # Adjust as per your directory structure
keystore_path = os.path.join(CERT_FOLDER, 'keystore.p12')  # Path to your keystore
keystore_password = os.getenv('KEYSTORE_PASSWORD')  # Securely retrieve your keystore password
alias = 'myalias'  # Replace with your actual alias

# Set up TTS models
tts_en = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=torch.cuda.is_available())
tts_fr = TTS(model_name="tts_models/fr/mai/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())
loaded_models = {"en": tts_en, "fr": tts_fr}


# Default speakers
default_speakers = {
    "en": "p225",  # Replace with your preferred speaker ID from tts_en.speakers
    # 'fr' model is single-speaker
}

# Load the BLIP image captioning model and processor
print("Loading image captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Image captioning model loaded.")

# Load speech recognition model
print("Loading speech recognition model...")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", ignore_mismatched_sizes=True)
print("Speech recognition model loaded.")

# Initialize inflect engine
p = inflect.engine()

# Load summarization pipeline
print("Loading summarization model...")
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", device=device)
print("Summarization model loaded.")

# Notification messages configuration
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
            speaker = default_speakers.get(message["language"], None)
            if speaker and hasattr(tts_model, 'speakers') and speaker in tts_model.speakers:
                tts_model.tts_to_file(text=message["text"], speaker=speaker, file_path=audio_path)
            else:
                tts_model.tts_to_file(text=message["text"], file_path=audio_path)
    print("Notification audios generated or already exist.")

generate_notification_audios()

def get_unique_id():
    return uuid.uuid4().hex

def preprocess_text(text):
    # Normalize Unicode characters to NFKD form and encode to ASCII to remove accents
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

    # Replace special characters with words
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
        # Add more as needed
    }
    for symbol, word in replacements.items():
        text = text.replace(symbol, word)

    # Remove problematic characters
    text = text.replace("'", '')  # Remove single quotes
    text = text.replace('"', '')  # Remove double quotes if necessary

    # Replace newline characters with period and space to introduce a pause
    text = text.replace('\n', '. ')

    # Remove any remaining non-printable characters
    text = ''.join(c for c in text if c.isprintable())

    # Normalize spaces
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
        # Add more as needed
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
                if block['type'] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            span_text = span["text"]
                            if font_size >= 12:  # Adjust the size threshold as needed
                                text += span_text.upper() + "\n"
                            else:
                                text += span_text + " "
            text += "\n"
        return text
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.style.name.startswith('Heading'):
                full_text.append(paragraph.text.upper())  # Emphasize headings
            else:
                full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Failed to extract text from DOCX: {e}")
        return ""

def extract_text(file_path):
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        elif file_path.endswith(".pdf"):
            # First, try extracting text normally
            text = extract_text_with_headings(file_path)
            if not text.strip():
                # If extraction fails or text is empty, try OCR
                print("Performing OCR on scanned PDF.")
                text = extract_text_from_scanned_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            print("Unsupported file format.")
            return ""

        # Preprocess and normalize the text
        text = preprocess_text(text)
        text = expand_abbreviations(text)
        text = normalize_numbers(text)
        return text

    except Exception as e:
        print(f"Failed to extract text: {e}")
        return ""

def summarize_text(text, max_length=150, min_length=40):
    """
    Generates a summary for the provided text using the BART model.

    Args:
        text (str): The input text to summarize.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    try:
        # The BART model has a maximum token limit; handle accordingly
        # Split text into chunks if it exceeds the limit
        max_chunk_size = 1024  # Adjust based on model's max input size
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

        # Combine summaries if multiple chunks
        final_summary = ' '.join(summaries)
        return final_summary
    except Exception as e:
        print(f"Summarization failed: {e}")
        return "Failed to generate summary."

def generate_image_description(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = blip_processor(image, return_tensors="pt").to(device)
        blip_model.to(device)


        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": 150,
            "num_beams": 5,
            "length_penalty": 1.2,
            "no_repeat_ngram_size": 2,
        }

        # Generate caption with specified parameters
        with torch.no_grad():
            out = blip_model.generate(**inputs, **generation_kwargs)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        return description
    except Exception as e:
        print(f"Error generating image description: {e}")
        return "Failed to generate image description."

def generate_audio_task(text, language, audio_filename):
    """
    Generates an audio file from text using the specified language TTS model.

    Args:
        text (str): The text to convert to speech.
        language (str): The language code ('en' or 'fr').
        audio_filename (str): The filename to save the audio as.

    Returns:
        str: The filename of the generated audio file.
    """
    try:
        print(f"Generating audio for text: {text}")
        tts_model = loaded_models.get(language, tts_en)

        # Split text into chunks to handle large inputs
        max_chunk_size = 500  # Adjust based on TTS model's capabilities
        text_chunks = split_text_into_chunks(text, max_chunk_size)

        audio_segments = []

        for chunk in text_chunks:
            # Generate audio for each chunk
            if hasattr(tts_model, 'speakers') and tts_model.speakers:
                speaker = default_speakers.get(language, tts_model.speakers[0])
                wav = tts_model.tts(chunk, speaker=speaker, progress_bar=False)
            else:
                wav = tts_model.tts(chunk, progress_bar=False)

            # Convert audio data to int16 PCM format
            if isinstance(wav, list) or isinstance(wav, np.ndarray):
                # Assuming wav is a float32 numpy array in range [-1, 1]
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
                # Fallback: Attempt to convert to bytes
                wav_bytes = str(wav).encode('utf-8')

            # Create an AudioSegment for the chunk
            audio_segment = AudioSegment(
                wav_bytes,
                frame_rate=int(tts_model.synthesizer.output_sample_rate),  # Ensure integer
                sample_width=2,  # 16-bit audio
                channels=1
            )

            audio_segments.append(audio_segment)

        # Combine all audio segments into one
        combined_audio = sum(audio_segments)

        # Save audio to file
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

        combined_audio.export(audio_path, format="wav")
        return audio_filename

    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def save_summary(summary_text):
    summary_id = get_unique_id()
    summary_filename = f"{summary_id}.txt"
    summary_path = os.path.join(SUMMARIES_FOLDER, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    return summary_filename

# Define allowed file types
def allowed_file(filename, filetype):
    if filetype == 'document':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx'}
    elif filetype == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    return False

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"  # Default to English if detection fails

def document_needs_signature(text):
    """
    Determine if a document needs to be signed based on its content.
    """
    signature_keywords = ['sign here', 'signature', 'please sign', 'signatory', 'authorized signature', 'sign and date']
    text_lower = text.lower()
    for keyword in signature_keywords:
        if keyword in text_lower:
            return True
    return False

def generate_tts_audio(text, language='en'):
    try:
        tts_model = loaded_models.get(language, tts_en)
        # Check if the model is multi-speaker and set the speaker
        if hasattr(tts_model, 'speakers') and tts_model.speakers:
            speaker = default_speakers.get(language, tts_model.speakers[0])
            wav = tts_model.tts(text, speaker=speaker, progress_bar=False)
            audio_id = get_unique_id()
            audio_filename = f"{audio_id}.wav"
            audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
            tts_model.tts_to_file(text=text, speaker=speaker, file_path=audio_path)
        else:
            # For single-speaker models
            wav = tts_model.tts(text, progress_bar=False)
            audio_id = get_unique_id()
            audio_filename = f"{audio_id}.wav"
            audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
            tts_model.tts_to_file(text=text, file_path=audio_path)
        return audio_filename
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return None

def split_text_into_sections(text, max_section_length=500):
    sentences = split_text_into_sentences(text)
    sections = []
    current_section = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if len(current_section) + len(sentence) + 1 <= max_section_length:
            if current_section:
                current_section += " " + sentence
            else:
                current_section = sentence
        else:
            if current_section:
                sections.append(current_section)
            current_section = sentence

    if current_section:
        sections.append(current_section)

    return sections

# Function to generate a self-signed certificate
def generate_self_signed_cert(cert_file_path, key_file_path):
    # Check if certificate already exists
    if os.path.exists(cert_file_path) and os.path.exists(key_file_path):
        return

    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Write private key to file
    with open(key_file_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Create a self-signed certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),  # Adjust as needed
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
        datetime.datetime.utcnow() + datetime.timedelta(days=3650)  # Valid for 10 years
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
        critical=False,
    ).sign(key, hashes.SHA256(), default_backend())

    # Write certificate to file
    with open(cert_file_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


# Configure logginglogging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_digital_signature(input_pdf_path, output_pdf_path, keystore_path, keystore_password, alias, jsignpdf_jar_path):
    """
    Applies a digital signature to a PDF document using jsignpdf via subprocess.
    """
    try:
        # Validate required inputs
        if not all([input_pdf_path, output_pdf_path, keystore_path, alias, jsignpdf_jar_path]):
            logger.error(f"One or more inputs are None: input={input_pdf_path}, output={output_pdf_path}, "
                         f"keystore={keystore_path}, alias={alias}, jar={jsignpdf_jar_path}")
            return False

        # Default keystore password to an empty string if None
        if keystore_password is None:
            keystore_password = "123"

        # Ensure files exist
        if not os.path.exists(input_pdf_path):
            logger.error(f"Input PDF not found: {input_pdf_path}")
            return False
        if not os.path.exists(keystore_path):
            logger.error(f"Keystore not found: {keystore_path}")
            return False
        if not os.path.exists(jsignpdf_jar_path):
            logger.error(f"jsignpdf JAR not found: {jsignpdf_jar_path}")
            return False

        # Construct the command
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

        # Log the command
        logger.info(f"Running command: {' '.join(command)}")

        # Execute the command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check for errors
        if result.returncode != 0:
            logger.error(f"jsignpdf error: {result.stderr}")
            return False

        logger.info("PDF signed successfully.")
        return True

    except Exception as e:
        logger.exception(f"Exception during PDF signing: {e}")
        return False


from pydub import AudioSegment
import io

# Routes

@app.route('/')
@login_required
def index():
    try:
        history = current_user.history_items
        return render_template('index.html', history=history, notification_messages=notification_messages)
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return "An error occurred while loading the page.", 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        # In a real app, you should verify the username and password
        user = User.query.filter_by(username=username).first()
        if not user:
            # Create a new user for simplicity
            user = User(username=username)
            db.session.add(user)
            db.session.commit()
        login_user(user)
        return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload_document', methods=['POST'])
@login_required
def upload_document():
    if 'document' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))
    file = request.files['document']
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename, 'document'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(DOCUMENTS_FOLDER, filename)
        file.save(file_path)

        # Extract text
        text = extract_text(file_path)
        print(f"Extracted text: {text}")
        if not text:
            flash('Failed to extract text from the document.')
            return redirect(url_for('index'))

        # Check if document needs signature
        needs_signature = document_needs_signature(text)

        # Create new document in the database
        new_document = HistoryItem(
            user_id=current_user.id,
            type='Document',
            text=text,
            filename=filename,
            needs_signature=needs_signature
        )
        db.session.add(new_document)
        db.session.commit()

        # Flash "item_added" message
        flash('item_added')

        return redirect(url_for('index'))
    else:
        flash('Unsupported file format.')
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
        db.session.commit()
        flash('Text updated successfully.')
        return redirect(url_for('index'))
    form.text.data = item.text
    return render_template('edit_text.html', form=form, item_id=item_id)

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    if 'image' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash('No selected file.')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename, 'image'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(IMAGES_FOLDER, filename)
        file.save(file_path)

        # Generate image description
        description = generate_image_description(file_path)

        # Check if audio already exists
        existing_audio = Audio.query.join(HistoryItem).filter(
            HistoryItem.user_id == current_user.id,
            HistoryItem.type == 'Image',
            HistoryItem.filename == filename,
            Audio.description == f"Audio of Description of {filename}"
        ).first()

        if not existing_audio:
            # Generate audio for the description
            language = detect_language(description)
            description_audio_filename = get_unique_id() + '.wav'
            generate_audio_task(description, language, description_audio_filename)
        else:
            description_audio_filename = existing_audio.filename

        # Add to history
        new_image = HistoryItem(
            user_id=current_user.id,
            type='Image',
            description=description,
            filename=filename
        )
        db.session.add(new_image)
        db.session.commit()

        # Add audio to the database
        if description_audio_filename:
            description_audio = Audio(
                history_item_id=new_image.id,
                filename=description_audio_filename,
                description=f"Audio of Description of {filename}"
            )
            db.session.add(description_audio)
            db.session.commit()

        # Flash "item_added" message
        flash('item_added')

        return redirect(url_for('index'))
    else:
        flash('Unsupported file format.')
        return redirect(url_for('index'))

@app.route('/summarize_document/<int:item_id>', methods=['POST'])
@login_required
def summarize_document_route(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Invalid history item.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Invalid history item.'}), 400
        return redirect(url_for('index'))
    if item.type != "Document":
        flash('Selected item is not a document.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Selected item is not a document.'}), 400
        return redirect(url_for('index'))
    text = item.text
    if not text:
        flash('No text available to summarize.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'No text available to summarize.'}), 400
        return redirect(url_for('index'))

    # Check if summary already exists
    existing_summary = HistoryItem.query.filter_by(
        user_id=current_user.id,
        type='Summary',
        description=f"Summary of {item.filename}"
    ).first()

    if existing_summary:
        flash('Summary has already been generated for this document.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'message': 'Summary already exists.'}), 200
        return redirect(url_for('index'))

    summary = summarize_text(text)
    if summary:
        # Create summary item in history
        new_summary = HistoryItem(
            user_id=current_user.id,
            type='Summary',
            text=summary,
            description=f"Summary of {item.filename}"
        )
        db.session.add(new_summary)
        db.session.commit()

        # Check if audio already exists
        existing_audio = Audio.query.filter_by(
            history_item_id=new_summary.id,
            description=f"Audio of Summary of {item.filename}"
        ).first()

        if not existing_audio:
            # Generate audio for the summary
            language = detect_language(summary)
            summary_audio_filename = get_unique_id() + '.wav'
            generate_audio_task(summary, language, summary_audio_filename)

            # Add audio to the database
            if summary_audio_filename:
                summary_audio = Audio(
                    history_item_id=new_summary.id,
                    filename=summary_audio_filename,
                    description=f"Audio of Summary of {item.filename}"
                )
                db.session.add(summary_audio)
                db.session.commit()
        else:
            summary_audio_filename = existing_audio.filename

        flash('summarizing_done')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'message': 'Summarization done.'}), 200

        return redirect(url_for('index'))
    else:
        flash('Failed to generate summary.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Failed to generate summary.'}), 500
        return redirect(url_for('index'))

@app.route('/generate_audio/<int:item_id>', methods=['POST'])
@login_required
def generate_audio_route(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Invalid history item.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Invalid history item.'}), 400
        return redirect(url_for('index'))

    # Check if audio already exists
    existing_audio = Audio.query.filter_by(
        history_item_id=item.id,
        description=f"Audio of {item.filename or item.description}"
    ).first()

    if existing_audio:
        flash('Audio has already been generated for this item.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'message': 'Audio already exists.'}), 200
        return redirect(url_for('index'))

    text = item.text or item.description
    language = detect_language(text)
    print(f"Detected language: {language}")
    print(f"Text to convert to audio: {text}")

    audio_filename = get_unique_id() + '.wav'
    generate_audio_task(text, language, audio_filename)
    if audio_filename:
        # Add audio filename to database
        audio_entry = Audio(
            history_item_id=item.id,
            filename=audio_filename,
            description=f"Audio of {item.filename or item.description}"
        )
        db.session.add(audio_entry)
        db.session.commit()

        # Flash 'audio_generated' message
        flash('audio_generated')

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'message': 'Audio generation started.'}), 200

        return redirect(url_for('index'))
    else:
        flash('Failed to generate audio.')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Failed to generate audio.'}), 500
        return redirect(url_for('index'))

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

    # Check if document needs signature
    needs_signature = document_needs_signature(text)
    if not needs_signature:
        flash('This document does not require a signature.')
        return redirect(url_for('index'))

    return render_template('voice_sign_document.html', item=item)

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

    # Start the voice-guided signing process
    # Generate the initial audio prompt
    prompt_text = "Do you wish to sign the document? Say 'Yes' to confirm or 'No' to cancel."
    audio_filename = generate_tts_audio(prompt_text, language='en')

    if not audio_filename:
        return jsonify({'success': False, 'message': 'Failed to generate audio prompt.'}), 500

    audio_url = url_for('static', filename='audio/' + audio_filename)

    # Save the current state in the session
    session['voice_signing'] = {
        'item_id': item_id,
        'state': 'ask_sign'
    }

    return jsonify({'success': True, 'audio_url': audio_url})

@app.route('/process_voice_command/<int:item_id>', methods=['POST'])
@login_required
def process_voice_command(item_id):
    # Get the current voice signing state from the session
    voice_signing = session.get('voice_signing', {})
    if not voice_signing or voice_signing.get('item_id') != item_id:
        return jsonify({'success': False, 'message': 'Voice signing session not found.'}), 400

    # Get the audio data from the request
    if 'audio_data' not in request.files:
        return jsonify({'success': False, 'message': 'No audio data provided.'}), 400

    audio_file = request.files['audio_data']
    audio_data = audio_file.read()
    mime_type = request.form.get('mime_type', 'audio/webm')

    # Convert audio data to WAV format using pydub
    try:
        format = mime_type.split('/')[1].split(';')[0]  # Extract format from MIME type
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=format)

        # Export the AudioSegment to WAV format in memory
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)

        # Read the WAV data using soundfile
        speech, sample_rate = sf.read(wav_io)
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return jsonify({'success': False, 'message': 'Failed to process audio.'}), 500

    # Perform speech recognition on the audio data
    try:
        # Resample if necessary
        if sample_rate != 16000:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_values = asr_processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
        asr_model.to(device)
        with torch.no_grad():
            logits = asr_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_processor.decode(predicted_ids[0])

        # Process the transcription
        response_text = transcription.lower()
        print(f"Recognized speech: {response_text}")

    except Exception as e:
        print(f"Error during speech recognition: {e}")
        return jsonify({'success': False, 'message': 'Failed to process audio.'}), 500

    # Now, handle the user's response based on the current state
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        return jsonify({'success': False, 'message': 'Invalid history item.'}), 400

    current_state = voice_signing.get('state')

    if current_state == 'ask_sign':
        if 'yes' in response_text:
            # Proceed to sign the document
            # Get the document file path
            document_filename = item.filename
            document_path = os.path.join(DOCUMENTS_FOLDER, document_filename)

            if not os.path.exists(document_path):
                print(f"Document not found: {document_path}")
                return jsonify({'success': False, 'message': 'Document not found.'}), 404

            # Check if the document is a PDF
            if not document_path.lower().endswith('.pdf'):
                print("Only PDF documents can be signed.")
                prompt_text = "Only PDF documents can be signed."
                audio_filename = generate_tts_audio(prompt_text)
                audio_url = url_for('static', filename='audio/' + audio_filename)
                return jsonify({'success': True, 'action': 'error', 'audio_url': audio_url})

            # Generate or use existing self-signed certificate
            cert_file_path = os.path.join(CERT_FOLDER, 'cert.pem')
            key_file_path = os.path.join(CERT_FOLDER, 'key.pem')

            generate_self_signed_cert(cert_file_path, key_file_path)

            # Apply digital signature
            # Extract the base name and extension of the original filename
            base_name, extension = document_filename.rsplit('.', 1)


            # Ensure the signed filename is correctly prefixed
            if not base_name.startswith("signed_"):
                signed_document_filename = f"signed_{base_name}{extension}"
            else:
                signed_document_filename = f"{base_name}{extension}"  # Avoid adding "signed_" twice

            
            signed_document_path = os.path.join(DOCUMENTS_FOLDER, signed_document_filename)

            # Apply digital signature using jsignpdf
            success = apply_digital_signature(
                input_pdf_path=document_path,
                output_pdf_path=signed_document_path,
                keystore_path=keystore_path,
                keystore_password=keystore_password,
                alias=alias,
                jsignpdf_jar_path=jsignpdf_jar
            )

            if success:
                print(f"Signed document created in: {DOCUMENTS_FOLDER}")
                
                try:
                    # Rename the file using the updated logic
                    new_signed_filepath = rename_file_to_clean_pattern(DOCUMENTS_FOLDER)
                    new_signed_filename = os.path.basename(new_signed_filepath)
                except FileNotFoundError as e:
                    logger.error(f"Error during renaming: {e}")
                    return jsonify({'success': False, 'message': str(e)}), 404
                
                # Update the database record
                item.signed = True
                item.signed_filename = new_signed_filename
                db.session.commit()

                prompt_text = "The document has been signed and renamed."
                audio_filename = generate_tts_audio(prompt_text)
                audio_url = url_for('static', filename='audio/' + audio_filename)
                return jsonify({'success': True, 'action': 'signed', 'audio_url': audio_url})




        elif 'no' in response_text:
            # Cancel signing
            voice_signing.clear()
            session['voice_signing'] = voice_signing
            prompt_text = "Signing cancelled."
            audio_filename = generate_tts_audio(prompt_text)
            if not audio_filename:
                return jsonify({'success': False, 'message': 'Failed to generate audio prompt.'}), 500
            audio_url = url_for('static', filename='audio/' + audio_filename)
            return jsonify({'success': True, 'action': 'cancelled', 'audio_url': audio_url})
        else:
            prompt_text = "Please say 'Yes' to sign or 'No' to cancel."
            audio_filename = generate_tts_audio(prompt_text)
            if not audio_filename:
                return jsonify({'success': False, 'message': 'Failed to generate audio prompt.'}), 500
            audio_url = url_for('static', filename='audio/' + audio_filename)
            return jsonify({'success': True, 'action': 'ask_sign', 'audio_url': audio_url})
    else:
        # Invalid state
        voice_signing.clear()
        session['voice_signing'] = voice_signing
        return jsonify({'success': False, 'message': 'Invalid state.'}), 500
    

def clean_signed_filename(original_filename):
    """
    Cleans up the signed document filename based on a specific pattern.

    Args:
        original_filename (str): The original signed document filename.

    Returns:
        str: The cleaned-up signed filename in the format signed_<original_name>.pdf.
    """
    # Extract the base name and extension
    base_name, extension = os.path.splitext(original_filename)
    
    # Ensure double extensions are handled (e.g., .pdf.pdf)
    if extension == ".pdf":
        base_name, second_extension = os.path.splitext(base_name)
        if second_extension == ".pdf":
            extension = ".pdf"  # Retain single .pdf
    
    # Look for the pattern *_signed_* and clean it
    if "_signed_" in base_name:
        original_name = base_name.split("_signed_")[0]  # Extract the part before _signed_
        new_filename = f"signed_{original_name}.pdf"
        return new_filename

    # If no pattern found, just prefix with "signed_"
    return f"signed_{base_name}.pdf"

import os

import glob
import os

import glob
import os

def rename_file_to_clean_pattern(directory):
    """
    Searches for a file containing 'signed' in its name within a directory,
    then renames it to a clean pattern.

    Args:
        directory (str): The directory to search for the signed file.

    Returns:
        str: The new file path after renaming.
    """
    # Search for files containing 'signed' in their name
    signed_files = glob.glob(os.path.join(directory, "*signed*"))
    
    if not signed_files:
        raise FileNotFoundError(f"No file containing 'signed' found in directory: {directory}")
    
    # Assume the first match is the file we need to rename
    original_filepath = signed_files[0]
    
    # Extract the base name and extension
    base_name, extension = os.path.splitext(os.path.basename(original_filepath))
    
    # Check and remove extra `.pdf` in base_name
    if extension == ".pdf" and base_name.endswith(".pdf"):
        base_name = base_name[:-4]  # Remove the extra `.pdf`
    
    # Construct the new filename
    cleaned_filename = f"signed_{base_name.replace('signed_', '').strip('_')}{extension}"
    new_filepath = os.path.join(directory, cleaned_filename)
    
    # Log for debugging
    print(f"Renaming file:\n  From: {original_filepath}\n  To: {new_filepath}")
    
    # Rename the file
    os.rename(original_filepath, new_filepath)
    return new_filepath



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

@app.route('/delete_history_item/<int:item_id>', methods=['POST'])
@login_required
def delete_history_item(item_id):
    item = HistoryItem.query.filter_by(id=item_id, user_id=current_user.id).first()
    if not item:
        flash('Invalid history item.')
        return redirect(url_for('index'))

    # Delete associated files
    if item.type == "Summary":
        # Optionally, delete the summary file if stored
        pass  # Implement if necessary
    elif item.type == "Document":
        # Delete the document file
        document_path = os.path.join(DOCUMENTS_FOLDER, item.filename)
        if os.path.exists(document_path):
            os.remove(document_path)
        # Delete signed document if exists
        if item.signed_filename:
            signed_document_path = os.path.join(DOCUMENTS_FOLDER, item.signed_filename)
            if os.path.exists(signed_document_path):
                os.remove(signed_document_path)
    elif item.type == "Image":
        # Delete the image file
        image_path = os.path.join(IMAGES_FOLDER, item.filename)
        if os.path.exists(image_path):
            os.remove(image_path)

    # Delete associated audios
    for audio in item.audios:
        audio_path = os.path.join(AUDIO_FOLDER, audio.filename)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        db.session.delete(audio)

    db.session.delete(item)
    db.session.commit()

    # Flash a single "item_deleted" message
    flash('item_deleted')

    return redirect(url_for('index'))

@app.route('/help')
def help_route():
    return render_template('help.html')

@app.route('/about')
def about_route():
    return render_template('about.html')

# Run the app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

