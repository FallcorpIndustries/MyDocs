# app.py

import os
import uuid
import json
import hashlib
import re
import unicodedata
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
from werkzeug.utils import secure_filename
from langdetect import detect
from pydub import AudioSegment
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from TTS.api import TTS
import nltk
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from docx import Document
import inflect
import pytesseract
from PIL import Image
import numpy as np

# Explicitly set the environment variable TOKENIZERS_PARALLELISM to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize NLTK
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DOCUMENTS_FOLDER = os.path.join(UPLOAD_FOLDER, 'documents')
IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'static', 'audio')
SUMMARIES_FOLDER = os.path.join(BASE_DIR, 'summaries')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')




# Create necessary directories
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(SUMMARIES_FOLDER, exist_ok=True)

# Set up TTS models
print("Loading TTS models...")
tts_en = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
tts_fr = TTS(model_name="tts_models/fr/mai/tacotron2-DDC", progress_bar=False, gpu=False)
loaded_models = {"en": tts_en, "fr": tts_fr}

# Default speakers
default_speakers = {
    "en": "p225",  # Replace with your preferred speaker ID from tts_en.speakers
    # 'fr' is single-speaker
}

# Load the BLIP image captioning model and processor
print("Loading image captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Image captioning model loaded.")

# Initialize inflect engine
p = inflect.engine()

# Load summarization pipeline
print("Loading summarization model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
print("Summarization model loaded.")

# Notification messages configuration
notification_messages = {
    "analyzing_document": {
        "text": "Analyzing Document",
        "filename": "analyzing_document.wav",
        "language": "en"
    },
    "analyzing_done": {
        "text": "Analyzing done you can press play to read the document",
        "filename": "analyzing_done.wav",
        "language": "en"
    },
    "summarizing_document": {
        "text": "Summarizing Document",
        "filename": "summarizing_document.wav",
        "language": "en"
    },
    "summarizing_done": {
        "text": "Summarizing done you can press play to read the summary",
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
    }
}

def generate_notification_audios():
    for key, message in notification_messages.items():
        audio_path = os.path.join(AUDIO_FOLDER, message["filename"])
        if not os.path.exists(audio_path):
            tts_model = loaded_models.get(message["language"], tts_en)
            speaker = default_speakers.get(message["language"], None)
            if speaker and hasattr(tts_model, 'speakers') and speaker in tts_model.speakers:
                wav = tts_model.tts(message["text"], speaker=speaker, progress_bar=False)
            else:
                wav = tts_model.tts(message["text"], progress_bar=False)
            
            # Convert and export the audio
            try:
                if isinstance(wav, list) or isinstance(wav, np.ndarray):
                    wav = np.array(wav)
                    wav = np.clip(wav, -1.0, 1.0)
                    wav = (wav * 32767).astype(np.int16)
                    wav_bytes = wav.tobytes()
                    audio_segment = AudioSegment(
                        wav_bytes,
                        frame_rate=int(tts_model.synthesizer.output_sample_rate),
                        sample_width=2,
                        channels=1
                    )
                    audio_segment.export(audio_path, format="wav")
                elif isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                    wav = np.clip(wav, -1.0, 1.0)
                    wav = (wav * 32767).astype(np.int16)
                    wav_bytes = wav.tobytes()
                    audio_segment = AudioSegment(
                        wav_bytes,
                        frame_rate=int(tts_model.synthesizer.output_sample_rate),
                        sample_width=2,
                        channels=1
                    )
                    audio_segment.export(audio_path, format="wav")
                elif isinstance(wav, bytes):
                    with open(audio_path, 'wb') as f:
                        f.write(wav)
                else:
                    print(f"Unsupported audio format for message '{key}'.")
            except Exception as e:
                print(f"Error generating audio for message '{key}': {e}")
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

def split_text_into_chunks(text, max_chunk_size=1000):
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
        inputs = blip_processor(image, return_tensors="pt")

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

def generate_audio(text, language="en"):
    """
    Generates an audio file from text using the specified language TTS model.

    Args:
        text (str): The text to convert to speech.
        language (str): The language code ('en' or 'fr').

    Returns:
        str: The filename of the generated audio file.
    """
    try:
        print(f"Generating audio for text: {text}")
        tts_model = loaded_models.get(language, tts_en)

        # Split text into chunks to handle large inputs
        max_chunk_size = 1000  # Adjust based on TTS model's capabilities
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
                wav = np.clip(wav, -1.0, 1.0)  # Ensure values are in [-1, 1]
                wav = (wav * 32767).astype(np.int16)  # Convert to int16
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
        audio_id = get_unique_id()
        audio_filename = f"{audio_id}.wav"
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

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    return history

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def get_file_id(file_path):
    # Use the absolute path to generate a unique hash
    abs_path = os.path.abspath(file_path)
    return hashlib.md5(abs_path.encode('utf-8')).hexdigest()

# Routes

@app.route('/')
def index():
    try:
        history = load_history()
        return render_template('index.html', history=history, notification_messages=notification_messages)
    except Exception as e:
        print(f"Error loading index page: {e}")
        return "An error occurred while loading the page.", 500

@app.route('/upload_document', methods=['POST'])
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

        # Add to history
        history = load_history()
        history.append(['Document', text, filename])
        save_history(history)

        # Flash "item_added" message
        flash('item_added')

        return redirect(url_for('index'))
    else:
        flash('Unsupported file format.')
        return redirect(url_for('index'))

@app.route('/upload_image', methods=['POST'])
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

        # Add to history
        history = load_history()
        history.append(['Image', description, filename])
        save_history(history)

        # Flash "item_added" message
        flash('item_added')

        return redirect(url_for('index'))
    else:
        flash('Unsupported file format.')
        return redirect(url_for('index'))

@app.route('/summarize_document/<int:index>', methods=['POST'])
def summarize_document_route(index):
    history = load_history()
    if index < 0 or index >= len(history):
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Invalid history index.'}), 400
        else:
            flash('Invalid history index.')
            return redirect(url_for('index'))
    item = history[index]
    if item[0] != 'Document':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Selected item is not a document.'}), 400
        else:
            flash('Selected item is not a document.')
            return redirect(url_for('index'))
    text = item[1]
    if not text:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'No text available to summarize.'}), 400
        else:
            flash('No text available to summarize.')
            return redirect(url_for('index'))
    
    # Check if request is AJAX
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Handle AJAX request
        summary = summarize_text(text)
        if summary:
            summary_filename = save_summary(summary)
            history.append(['Summary', summary, f"Summary of {item[2]}"])
            save_history(history)

            return jsonify({'success': True, 'message': 'Summarization completed successfully.'}), 200
        else:
            return jsonify({'success': False, 'message': 'Failed to generate summary.'}), 500
    else:
        # Handle non-AJAX request
        # Flash "summarizing_document" message
        flash('summarizing_document')

        summary = summarize_text(text)
        if summary:
            summary_filename = save_summary(summary)
            history.append(['Summary', summary, f"Summary of {item[2]}"])
            save_history(history)

            # Flash "summarizing_done" and "item_added" messages
            flash('summarizing_done')
            flash('item_added')

            flash('Summarization completed successfully.')
        else:
            flash('Failed to generate summary.')
            flash('summarizing_done')

        return redirect(url_for('index'))

@app.route('/generate_audio/<int:index>', methods=['POST'])
def generate_audio_route(index):
    history = load_history()
    if index < 0 or index >= len(history):
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Invalid history index.'}), 400
        else:
            flash('Invalid history index.')
            return redirect(url_for('index'))
    item = history[index]
    if item[0] not in ['Document', 'Summary']:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Selected item cannot be converted to audio.'}), 400
        else:
            flash('Selected item cannot be converted to audio.')
            return redirect(url_for('index'))
    text = item[1]
    language = detect_language(text)
    print(f"Detected language: {language}")
    print(f"Text to convert to audio: {text}")

    # Check if request is AJAX
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Handle AJAX request
        audio_filename = generate_audio(text, language)
        if audio_filename:
            # Add audio filename to history
            history.append(['Audio', audio_filename, f"Audio of {item[2]}"])
            save_history(history)

            return jsonify({'success': True, 'message': 'Audio generated successfully.'}), 200
        else:
            return jsonify({'success': False, 'message': 'Failed to generate audio.'}), 500
    else:
        # Handle non-AJAX request
        # Flash "analyzing_document" message
        flash('analyzing_document')

        audio_filename = generate_audio(text, language)
        if audio_filename:
            # Add audio filename to history
            history.append(['Audio', audio_filename, f"Audio of {item[2]}"])
            save_history(history)

            # Flash "analyzing_done" and "item_added" messages
            flash('analyzing_done')
            flash('item_added')

            flash('Audio generated successfully.')
            return redirect(url_for('index'))
        else:
            flash('Failed to generate audio.')
            flash('analyzing_done')
            return redirect(url_for('index'))

@app.route('/delete_history_item/<int:index>', methods=['POST'])
def delete_history_item(index):
    history = load_history()
    if index < 0 or index >= len(history):
        flash('Invalid history index.')
        return redirect(url_for('index'))
    item = history.pop(index)
    save_history(history)

    # Delete associated files
    if item[0] == 'Summary':
        summary_text = item[1]
        # Find and delete the summary file
        for summary_file in os.listdir(SUMMARIES_FOLDER):
            summary_path = os.path.join(SUMMARIES_FOLDER, summary_file)
            with open(summary_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content == summary_text:
                    os.remove(summary_path)
                    break
    elif item[0] == 'Document':
        # Delete cached audio file
        file_id = get_file_id(item[1])
        audio_file_path = os.path.join(AUDIO_FOLDER, f"{file_id}.wav")
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
    elif item[0] == 'Image':
        # Delete the image file
        image_path = os.path.join(IMAGES_FOLDER, item[2])
        if os.path.exists(image_path):
            os.remove(image_path)
    elif item[0] == 'Audio':
        # Delete the audio file
        audio_path = os.path.join(AUDIO_FOLDER, item[1])
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # Flash "item_deleted" message
    flash('item_deleted')

    flash('History item deleted successfully.')
    return redirect(url_for('index'))

@app.route('/help')
def help_route():
    return render_template('help.html')

@app.route('/about')
def about_route():
    return render_template('about.html')

# Utility Functions

def allowed_file(filename, filetype):
    if filetype == 'document':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx'}
    elif filetype == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    return False

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"  # Default to English if detection fails

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)