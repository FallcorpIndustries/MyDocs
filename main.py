import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import re
import json
import hashlib
from langdetect import detect
import soundfile as sf
from threading import Thread
from queue import Queue, SimpleQueue
import numpy as np
import nltk
from pydub import AudioSegment
import torch
import pygame
from PIL import Image, ImageTk
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from docx import Document
import inflect
import unicodedata
import uuid

# Unicode Garbage Bin Emoji
delete_icon = "🗑️"

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.set_num_channels(8)  # Ensure we have enough channels

# Initialize a separate channel for the waiting song
waiting_song_channel = pygame.mixer.Channel(1)  # Use channel 1 for the waiting song

# Download the punkt tokenizer for sentence splitting if not already downloaded
nltk.download('punkt', quiet=True)

# Base path for resources
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (PyInstaller)
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Paths for models and data
models_dir = os.path.join(base_path, 'models')
tts_models_dir = os.path.join(models_dir, 'tts_models')
transformers_cache_dir = os.path.join(models_dir, 'transformers')
audio_cache_dir = os.path.join(base_path, "audio_cache")
summary_cache_dir = os.path.join(base_path, "summary_cache")
waiting_song_path = os.path.join(base_path, 'waiting_song.wav')

# Create necessary directories
os.makedirs(tts_models_dir, exist_ok=True)
os.makedirs(transformers_cache_dir, exist_ok=True)
os.makedirs(audio_cache_dir, exist_ok=True)
os.makedirs(summary_cache_dir, exist_ok=True)

# Set environment variables before imports
os.environ['TORCH_HOME'] = tts_models_dir
os.environ['TRANSFORMERS_CACHE'] = transformers_cache_dir

# Now import libraries that use the environment variables
from TTS.api import TTS
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Load models for English and French
print("Loading TTS models...")
tts_en = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
tts_fr = TTS(model_name="tts_models/fr/mai/tacotron2-DDC", progress_bar=False, gpu=False)
loaded_models = {"en": tts_en, "fr": tts_fr}

# Default voices
default_speakers = {
    "en": "p225",  # Replace with your preferred speaker ID from tts_en.speakers
    # No need for 'fr' since the French model is single-speaker
}

# Load the BLIP image captioning model and processor
print("Loading image captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Image captioning model loaded.")

# Disable tokenizers parallelism to prevent warnings after forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize summarization pipeline
try:
    print("Loading BART summarization model...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    print("BART summarization model loaded successfully.")
except Exception as e:
    print(f"Failed to load BART summarization model: {e}")
    summarizer = None  # Handle gracefully if model loading fails

def get_unique_id():
    return uuid.uuid4().hex

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
    if not summarizer:
        speak_message("Summarization model is not available.", "en")
        return "Summarization model is unavailable."

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
        speak_message("An error occurred while summarizing the document.", "en")
        return "Failed to generate summary."

def summarize_document():
    """
    Initiates the summarization process for the selected document.
    """
    selected_item = history_list.focus()
    if selected_item:
        item_index = history_list.index(selected_item)
        item = history[item_index]
        item_type, data, name = item
        if item_type == 'Document':
            file_path = data  # Full path
            document_name = name
            # Extract text
            text = extract_text(file_path)
            if text:
                # Start summarization in a separate thread to keep the GUI responsive
                Thread(target=lambda: generate_and_display_summary(text, document_name)).start()
            else:
                speak_message("Failed to extract text from the selected document.", "en")
        else:
            speak_message("Please select a document to summarize.", "en")
    else:
        speak_message("No file selected. Please select a document to summarize.", "en")

def generate_and_display_summary(text, document_name):
    try:
        speak_message("Generating summary. Please wait.", "en")
        # Start waiting song
        start_waiting_song()
        summary = summarize_text(text)
    except Exception as e:
        print(f"Error during summarization: {e}")
        summary = None
    finally:
        # Stop waiting song
        stop_waiting_song()

    if summary:
        display_summary(summary)
        speak_message("Summary generated.", "en")
        # Inform the user that the summary will be read aloud
        speak_message("Reading the summary now.", "en")
        # Read the summary aloud sequentially
        speak_large_text(summary, language="en")

        # Save the summary to history with a meaningful name
        summary_name = f"Summary of {document_name}"
        history.append(['Summary', summary, summary_name])
        history_list.insert('', 'end', values=('Summary', summary_name, delete_icon))

        # Optionally, save the summary to a text file
        save_summary_to_file(summary)

    else:
        display_text("Failed to generate summary.")
        speak_message("Failed to generate summary.", "en")

def save_summary_to_file(summary_text):
    # Generate a unique file name
    summary_file_path = os.path.join(summary_cache_dir, f"summary_{get_unique_id()}.txt")
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    return summary_file_path

def speak_large_text(text, language="en", chunk_size=200):
    """
    Splits the text into smaller chunks and reads each chunk aloud.

    Args:
        text (str): The text to read.
        language (str): The language code for TTS.
        chunk_size (int): The maximum number of characters per chunk.
    """
    # Stop any existing playback
    audio_player.stop()

    sentences = split_text_into_sentences(text)
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            speak_message(current_chunk, language=language)
            current_chunk = sentence

    if current_chunk:
        speak_message(current_chunk, language=language)

def split_text_into_chunks(text, max_chunk_size=1000):
    """
    Splits text into chunks not exceeding the specified maximum size.

    Args:
        text (str): The input text to split.
        max_chunk_size (int): The maximum number of characters per chunk.

    Returns:
        list: A list of text chunks.
    """
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

def display_summary(summary):
    """
    Displays the summary in the preview area with a distinguishing label.

    Args:
        summary (str): The summary text to display.
    """
    preview_label.config(text="Summary:\n" + summary, image='')

# Helper function to detect language
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"  # Default to English if detection fails

# Message queue and flags for audio feedback
message_queue = SimpleQueue()
is_speaking = False
last_message = ""

# Function to speak messages for audio feedback
def speak_message(message, language="en"):
    global last_message
    last_message = message
    message_queue.put((message, language))
    Thread(target=process_message_queue).start()

def process_message_queue():
    global is_speaking
    if is_speaking:
        return
    is_speaking = True
    while not message_queue.empty():
        message, language = message_queue.get()
        tts_model = loaded_models.get(language, loaded_models["en"])

        # Stop the waiting song before playing the message
        stop_waiting_song()

        # Check if model is multi-speaker
        if getattr(tts_model, 'is_multi_speaker', False):
            # Get the default speaker from the dictionary
            speaker = default_speakers.get(language)
            # If no default speaker specified or not found, use the first speaker in the list
            if not speaker or speaker not in tts_model.speakers:
                speaker = tts_model.speakers[0]
                print(f"Speaker '{speaker}' not found or not specified. Using default speaker '{speaker}'.")
            else:
                print(f"Using speaker: {speaker}")

            wav = tts_model.tts(message, speaker=speaker, progress_bar=False, gpu=False)
        else:
            # Single-speaker model
            wav = tts_model.tts(message, progress_bar=False, gpu=False)

        sample_rate = tts_model.synthesizer.output_sample_rate

        # Convert 'wav' to a NumPy array if it's not already one
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)

        # Scale the waveform to int16 for 'pygame' playback
        wav_int16 = (wav * 32767).astype(np.int16)

        # Save the audio to a temporary file
        temp_audio_path = "temp_message.wav"
        sf.write(temp_audio_path, wav_int16, sample_rate)

        # Play the audio using pygame
        audio_player.play(temp_audio_path)
    is_speaking = False

def repeat_last_message():
    if last_message:
        Thread(target=lambda: speak_message(last_message)).start()

# Function to start the waiting song
def start_waiting_song():
    if os.path.exists(waiting_song_path):
        waiting_song = pygame.mixer.Sound(waiting_song_path)
        waiting_song_channel.play(waiting_song, loops=-1)  # Play in a loop
    else:
        print("Waiting song file not found.")

# Function to stop the waiting song
def stop_waiting_song():
    waiting_song_channel.stop()

# Function to preprocess text
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

# Function to expand abbreviations
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

# Function to normalize numbers
def normalize_numbers(text):
    p = inflect.engine()
    words = text.split()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Function to split text into sentences
def split_text_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    return sentences

# Function to generate audio for a chunk
def generate_audio_for_chunk(chunk, language, output_path):
    try:
        tts_model = loaded_models.get(language, loaded_models["en"])  # Default to English
        chunk = preprocess_text(chunk)  # Preprocess text

        if getattr(tts_model, 'is_multi_speaker', False):
            # Get the default speaker from the dictionary
            speaker = default_speakers.get(language)
            # If no default speaker specified or not found, use the first speaker in the list
            if not speaker or speaker not in tts_model.speakers:
                speaker = tts_model.speakers[0]
                print(f"Speaker '{speaker}' not found or not specified. Using default speaker '{speaker}'.")
            else:
                print(f"Using speaker: {speaker}")

            wav = tts_model.tts(chunk, speaker=speaker, progress_bar=False, gpu=False)
        else:
            # Single-speaker model
            wav = tts_model.tts(chunk, progress_bar=False, gpu=False)

        sample_rate = tts_model.synthesizer.output_sample_rate
        sf.write(output_path, wav, sample_rate, format="WAV")
        return output_path
    except Exception as e:
        print(f"Error generating audio for chunk:\n{chunk}\n{e}")
        return None

# Function to generate and play audio in real-time
def generate_and_play_audio(text, audio_file_path, callback=None):
    chunks = split_text_into_chunks(text)
    if not chunks:
        speak_message("No valid text found to process.")
        if callback:
            callback()
        return False

    def generate_chunks():
        for i, chunk in enumerate(chunks):
            language = detect_language(chunk)
            if language not in loaded_models:
                language = "en"  # Default to English if language is not supported
            output_path = f"temp_chunk_{i}.wav"
            print(f"Processing chunk {i + 1}/{len(chunks)}...")
            audio_file = generate_audio_for_chunk(chunk, language, output_path)
            if audio_file:
                yield AudioSegment.from_wav(audio_file)
                os.remove(output_path)
            else:
                continue

    # Create a queue to hold audio segments
    audio_queue = Queue()

    # Function to generate audio and put into queue
    def producer():
        for audio_segment in generate_chunks():
            audio_queue.put(audio_segment)
        audio_queue.put(None)  # Signal that production is done

    # Function to play audio from queue
    def consumer():
        combined_audio = AudioSegment.empty()
        while True:
            audio_segment = audio_queue.get()
            if audio_segment is None:
                break
            # Stop any existing playback
            audio_player.stop()
            # Play the audio segment
            play_audio_segment(audio_segment)
            # Append to combined audio
            combined_audio += audio_segment

        # Save the combined audio
        combined_audio.export(audio_file_path, format="wav")
        # Call the callback if provided
        if callback:
            callback()

    # Start producer and consumer threads
    Thread(target=producer).start()
    Thread(target=consumer).start()

def play_audio_segment(audio_segment):
    # Stop the waiting song before playing the audio segment
    stop_waiting_song()

    # Save segment to a temporary file
    temp_audio_path = "temp_segment.wav"
    audio_segment.export(temp_audio_path, format="wav")

    # Stop any existing playback
    audio_player.stop()
    # Play the audio using pygame
    audio_player.play(temp_audio_path)

    # Remove the temporary file after playback
    os.remove(temp_audio_path)

# AudioPlayer class using pygame
class AudioPlayer:
    def __init__(self):
        self.paused = False

    def play(self, file_path):
        if os.path.exists(file_path):
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                self.paused = False

                # Wait until playback is finished
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)  # Wait in small increments

            except Exception as e:
                print(f"Error playing audio file {file_path}: {e}")
        else:
            print(f"Audio file {file_path} does not exist.")

    def pause(self):
        if not self.paused:
            pygame.mixer.music.pause()
            self.paused = True

    def resume(self):
        if self.paused:
            pygame.mixer.music.unpause()
            self.paused = False

    def stop(self):
        pygame.mixer.music.stop()
        self.paused = False

# Instantiate audio player
audio_player = AudioPlayer()

# Function to play audio
def play_audio(file_path):
    # Stop any existing playback
    audio_player.stop()
    audio_player.play(file_path)

# Function to toggle play/pause
def toggle_play_pause():
    if audio_player.paused:
        audio_player.resume()
        play_pause_button.config(text="Pause")
    else:
        audio_player.pause()
        play_pause_button.config(text="Play")

# Extract text from scanned PDFs using OCR
def extract_text_from_scanned_pdf(file_path):
    try:
        pages = convert_from_path(file_path)
        text = ""
        for page_number, page_data in enumerate(pages):
            page_text = pytesseract.image_to_string(page_data)
            text += page_text + "\n"
        return text
    except Exception as e:
        speak_message(f"Failed to extract text from scanned PDF: {e}")
        return ""

# Extract text from PDFs with layout analysis
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
        speak_message(f"Failed to extract text from PDF: {e}")
        return ""

# Extract text from DOCX including headings
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
        speak_message(f"Failed to extract text from DOCX: {e}")
        return ""

# Extract text from a file
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
                speak_message("Performing OCR on scanned PDF.")
                text = extract_text_from_scanned_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            speak_message("Unsupported file format.")
            return ""

        # Preprocess and normalize the text
        text = preprocess_text(text)
        text = expand_abbreviations(text)
        text = normalize_numbers(text)
        return text

    except Exception as e:
        speak_message(f"Failed to extract text: {e}")
        return ""

# Function to generate a unique identifier for files
def get_file_id(file_path):
    # Use the absolute path to generate a unique hash
    abs_path = os.path.abspath(file_path)
    return hashlib.md5(abs_path.encode('utf-8')).hexdigest()

# Function to generate image description
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
        return ""

# Function to handle image upload and description
def browse_image():
    image_path = filedialog.askopenfilename(
        filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"))]
    )
    if image_path:
        # Update history
        file_name = os.path.basename(image_path)
        history.append(['Image', image_path, file_name])
        history_list.insert('', 'end', values=('Image', file_name, delete_icon))

        # Display image and description
        display_image(image_path)
        description = generate_image_description(image_path)
        display_text(description)

# Function to display image in the preview area
def display_image(image_path):
    image = Image.open(image_path)
    image = image.resize((400, 400), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    preview_label.config(image=photo, text='')
    preview_label.image = photo  # Keep a reference

# Function to display text in the preview area
def display_text(text):
    preview_label.config(text=text, image='')

# Function to read document from file path and text
def read_document_file(file_path, text, start_from_beginning=True):
    file_id = get_file_id(file_path)
    audio_file_path = os.path.join(audio_cache_dir, f"{file_id}.wav")

    if os.path.exists(audio_file_path):
        if start_from_beginning:
            speak_message("Starting from the beginning.")
            playback_positions[file_id] = 0
        else:
            speak_message("Resuming from where you left off.")
        # Play existing audio file
        play_audio_with_resume(audio_file_path, file_id)
    else:
        # Generate and play audio in real-time
        speak_message("Analyzing the document. Please wait.")
        # Start waiting song
        start_waiting_song()
        generate_and_play_audio(text, audio_file_path, callback=stop_waiting_song)

def play_audio_with_resume(file_path, file_id):
    # Stop any existing playback
    audio_player.stop()
    start_pos = playback_positions.get(file_id, 0)
    audio_segment = AudioSegment.from_file(file_path)
    if start_pos > 0:
        remaining_audio = audio_segment[start_pos:]
    else:
        remaining_audio = audio_segment
    temp_audio_path = "temp_resume_audio.wav"
    remaining_audio.export(temp_audio_path, format="wav")
    audio_player.play(temp_audio_path)

# UI functions
def browse_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Supported Documents", "*.txt *.pdf *.docx")]
    )
    if file_path:
        file_name = os.path.basename(file_path)
        # Update history with 'Document' type and delete icon
        history.append(['Document', file_path, file_name])
        history_list.insert('', 'end', values=('Document', file_name, delete_icon))

        # Display document text
        text = extract_text(file_path)
        if text:
            display_text(text[:1000] + '...')  # Display first 1000 characters
        else:
            display_text("Failed to extract text from the document.")

        # Start reading the document in a separate thread
        Thread(target=lambda: read_document_file(file_path, text, start_from_beginning=True)).start()

def read_from_beginning():
    selected_items = history_list.selection()
    if selected_items:
        selected_item = selected_items[0]
        item_index = history_list.index(selected_item)
        global current_history_index
        current_history_index = item_index
        item = history[current_history_index]
        item_type, data, name = item
        if item_type == 'Document':
            file_path = data  # Full path
            text = extract_text(file_path)
            if text:
                # Start reading the document from beginning
                Thread(target=lambda: read_document_file(file_path, text, start_from_beginning=True)).start()
            else:
                speak_message("Failed to extract text from the selected document.")
        else:
            speak_message("Please select a document from the history to read.")
    else:
        speak_message("No file selected. Please select a file to read.")

def resume_reading():
    selected_items = history_list.selection()
    if selected_items:
        selected_item = selected_items[0]
        item_index = history_list.index(selected_item)
        global current_history_index
        current_history_index = item_index
        item = history[current_history_index]
        item_type, data, name = item
        if item_type == 'Document':
            file_path = data  # Full path
            text = extract_text(file_path)
            if text:
                # Start reading the document from last position
                Thread(target=lambda: read_document_file(file_path, text, start_from_beginning=False)).start()
            else:
                speak_message("Failed to extract text from the selected document.")
        else:
            speak_message("Please select a document from the history to read.")
    else:
        speak_message("No file selected. Please select a file to read.")

# Function to handle selection from history
def on_history_select(event):
    selected_item = history_list.focus()
    if selected_item:
        item_values = history_list.item(selected_item, 'values')
        item_index = history_list.index(selected_item)
        global current_history_index
        current_history_index = item_index
        item_type, data, name = history[current_history_index]
        if item_type == 'Image':
            # Display image and description
            display_image(data)
            description = generate_image_description(data)
            display_text(description)
            # Optionally, read the image description aloud
            Thread(target=lambda: speak_message(description)).start()
        elif item_type == 'Document':
            # Display document text
            text = extract_text(data)
            if text:
                display_text(text[:1000] + '...')  # Display first 1000 characters
            else:
                display_text("Failed to extract text from the document.")
        elif item_type == 'Summary':
            # Display the summary text
            display_summary(data)
            # Optionally, read the summary aloud
            Thread(target=lambda: speak_large_text(data, language="en")).start()

def play_previous():
    global current_history_index
    if current_history_index > 0:
        current_history_index -= 1
        select_history_item(current_history_index)
    else:
        speak_message("No previous item in history.")

def play_next():
    global current_history_index
    if current_history_index < len(history) - 1:
        current_history_index += 1
        select_history_item(current_history_index)
    else:
        speak_message("No next item in history.")

def select_history_item(index):
    item = history[index]
    item_type, data, name = item
    # Update selection in the Treeview
    history_list.selection_set(history_list.get_children()[index])
    history_list.focus(history_list.get_children()[index])
    if item_type == 'Document':
        text = extract_text(data)
        if text:
            display_text(text[:1000] + '...')
            Thread(target=lambda: read_document_file(data, text)).start()
    elif item_type == 'Image':
        display_image(data)
        description = generate_image_description(data)
        display_text(description)
        # Optionally, read the image description aloud
        Thread(target=lambda: speak_message(description)).start()
    elif item_type == 'Summary':
        display_summary(data)
        # Optionally, read the summary aloud
        Thread(target=lambda: speak_large_text(data, language="en")).start()

def on_treeview_click(event):
    # Identify the row and column clicked
    region = history_list.identify("region", event.x, event.y)
    if region != "cell":
        return

    column = history_list.identify_column(event.x)
    row = history_list.identify_row(event.y)

    # Assuming 'Action' is the third column (#3)
    if column == '#3' and row:
        item = history_list.item(row)
        values = item['values']
        if len(values) >= 3:
            item_type, name, _ = values
        else:
            return  # Not enough values, possibly malformed entry

        # Find the data from the history
        data = None
        for hist_item in history:
            if hist_item[0] == item_type and hist_item[2] == name:
                data = hist_item[1]
                break

        if data is None:
            return

        # Confirm deletion
        confirm = messagebox.askyesno("Delete Confirmation", "Are you sure you want to delete this item?")
        if not confirm:
            return

        # Remove from Treeview
        history_list.delete(row)

        # Remove from history list
        for idx, hist_item in enumerate(history):
            if hist_item[0] == item_type and hist_item[1] == data:
                del history[idx]
                break

        # Delete associated files
        if item_type == 'Summary':
            # Find and delete the summary file
            summary_files = os.listdir(summary_cache_dir)
            for file_name in summary_files:
                file_path = os.path.join(summary_cache_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content == data:
                        os.remove(file_path)
                        break
        elif item_type == 'Document':
            # Delete cached audio file
            file_id = get_file_id(data)
            audio_file_path = os.path.join(audio_cache_dir, f"{file_id}.wav")
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
        elif item_type == 'Image':
            # Delete the image file
            if os.path.exists(data):
                os.remove(data)

        # Provide audio feedback
        speak_message("Item deleted from history.")

# Function to add focus highlight to widgets
def add_focus_highlight(widget):
    widget.bind('<FocusIn>', lambda e: widget.config(style='Focused.TButton'))
    widget.bind('<FocusOut>', lambda e: widget.config(style='TButton'))

# Setup keyboard shortcuts
def setup_keyboard_shortcuts():
    root.bind('<Alt-b>', lambda e: browse_file())
    root.bind('<Alt-i>', lambda e: browse_image())
    root.bind('<Alt-s>', lambda e: summarize_document())
    root.bind('<Alt-f>', lambda e: read_from_beginning())
    root.bind('<Alt-e>', lambda e: resume_reading())
    root.bind('<Alt-p>', lambda e: toggle_play_pause())
    root.bind('<Alt-Left>', lambda e: play_previous())
    root.bind('<Alt-Right>', lambda e: play_next())
    root.bind('<Escape>', lambda e: on_closing())
    root.bind('<Alt-h>', lambda e: show_help())
    root.bind('<Alt-r>', lambda e: repeat_last_message())
    root.bind('<Alt-a>', lambda e: show_about())  # New shortcut for About

def show_help():
    help_text = """
    **How to Use the Application:**

    - **Navigation**: Use the `Tab` key to navigate through the interactive widgets.
    - **Keyboard Shortcuts**:
      - `Alt + B`: Browse Document
      - `Alt + I`: Browse Image
      - `Alt + S`: Summarize Document
      - `Alt + F`: Read from Beginning
      - `Alt + E`: Resume Reading
      - `Alt + P`: Play or Pause
      - `Alt + Left Arrow`: Previous
      - `Alt + Right Arrow`: Next
      - `Alt + H`: Help
      - `Alt + R`: Repeat Last Message
      - `Alt + A`: About
      - `Escape`: Exit Application

    **Instructions:**

    - **Browse Document**: Open a dialog to select a document to read.
    - **Browse Image**: Open a dialog to select an image to describe.
    - **Summarize Document**: Summarize the selected document.
    - **Read from Beginning**: Start reading the selected document from the beginning.
    - **Resume Reading**: Resume reading the selected document from where you left off.
    - **Playback Controls**:
      - **Previous**: Go to the previous item in the history.
      - **Play/Pause**: Toggle playback of the current audio.
      - **Next**: Go to the next item in the history.
    """
    # Use a Toplevel window to display the help text
    help_window = tk.Toplevel(root)
    help_window.title("Help")
    help_window.geometry("600x400")

    # Make sure the window is accessible
    help_window.focus_set()

    # Create a Text widget to display the help text
    help_text_widget = tk.Text(help_window, wrap=tk.WORD)
    help_text_widget.insert(tk.END, help_text)
    help_text_widget.config(state=tk.DISABLED)
    help_text_widget.pack(fill=tk.BOTH, expand=True)

    # Add a scrollbar
    scrollbar = tk.Scrollbar(help_window, command=help_text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    help_text_widget.config(yscrollcommand=scrollbar.set)

def show_about():
    about_text = """
    Document Reader and Image Describer
    Version 1.0
    Developed by FallcorpIndustries

    FallcorpIndustries is dedicated to creating accessible software solutions for everyone.
    """
    messagebox.showinfo("About", about_text)

# Main application window
root = tk.Tk()
root.title("Document Reader and Image Describer for Blind Users")
root.geometry("800x600")

# Setup keyboard shortcuts
setup_keyboard_shortcuts()

# Global variables
current_history_index = -1  # To keep track of the current item in history
playback_positions = {}  # To keep track of playback positions

# Load history from file if it exists
history = []
history_file_path = os.path.join(base_path, 'history.json')
if os.path.exists(history_file_path):
    with open(history_file_path, 'r') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []

# Create the main frames
toolbar = tk.Frame(root)
toolbar.pack(side=tk.TOP, fill=tk.X)

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(main_frame, width=200)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Style for focus highlight
style = ttk.Style()
style.theme_use('clam')
style.configure('Focused.TButton', foreground='blue')
style.configure('TButton', foreground='black')

# Toolbar buttons
browse_file_button = ttk.Button(toolbar, text="Browse Document", command=browse_file, takefocus=True)
browse_file_button.pack(side=tk.LEFT, padx=2, pady=2)

browse_image_button = ttk.Button(toolbar, text="Browse Image", command=browse_image, takefocus=True)
browse_image_button.pack(side=tk.LEFT, padx=2, pady=2)

# Summarize Document Button
summarize_button = ttk.Button(toolbar, text="Summarize Document", command=summarize_document, takefocus=True)
summarize_button.pack(side=tk.LEFT, padx=2, pady=2)

# Create a frame in right_frame to hold the buttons at the top right
top_right_controls = tk.Frame(right_frame)
top_right_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# Create the Read from Beginning and Resume Reading buttons in top_right_controls
resume_button = ttk.Button(top_right_controls, text="Resume Reading", command=resume_reading, takefocus=True)
resume_button.pack(side=tk.RIGHT, padx=2)

read_beginning_button = ttk.Button(top_right_controls, text="Read from Beginning", command=read_from_beginning, takefocus=True)
read_beginning_button.pack(side=tk.RIGHT, padx=2)

# Preview area in right frame
preview_frame = tk.Frame(right_frame, bd=2, relief=tk.SUNKEN)
preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

preview_label = ttk.Label(preview_frame, text="Preview Area", wraplength=400)
preview_label.pack(pady=10)

# Playback controls
controls_frame = tk.Frame(preview_frame)
controls_frame.pack(side=tk.BOTTOM, pady=10)

prev_button = ttk.Button(controls_frame, text="Previous", command=play_previous, takefocus=True)
prev_button.pack(side=tk.LEFT, padx=5)

play_pause_button = ttk.Button(controls_frame, text="Pause", command=toggle_play_pause, takefocus=True)
play_pause_button.pack(side=tk.LEFT, padx=5)

next_button = ttk.Button(controls_frame, text="Next", command=play_next, takefocus=True)
next_button.pack(side=tk.LEFT, padx=5)

# History list in left frame with 'Action' column
history_label = ttk.Label(left_frame, text="History")
history_label.pack(pady=5)

history_list = ttk.Treeview(left_frame, columns=('Type', 'Name', 'Action'), show='headings')
history_list.heading('Type', text='Type')
history_list.heading('Name', text='Name')
history_list.heading('Action', text='Action')
history_list.column('Type', width=80, anchor='center')
history_list.column('Name', width=140)
history_list.column('Action', width=80, anchor='center')  # Adjust width as needed
history_list.pack(fill=tk.BOTH, expand=True)

# Bind the TreeviewSelect event for displaying selected item
history_list.bind('<<TreeviewSelect>>', on_history_select)

# Bind left mouse button click to the on_treeview_click function
history_list.bind("<Button-1>", on_treeview_click)

# Load history into the Treeview with delete icons
for item in history:
    item_type, data, name = item
    history_list.insert('', 'end', values=(item_type, name, delete_icon))

# Function to handle application exit
def on_closing():
    # Save history to a file
    with open(history_file_path, 'w') as f:
        json.dump(history, f)
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Set widget names for screen reader compatibility
browse_file_button._name = "Browse Document Button"
browse_image_button._name = "Browse Image Button"
summarize_button._name = "Summarize Document Button"
read_beginning_button._name = "Read from Beginning Button"
resume_button._name = "Resume Reading Button"
prev_button._name = "Previous Button"
play_pause_button._name = "Play or Pause Button"
next_button._name = "Next Button"
history_label._name = "History Label"
preview_label._name = "Preview Area Label"

# Add focus highlight to widgets
interactive_widgets = [
    browse_file_button,
    browse_image_button,
    summarize_button,
    read_beginning_button,
    resume_button,
    prev_button,
    play_pause_button,
    next_button
]

for widget in interactive_widgets:
    add_focus_highlight(widget)
    # Audio feedback on focus is delegated to the screen reader

# Create the menu bar
menu_bar = tk.Menu(root)

# Add "Help" menu
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="Help", command=show_help)
menu_bar.add_cascade(label="Help", menu=help_menu)

# Add "About" menu
about_menu = tk.Menu(menu_bar, tearoff=0)
about_menu.add_command(label="About", command=show_about)
menu_bar.add_cascade(label="About", menu=about_menu)

# Configure the menu bar
root.config(menu=menu_bar)

# Provide an initial audio message
Thread(target=lambda: speak_message(
    "Welcome to the Document Reader and Image Describer. Please select a file or image to proceed."
)).start()

# Start the main loop
root.mainloop()
