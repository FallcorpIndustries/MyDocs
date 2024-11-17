from TTS.api import TTS

# Initialize the TTS model
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
# Define the input text and reference speaker audio
text_fr = "Bonjour, je teste le clonage de voix avec ce modèle."  # French text
text_en = "Hello, I am testing voice cloning with this model."    # English text
reference_audio = "cleng.wav"  # Replace with the path to your speaker's reference audio

# Generate speech in French with voice cloning
tts.tts_with_vc_to_file(
    text=text_en,
    speaker_wav=reference_audio,
    file_path="output_english.wav"
)


print("Voice cloning outputs have been generated.")
