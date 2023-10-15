from flask import Flask, request, jsonify
import torch
import sounddevice as sd
import whisper # pip install -U openai-whisper
from scipy.io.wavfile import write, read

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def record_audio(duration:int=60, filename:str="new_audio.wav") -> list:
    """ Records an arbitrary length audio message and saves it at 16000.

    Args:
        duration (int, optional): Duration of recording. Defaults to 60.
        filename (str, optional): Filename. Defaults to "new_audio.wav".
        fs (int, optional): Sampling frequency. Defaults to 16000.

    Returns:
        _type_: wav
    """
    FS = 16000
    # Start recording for 'duration' seconds
    print("Recording.")
    recording = sd.rec(int(duration * FS), samplerate=FS, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, FS, recording)  # Save as WAV file
    print("Done recording.")
    return recording

def transcribe(filename:str="", s:int=30) -> str:
    """ Transcribes an audio file using whisper.

    Args:
        file (str, optional): filename. Defaults to "".
        s (int): sample recording length.
    """

    if filename == "":
        filename = "./audio_files/audio_file.wav"
        record_audio(s, filename)
    
    model = whisper.load_model("base.en") # base 74 M, small 244 M, medium 769 M parameters use .en for english only
    result = model.transcribe(filename)
    print(result["text"])

    with open("transcription_files/pipeline_transcription_" + filename.split(".")[0] + ".txt", "w") as f:
        f.write(result["text"])

    return result

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    audio_file = request.files['file']
    audio_file_path = "uploaded_audio.wav"
    audio_file.save(audio_file_path)

    transcription_result = transcribe(audio_file_path)
    return jsonify(transcription_result)

if __name__ == '__main__':
    app.run(debug=True)