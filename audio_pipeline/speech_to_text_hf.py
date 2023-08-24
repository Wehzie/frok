import torch
import sounddevice as sd
from transformers import pipeline
from scipy.io.wavfile import write, read
from transformers import WhisperProcessor, WhisperForConditionalGeneration

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

def run_pipeline(filename:str="", s:int=30) -> str:
    """ Transcribes an audio file using whisper-large-v2 through hugging faces pipelines.

    Args:
        file (str, optional): filename. Defaults to "".
        s (int): 
    """

    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    chunk_length_s=s,
    device=device,
    )

    if filename == "":
        filename = "./audio_files/audio_file.wav"
        sample = record_audio(s, filename)
    else:
        sample = read(filename)
    
    prediction = pipe(sample[1])["text"]

    with open("transcription_files/pipeline_transcription_" + filename.split(".")[0] + ".txt", "w") as f:
        f.write(prediction)

    print(prediction)
    return prediction

def run_model_instance(filename:str = "", s:int = 30) -> str:
    """ Transcribes an audio file using the instanciated model shisper-large-v2.

    Args:
        file (str, optional): filename. Defaults to "".
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    model.config.forced_decoder_ids = None
    
    if filename == "":
        filename = "./audio_files/audio_file.wav"
        sample = record_audio(s, filename)
    else:
        sample = read(filename)

    # load dummy dataset and read audio files
    input_features = processor(sample[1], sampling_rate = sample[0], return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    with open("transcription_files/model_transcription_" + filename.split(".")[0] + ".txt", "w") as f:
        f.write(transcription[0])

    print(transcription)
    return transcription

def main():
    """ Transcription tool using whisper-large-v2
    """
    _ = run_pipeline("./audio_files/audio_file.wav")
    _ = run_model_instance("./audio_files/audio_file.wav")

if __name__ == "__main__":
    main()