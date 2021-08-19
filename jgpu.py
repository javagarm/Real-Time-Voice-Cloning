from scipy.ndimage.measurements import label
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
import os
import sounddevice as sd
from audioread.exceptions import NoBackendError
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
app = FastAPI()
    # Get the reference audio filepath
    #message = "Reference voice"
@app.post("/train")
async def train(audio: UploadFile = File(...),label: str = Form(...)):
    print(audio.filename)
    try:
        os.mkdir("audios")
        print(os.getcwd())
    except Exception as e:
        print(e) 
    file_name = os.getcwd()+"/audios/"+audio.filename.replace(" ", "-").replace("\"", "").replace("\'", "")
    with open(file_name,'wb+') as f:
        f.write(audio.file.read())
        f.close()
    #in_fpath = Path("Trumpp.mp3".replace("\"", "").replace("\'", ""))
    print(audio.filename)
    in_fpath = "audios/"+audio.filename
    print(in_fpath)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--enc_model_fpath", type=Path,default="encoder/saved_models/pretrained.pt")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,default="synthesizer/saved_models/pretrained/pretrained.pt")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,default="vocoder/saved_models/pretrained/pretrained.pt")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--no_mp3_support", action="store_true", help=\
        "If True, disallows loading mp3 files to prevent audioread errors when ffmpeg is not installed.")
    args = parser.parse_args()
    #if in_fpath.suffix.lower() == ".mp3" and args.no_mp3_support:
     #   print("Can't Use mp3 files please try again:")
    check_model_paths(encoder_path=args.enc_model_fpath,
                      synthesizer_path=args.syn_model_fpath,
                      vocoder_path=args.voc_model_fpath)
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    text = label #input("Write a sentence (+-20 words) to be synthesized:\n")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    if args.seed is not None:
        torch.manual_seed(args.seed)
        vocoder.load_model(args.voc_model_fpath)
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)
    if not args.no_sound:
        try:
            sd.stop()
            sd.play(generated_wav, synthesizer.sample_rate)
        except sd.PortAudioError as e:
            print("\nCaught exception: %s" % repr(e))
        except:
            raise
    filename = "demo_output.wav"
    print(generated_wav.dtype)
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print("--------Wav file saved in same folder as demo_output.wav----")
    return(filename)
if __name__ == "__main__":
    uvicorn.run("jcpuapi:app", host="127.0.0.1", port=8000,reload=True)