from fastapi import FastAPI, Form
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
from audioread.exceptions import NoBackendError
import csv
import json
import os
import uvicorn

import numpy as np

from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

    if __name__ == '__main__':
    ## Info & args
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                            default="encoder/saved_models/pretrained.pt",
                            help="Path to a saved encoder")
        parser.add_argument("-s", "--syn_model_fpath", type=Path, 
                            default="synthesizer/saved_models/pretrained/pretrained.pt",
                            help="Path to a saved synthesizer")
        parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                            default="vocoder/saved_models/pretrained/pretrained.pt",
                            help="Path to a saved vocoder")
        parser.add_argument("--cpu", action="store_true", help=\
            "If True, processing is done on CPU, even when a GPU is available.")
        parser.add_argument("--no_sound", action="store_true", help=\
            "If True, audio won't be played.")
        parser.add_argument("--seed", type=int, default=None, help=\
            "Optional random number seed value to make toolbox deterministic.")
        parser.add_argument("--no_mp3_support", action="store_true", help=\
            "If True, disallows loading mp3 files to prevent audioread errors when ffmpeg is not installed.")
        args = parser.parse_args()
        print_args(args, parser)
        if not args.no_sound:
            import sounddevice as sd
        
        if args.cpu:
            # Hide GPUs from Pytorch to force CPU processing
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if not args.no_mp3_support:
            try:
                librosa.load("samples/1320_00000.mp3")
            except NoBackendError:
                print("Librosa will be unable to open mp3 files if additional software is not installed.\n"
                    "Please install ffmpeg or add the '--no_mp3_support' option to proceed without support for mp3 files.")
                exit(-1)
        print("Running a test of your configuration...\n")
        
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device_id)
            ## Print some environment information (for debugging purposes)
            print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
                "%.1fGb total memory.\n" % 
                (torch.cuda.device_count(),
                device_id,
                gpu_properties.name,
                gpu_properties.major,
                gpu_properties.minor,
                gpu_properties.total_memory / 1e9))
        else:
            print("Using CPU for inference.\n")
        ## Remind the user to download pretrained models if needed
        check_model_paths(encoder_path=args.enc_model_fpath,
                        synthesizer_path=args.syn_model_fpath,
                        vocoder_path=args.voc_model_fpath)
        
        ## Load the models one by one.
        print("Preparing the encoder, the synthesizer and the vocoder...")
        encoder.load_model(args.enc_model_fpath)
        synthesizer = Synthesizer(args.syn_model_fpath)
        vocoder.load_model(args.voc_model_fpath)

        encoder.embed_utterance(np.zeros(encoder.sampling_rate))

        embed = np.random.rand(speaker_embedding_size)

        embed /= np.linalg.norm(embed)

        embeds = [embed, np.zeros(speaker_embedding_size)]
        texts = ["test 1", "test 2"]
        print("\tTesting the synthesizer... (loading the model will output a lot of text)")
        mels = synthesizer.synthesize_spectrograms(texts, embeds)
        mel = np.concatenate(mels, axis=1)
        no_action = lambda *args: None
        vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

        num_generated = 0
        if text == str:
            try:
                # Get the reference audio filepath
                message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                        "wav, m4a, flac, ...):\n"
                in_fpath = Path("Trumpp.mp3".replace("\"", "").replace("\'", ""))

                if in_fpath.suffix.lower() == ".mp3" and args.no_mp3_support:
                    print("Can't Use mp3 files please try again:")
                
                preprocessed_wav = encoder.preprocess_wav(in_fpath)
                original_wav, sampling_rate = librosa.load(str(in_fpath))
                preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
                print("Loaded file succesfully")
                embed = encoder.embed_utterance(preprocessed_wav)
                #text= inputtoread
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
                        print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
                    except:
                        raise
                    
                # Save it on the disk
                filename = "demo_output_%02d.wav" % num_generated
                print(generated_wav.dtype)
                sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
                num_generated += 1
                print("\nSaved output as %s\n\n" % filename)
            except Exception as e:
                print("Caught exception: %s" % repr(e))
                print("Restarting\n")
        else:
            print("d")
    return {}
