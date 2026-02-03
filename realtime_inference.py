"""
Real-time Music Structure Analysis using SongFormer
Captures audio from microphone and performs live inference
"""

import os
import sys
import queue
import threading
import numpy as np
import torch
import librosa
import sounddevice as sd
from collections import deque
from datetime import datetime

os.chdir(os.path.join("src", "SongFormer"))
sys.path.append(os.path.join("..", "third_party"))
sys.path.append(".")

# Monkey patch for msaf
import scipy

scipy.inf = np.inf

from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz
from omegaconf import OmegaConf
from ema_pytorch import EMA
import importlib
import math
from dataset.label2id import (
    DATASET_ID_ALLOWED_LABEL_IDS,
    DATASET_LABEL_TO_DATASET_ID,
    ID_TO_LABEL,
)
from postprocessing.functional import postprocess_functional_structure

# Constants
MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")
INPUT_SAMPLING_RATE = 24000
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333
DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]
TIME_DUR = 420
ANALYSIS_WINDOW = 30  # Analyze every 30 seconds of audio
OVERLAP = 15  # 15 second overlap for smooth analysis


class RealtimeAnalyzer:
    """Real-time music structure analyzer"""

    def __init__(
        self,
        model_name="SongFormer",
        checkpoint="SongFormer.safetensors",
        config_path="SongFormer.yaml",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Audio buffer
        self.audio_buffer = deque(maxlen=INPUT_SAMPLING_RATE * TIME_DUR)
        self.audio_queue = queue.Queue()
        self.running = False

        # Analysis state
        self.total_audio_received = 0
        self.last_analysis_time = 0
        self.analysis_results = []

        # Load models
        print("Loading models...")
        self.load_models(model_name, checkpoint, config_path)
        print("Models loaded successfully!")

    def load_models(self, model_name, checkpoint, config_path):
        """Load MuQ, MusicFM and MSA models"""
        # Load MuQ
        print("Loading MuQ...")
        self.muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq_model = self.muq_model.to(self.device).eval()

        # Load MusicFM
        print("Loading MusicFM...")
        self.musicfm_model = MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
            model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
        )
        self.musicfm_model = self.musicfm_model.to(self.device).eval()

        # Load MSA model
        print("Loading SongFormer...")
        module = importlib.import_module("models." + str(model_name))
        Model = getattr(module, "Model")
        self.hp = OmegaConf.load(os.path.join("configs", config_path))
        self.msa_model = Model(self.hp)

        ckpt = self.load_checkpoint(os.path.join("ckpts", checkpoint))
        if ckpt.get("model_ema", None) is not None:
            model_ema = EMA(self.msa_model, include_online_model=False)
            model_ema.load_state_dict(ckpt["model_ema"])
            self.msa_model.load_state_dict(model_ema.ema_model.state_dict())
        else:
            self.msa_model.load_state_dict(ckpt["model"])

        self.msa_model.to(self.device).eval()

        # Prepare label masks
        self.dataset_id2label_mask = {}
        for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
            self.dataset_id2label_mask[key] = np.ones(128, dtype=bool)
            self.dataset_id2label_mask[key][allowed_ids] = False

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint from path"""
        if checkpoint_path.endswith(".pt"):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        elif checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint = {"model_ema": load_file(checkpoint_path, device=self.device)}
        else:
            raise ValueError("Unsupported checkpoint format")
        return checkpoint

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")

        # Convert to mono if stereo
        if len(indata.shape) > 1:
            audio_data = indata[:, 0]
        else:
            audio_data = indata

        self.audio_queue.put(audio_data.copy())

    def process_audio_buffer(self):
        """Process accumulated audio buffer"""
        if len(self.audio_buffer) < INPUT_SAMPLING_RATE * ANALYSIS_WINDOW:
            return None

        # Convert buffer to tensor
        audio_array = np.array(self.audio_buffer)
        audio = torch.tensor(audio_array, dtype=torch.float32).to(self.device)

        # Analyze the audio segment
        result = self.analyze_segment(audio)
        return result

    def analyze_segment(self, audio):
        """Analyze a segment of audio"""
        with torch.no_grad():
            try:
                # Get embeddings from both models
                muq_output = self.muq_model(
                    audio.unsqueeze(0), output_hidden_states=True
                )
                muq_embd = muq_output["hidden_states"][10]

                _, musicfm_hidden_states = self.musicfm_model.get_predictions(
                    audio.unsqueeze(0)
                )
                musicfm_embd = musicfm_hidden_states[10]

                # Combine embeddings
                embd_lens = [muq_embd.shape[1], musicfm_embd.shape[1]]
                min_embd_len = min(embd_lens)
                muq_embd = muq_embd[:, :min_embd_len, :]
                musicfm_embd = musicfm_embd[:, :min_embd_len, :]

                embd = torch.cat([musicfm_embd, muq_embd], dim=-1)

                # Inference
                dataset_ids = torch.Tensor(DATASET_IDS).to(
                    self.device, dtype=torch.long
                )
                msa_info, chunk_logits = self.msa_model.infer(
                    input_embeddings=embd,
                    dataset_ids=dataset_ids,
                    label_id_masks=torch.Tensor(
                        self.dataset_id2label_mask[
                            DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]
                        ]
                    )
                    .to(self.device, dtype=bool)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    with_logits=True,
                )

                # Post-process
                logits = {
                    "function_logits": chunk_logits["function_logits"][0].unsqueeze(0),
                    "boundary_logits": chunk_logits["boundary_logits"][0].unsqueeze(0),
                }

                msa_output = postprocess_functional_structure(logits, self.hp)

                return msa_output

            except Exception as e:
                print(f"Error during analysis: {e}")
                import traceback

                traceback.print_exc()
                return None

    def format_result(self, msa_output, time_offset):
        """Format analysis result with time offset"""
        if not msa_output:
            return []

        segments = []
        for i in range(len(msa_output) - 1):
            start_time = msa_output[i][0] + time_offset
            end_time = msa_output[i + 1][0] + time_offset
            label = msa_output[i][1]
            segments.append({"start": start_time, "end": end_time, "label": label})

        return segments

    def print_results(self, segments):
        """Print analysis results"""
        if not segments:
            return

        print("\n" + "=" * 60)
        print(f"Analysis Update - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        for seg in segments[-5:]:  # Show last 5 segments
            print(f"{seg['start']:6.1f}s - {seg['end']:6.1f}s : {seg['label']}")
        print("=" * 60 + "\n")

    def audio_processing_thread(self):
        """Thread to process incoming audio"""
        while self.running:
            try:
                # Get audio from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Add to buffer
                self.audio_buffer.extend(audio_chunk)
                self.total_audio_received += len(audio_chunk)

                # Check if it's time to analyze
                current_time = self.total_audio_received / INPUT_SAMPLING_RATE
                if current_time - self.last_analysis_time >= ANALYSIS_WINDOW - OVERLAP:
                    print(f"\n[{current_time:.1f}s] Analyzing audio segment...")

                    result = self.process_audio_buffer()
                    if result:
                        time_offset = max(0, self.last_analysis_time)
                        segments = self.format_result(result, time_offset)

                        # Update results
                        self.analysis_results.extend(segments)
                        self.print_results(segments)

                    self.last_analysis_time = current_time

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback

                traceback.print_exc()

    def start(self, device_id=None):
        """Start real-time analysis"""
        self.running = True

        # Start processing thread
        process_thread = threading.Thread(
            target=self.audio_processing_thread, daemon=True
        )
        process_thread.start()

        # Start audio stream
        print(f"\nStarting audio capture from microphone...")
        print(f"Sampling rate: {INPUT_SAMPLING_RATE} Hz")
        print(f"Analysis window: {ANALYSIS_WINDOW}s with {OVERLAP}s overlap")
        print("Press Ctrl+C to stop\n")

        try:
            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=INPUT_SAMPLING_RATE,
                callback=self.audio_callback,
                blocksize=INPUT_SAMPLING_RATE // 4,  # 0.25 second blocks
            ):
                while self.running:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.running = False
            process_thread.join(timeout=2)

            # Save results
            if self.analysis_results:
                self.save_results()

    def save_results(self):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realtime_analysis_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write("Real-time Music Structure Analysis Results\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Total duration: {self.total_audio_received / INPUT_SAMPLING_RATE:.1f}s\n"
            )
            f.write("\nSegments:\n")
            f.write("=" * 60 + "\n")

            for seg in self.analysis_results:
                f.write(f"{seg['start']:6.1f}s - {seg['end']:6.1f}s : {seg['label']}\n")

        print(f"\nResults saved to: {filename}")


def list_audio_devices():
    """List available audio input devices"""
    print("\nAvailable audio input devices:")
    print("=" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            print(f"[{i}] {device['name']}")
            print(
                f"    Channels: {device['max_input_channels']}, "
                f"Sample rate: {device['default_samplerate']} Hz"
            )
    print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Music Structure Analysis")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID (use --list-devices to see options)",
    )
    parser.add_argument("--model", type=str, default="SongFormer", help="Model name")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="SongFormer.safetensors",
        help="Checkpoint file",
    )
    parser.add_argument(
        "--config", type=str, default="SongFormer.yaml", help="Config file"
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Initialize analyzer
    analyzer = RealtimeAnalyzer(
        model_name=args.model, checkpoint=args.checkpoint, config_path=args.config
    )

    # Start analysis
    analyzer.start(device_id=args.device)


if __name__ == "__main__":
    main()
