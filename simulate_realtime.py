"""
Simulated real-time inference from audio file
This script demonstrates real-time analysis by streaming an audio file
"""

import os
import sys
import time
import numpy as np
import torch
import librosa
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SONGFORMER_DIR = os.path.join(SCRIPT_DIR, "src", "SongFormer")

# Change to SongFormer directory for model loading
os.chdir(SONGFORMER_DIR)
sys.path.append(os.path.join(SONGFORMER_DIR, "..", "third_party"))
sys.path.append(SONGFORMER_DIR)

# Monkey patch for msaf
import scipy

scipy.inf = np.inf

from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz
from omegaconf import OmegaConf
from ema_pytorch import EMA
import importlib

from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
from postprocessing.functional import postprocess_functional_structure

# Constants
MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")
INPUT_SAMPLING_RATE = 24000
DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]
CHUNK_DURATION = 30  # Analyze in 30-second chunks


def load_models(device, model_name="SongFormer", config_path="SongFormer.yaml", checkpoint="SongFormer.safetensors"):
    """Load all required models"""
    print("Loading models...")

    # Load MuQ
    print("  Loading MuQ...")
    muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq_model = muq_model.to(device).eval()

    # Load MusicFM
    print("  Loading MusicFM...")
    musicfm_model = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
        model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
    )
    musicfm_model = musicfm_model.to(device).eval()

    # Load SongFormer
    print("  Loading SongFormer...")
    module = importlib.import_module(f"models.{model_name}")
    Model = getattr(module, "Model")
    hp = OmegaConf.load(os.path.join("configs", config_path))
    msa_model = Model(hp)

    # Load checkpoint
    checkpoint_path = os.path.join("ckpts", checkpoint)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        ckpt = {"model_ema": load_file(checkpoint_path, device=device)}
    else:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if ckpt.get("model_ema", None) is not None:
        model_ema = EMA(msa_model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        msa_model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        msa_model.load_state_dict(ckpt["model"])

    msa_model.to(device).eval()

    # Prepare label mask
    dataset_id2label_mask = {}
    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(128, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    print("Models loaded successfully!\n")

    return muq_model, musicfm_model, msa_model, hp, dataset_id2label_mask


def analyze_chunk(
    audio_chunk, muq_model, musicfm_model, msa_model, hp, label_mask, device
):
    """Analyze a chunk of audio"""
    with torch.no_grad():
        # Get embeddings
        muq_output = muq_model(audio_chunk.unsqueeze(0), output_hidden_states=True)
        muq_embd = muq_output["hidden_states"][10]

        _, musicfm_hidden_states = musicfm_model.get_predictions(
            audio_chunk.unsqueeze(0)
        )
        musicfm_embd = musicfm_hidden_states[10]

        # Align and combine embeddings
        min_len = min(muq_embd.shape[1], musicfm_embd.shape[1])
        embd = torch.cat(
            [musicfm_embd[:, :min_len, :], muq_embd[:, :min_len, :]], dim=-1
        )

        # Inference
        dataset_ids = torch.Tensor(DATASET_IDS).to(device, dtype=torch.long)
        msa_info, chunk_logits = msa_model.infer(
            input_embeddings=embd,
            dataset_ids=dataset_ids,
            label_id_masks=torch.Tensor(label_mask)
            .to(device, dtype=bool)
            .unsqueeze(0)
            .unsqueeze(0),
            with_logits=True,
        )

        # Post-process
        logits = {
            "function_logits": chunk_logits["function_logits"][0].unsqueeze(0),
            "boundary_logits": chunk_logits["boundary_logits"][0].unsqueeze(0),
        }

        msa_output = postprocess_functional_structure(logits, hp)

        return msa_output


def stream_analyze_file(
    audio_path,
    muq_model,
    musicfm_model,
    msa_model,
    hp,
    label_mask,
    device,
    simulate_realtime=True,
):
    """Analyze audio file in streaming fashion"""
    print(f"Loading audio file: {audio_path}")

    # Load full audio
    wav, sr = librosa.load(audio_path, sr=INPUT_SAMPLING_RATE)
    audio = torch.tensor(wav, dtype=torch.float32).to(device)

    total_duration = len(wav) / INPUT_SAMPLING_RATE
    print(f"Audio duration: {total_duration:.1f} seconds")
    print(f"Analyzing in {CHUNK_DURATION}-second chunks...\n")

    all_results = []
    chunk_size = CHUNK_DURATION * INPUT_SAMPLING_RATE
    overlap = 15 * INPUT_SAMPLING_RATE  # 15 second overlap

    position = 0
    chunk_num = 0

    while position < len(audio):
        end_pos = min(position + chunk_size, len(audio))

        if end_pos - position < INPUT_SAMPLING_RATE * 5:  # Skip very short chunks
            break

        chunk_num += 1
        current_time = position / INPUT_SAMPLING_RATE

        print(
            f"[{current_time:.1f}s - {end_pos/INPUT_SAMPLING_RATE:.1f}s] Analyzing chunk {chunk_num}..."
        )

        if simulate_realtime:
            time.sleep(2)  # Simulate processing time

        # Analyze chunk
        audio_chunk = audio[position:end_pos]

        try:
            result = analyze_chunk(
                audio_chunk, muq_model, musicfm_model, msa_model, hp, label_mask, device
            )

            if result:
                # Adjust timestamps relative to full audio
                adjusted_result = []
                for timestamp, label in result:
                    adjusted_timestamp = timestamp + current_time
                    adjusted_result.append((adjusted_timestamp, label))

                # Display results for this chunk
                print(f"  Detected segments in this chunk:")
                for i in range(len(adjusted_result) - 1):
                    start = adjusted_result[i][0]
                    end = adjusted_result[i + 1][0]
                    label = adjusted_result[i][1]
                    print(f"    {start:6.1f}s - {end:6.1f}s : {label}")

                all_results.extend(adjusted_result)

            print()

        except Exception as e:
            print(f"  Error analyzing chunk: {e}\n")

        # Move to next chunk with overlap
        position += chunk_size - overlap

    return all_results


def save_results(results, output_path):
    """Save results to file"""
    with open(output_path, "w") as f:
        f.write("Music Structure Analysis Results (Simulated Real-time)\n")
        f.write("=" * 60 + "\n\n")

        if not results:
            f.write("No segments detected.\n")
            return

        # Remove duplicates and sort
        unique_results = sorted(set(results), key=lambda x: x[0])

        for i in range(len(unique_results) - 1):
            start = unique_results[i][0]

            label = unique_results[i][1]
            f.write(f"{start:6.1f} {label}\n")

        # Add end marker
        if unique_results and unique_results[-1][1] == "end":
            f.write(f"{unique_results[-1][0]:6.1f} end\n")

    print(f"Results saved to: {output_path}")


def main():
    """Main entry point"""
    import argparse

    try:
        parser = argparse.ArgumentParser(
            description="Simulated real-time music structure analysis from audio file"
        )
        parser.add_argument("audio_file", type=str, help="Path to audio file")
        parser.add_argument("--output", type=str, default=None, help="Output file path")
        parser.add_argument(
            "--no-simulate",
            action="store_true",
            help="Don't simulate real-time delays (process as fast as possible)",
        )
        parser.add_argument("--model", type=str, default="SongFormer", help="Model name")
        parser.add_argument(
            "--config", type=str, default="SongFormer.yaml", help="Config file"
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="SongFormer.safetensors",
            help="Checkpoint file",
        )

        args = parser.parse_args()

        # Convert audio file path to absolute path before changing directory
        audio_file = os.path.abspath(args.audio_file)
        
        # Convert output path to absolute path if provided
        if args.output:
            output_path = os.path.abspath(args.output)
        else:
            output_path = None

        # Check if audio file exists
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            return 1

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}\n")

        # Load models
        muq_model, musicfm_model, msa_model, hp, label_mask = load_models(
            device, args.model, args.config, args.checkpoint
        )

        # Get label mask for dataset
        dataset_id = DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]
        mask = label_mask[dataset_id]

        # Analyze file
        results = stream_analyze_file(
            audio_file,
            muq_model,
            musicfm_model,
            msa_model,
            hp,
            mask,
            device,
            simulate_realtime=not args.no_simulate,
        )

        # Save results
        if output_path:
            final_output_path = output_path
        else:
            base_name = Path(audio_file).stem
            # Save to script directory
            final_output_path = os.path.join(SCRIPT_DIR, f"{base_name}_realtime_analysis.txt")

        save_results(results, final_output_path)

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print("=" * 60)
        return 0
    except Exception as exc:
        print(f"Error during analysis: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
