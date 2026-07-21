"""
Command Line Interface (CLI) for audio-pipeline.
"""

import os
import sys

import click

from src.audio_pipeline.offline.wav_to_parquet import WavToParquetConverter


@click.group()
def main():
    """Audio Pipeline Command Line Tool."""
    pass


@main.command()
@click.option(
    "--input",
    "-i",
    "input_wav",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to input WAV audio file.",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(writable=True, file_okay=False, dir_okay=True),
    help="Directory to write output artefacts (acoustic_features.parquet, "
    "speech_segments.parquet, speaker_turns.parquet, extraction_metadata.json).",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to configuration YAML file.",
)
@click.option(
    "--enable-vad",
    is_flag=True,
    default=False,
    help="Enable Silero VAD (requires torch). Populates vad_probability_mean and voiced_ratio.",
)
@click.option(
    "--enable-yamnet",
    is_flag=True,
    default=False,
    help="Enable YAMNet ONNX distress-event detection. Downloads ~100 MB model on first run.",
)
@click.option(
    "--enable-diarizer",
    is_flag=True,
    default=False,
    help="Enable Pyannote speaker diarization. Requires HF_TOKEN environment variable.",
)
@click.option(
    "--hf-token",
    "hf_token",
    required=False,
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace access token for Pyannote (also read from HF_TOKEN env var).",
)
def wav_to_parquet(
    input_wav: str,
    output_dir: str,
    config_path: str,
    enable_vad: bool,
    enable_yamnet: bool,
    enable_diarizer: bool,
    hf_token: str,
):
    """
    Convert a WAV file to a Parquet file with extracted acoustic features and quality metrics.

    Writes four artefacts inside OUTPUT_DIR:
      acoustic_features.parquet  — one row per fixed feature window
      speech_segments.parquet    — VAD speech boundaries
      speaker_turns.parquet      — Pyannote speaker turn boundaries
      extraction_metadata.json   — model versions and provenance
    """
    click.echo(f"Processing audio file: {input_wav}...")

    try:
        # Load config
        converter_base = WavToParquetConverter.from_config(config_path)

        # Instantiate requested models
        vad = None
        yamnet = None
        diarizer = None

        if enable_vad:
            click.echo("Loading Silero VAD...")
            from src.audio_pipeline.segmentation.silero_vad import SileroVAD
            vad = SileroVAD()
            click.echo("  ✓ Silero VAD loaded.")

        if enable_yamnet:
            click.echo("Loading YAMNet ONNX detector (downloads on first run)...")
            from src.audio_pipeline.features.yamnet_detector import YAMNetDetector
            yamnet = YAMNetDetector()
            click.echo("  ✓ YAMNet loaded.")

        if enable_diarizer:
            token = hf_token or os.environ.get("HF_TOKEN")
            if not token:
                click.echo(
                    "Error: --enable-diarizer requires HF_TOKEN. "
                    "Set the HF_TOKEN environment variable or pass --hf-token.",
                    err=True,
                )
                sys.exit(1)
            click.echo("Loading Pyannote diarizer...")
            from src.audio_pipeline.speakers.pyannote_diarizer import PyannoteDiarizer
            diarizer = PyannoteDiarizer(hf_token=token)
            click.echo("  ✓ Pyannote diarizer loaded.")

        # Build converter with injected models
        converter = WavToParquetConverter(
            config=converter_base.config,
            vad=vad,
            yamnet=yamnet,
            diarizer=diarizer,
        )

        converter.process(input_wav, output_dir)
        click.echo(f"Successfully wrote feature record dataset to: {output_dir}")
        sys.exit(0)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Value Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(3)


# ---------------------------------------------------------------------------
# Legacy shim — kept so that existing tests that import wav_to_parquet
# directly by name from cli continue to work.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
