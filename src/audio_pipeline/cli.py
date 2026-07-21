"""
Command Line Interface (CLI) for audio-pipeline.
"""

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
    "--output",
    "-o",
    "output_parquet",
    required=True,
    type=click.Path(writable=True, file_okay=True, dir_okay=False),
    help="Path to save output Parquet file.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to configuration YAML file.",
)
def wav_to_parquet(input_wav: str, output_parquet: str, config_path: str):
    """
    Convert a WAV file to a Parquet file with extracted acoustic features and quality metrics.
    """
    click.echo(f"Processing audio file: {input_wav}...")
    try:
        converter = WavToParquetConverter.from_config(config_path)
        converter.process(input_wav, output_parquet)
        click.echo(f"Successfully wrote feature record dataset to: {output_parquet}")
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


if __name__ == "__main__":
    main()
