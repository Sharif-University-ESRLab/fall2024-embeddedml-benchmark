import tensorflow_datasets as tfds
import argparse
import os

def download_speech_commands(data_dir):
    print(f"Downloading Speech Commands dataset to '{data_dir}' (if not already present)...")
    
    ds, info = tfds.load('speech_commands', data_dir=data_dir, with_info=True, as_supervised=True)
    
    print("Download and preparation complete.")
    print(info)

def main():
    parser = argparse.ArgumentParser(description="Download Speech Commands v2 Dataset.")
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.expanduser("~"), 'data'),
                        help='Directory to download the data to (default: $HOME/data)')
    args = parser.parse_args()

    download_speech_commands(args.data_dir)

if __name__ == "__main__":
    main()
