import os
from typing import Dict, List

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/track1_2-train/Track1.2/train"))
metadata_path = os.path.join(dataset_path, "metadata.pt")
labels_path = os.path.join(dataset_path, "label.txt")

def extract_labels() -> Dict[str, int]: pass
def extract_audio_files() -> List[str]: pass