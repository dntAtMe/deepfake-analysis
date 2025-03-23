import os
import pytest
from utils.dataset.wavefake_extractor import extract_labels, extract_audio_files, wav_path, labels_path

@pytest.fixture
def mock_dataset(tmp_path, monkeypatch):
    """Create a temporary dataset structure for testing."""
    # Create test directory structure
    dataset_dir = tmp_path / "Track1.2" / "train"
    wav_dir = dataset_dir / "wav"
    wav_dir.mkdir(parents=True)
    
    # Create test wav files (empty files for testing)
    test_files = ['ADD2023_T1.2_T_00000001.wav', 'ADD2023_T1.2_T_00000002.wav', 'ADD2023_T1.2_T_00000003.wav']
    for file in test_files:
        (wav_dir / file).touch()
    
    # Create test label file
    label_file = dataset_dir / "label.txt"
    label_content = """ADD2023_T1.2_T_00000001.wav fake
ADD2023_T1.2_T_00000002.wav genuine
ADD2023_T1.2_T_00000003.wav fake"""
    label_file.write_text(label_content)
    
    # Monkeypatch the paths in the module
    monkeypatch.setattr('utils.dataset.wavefake_extractor.wav_path', str(wav_dir))
    monkeypatch.setattr('utils.dataset.wavefake_extractor.labels_path', str(label_file))
    
    return dataset_dir, wav_dir, label_file

def test_extract_labels(mock_dataset):
    """Test the extract_labels function."""
    _, _, _ = mock_dataset
    
    # Get labels
    labels = extract_labels()
    
    assert labels['ADD2023_T1.2_T_00000001.wav'] == 1
    assert labels['ADD2023_T1.2_T_00000002.wav'] == 0
    assert labels['ADD2023_T1.2_T_00000003.wav'] == 1
    assert len(labels) == 3

def test_extract_audio_files(mock_dataset):
    """Test the extract_audio_files function."""
    _, wav_dir, _ = mock_dataset
    
    # Get audio files
    audio_files = extract_audio_files()
    
    # Verify dictionary structure and content
    assert len(audio_files) == 3
    
    # Check if all paths and filenames are correct
    expected_files = [
        'ADD2023_T1.2_T_00000001.wav',
        'ADD2023_T1.2_T_00000002.wav',
        'ADD2023_T1.2_T_00000003.wav'
    ]
    
    for filename in expected_files:
        abs_path = os.path.join(str(wav_dir), filename)
        assert abs_path in audio_files
        assert audio_files[abs_path] == filename

def test_extract_labels_empty_file(tmp_path, monkeypatch):
    """Test extract_labels with an empty label file."""
    # Create empty dataset structure
    dataset_dir = tmp_path / "Track1.2" / "train"
    dataset_dir.mkdir(parents=True)
    label_file = dataset_dir / "label.txt"
    label_file.touch()  # Create empty file
    
    # Monkeypatch the labels_path
    monkeypatch.setattr('utils.dataset.wavefake_extractor.labels_path', str(label_file))
    
    # Get labels
    labels = extract_labels()
    assert len(labels) == 0

def test_extract_audio_files_empty_directory(tmp_path, monkeypatch):
    """Test extract_audio_files with an empty wav directory."""
    # Create empty wav directory
    wav_dir = tmp_path / "Track1.2" / "train" / "wav"
    wav_dir.mkdir(parents=True)
    
    # Monkeypatch the wav_path
    monkeypatch.setattr('utils.dataset.wavefake_extractor.wav_path', str(wav_dir))
    
    # Get audio files
    audio_files = extract_audio_files()
    assert len(audio_files) == 0

def test_extract_audio_files_non_wav_files(tmp_path, monkeypatch):
    """Test extract_audio_files with mixed file types."""
    # Create wav directory with mixed files
    wav_dir = tmp_path / "Track1.2" / "train" / "wav"
    wav_dir.mkdir(parents=True)
    
    # Create test files with different extensions
    (wav_dir / "test1.wav").touch()
    (wav_dir / "test2.txt").touch()
    (wav_dir / "test3.mp3").touch()
    
    # Monkeypatch the wav_path
    monkeypatch.setattr('utils.dataset.wavefake_extractor.wav_path', str(wav_dir))
    
    # Get audio files
    audio_files = extract_audio_files()
    
    # Verify only .wav files are included
    assert len(audio_files) == 1
    abs_path = os.path.join(str(wav_dir), "test1.wav")
    assert abs_path in audio_files
    assert audio_files[abs_path] == "test1.wav" 