import os
import pytest
import pandas as pd
from utils.dataset.inthewild_extractor import extract_labels, extract_audio_files, dataset_path

@pytest.fixture
def mock_metadata(tmp_path, monkeypatch):
    """Create a temporary metadata file for testing."""
    # Create test data
    data = {
        'file': ['test1.wav', 'test2.wav', 'test3.wav'],
        'speaker': ['Speaker1', 'Speaker2', 'Speaker3'],
        'label': ['bona-fide', 'spoof', 'bona-fide']
    }
    df = pd.DataFrame(data)
    
    # Create temporary directory structure
    dataset_dir = tmp_path / "release_in_the_wild"
    dataset_dir.mkdir()
    
    # Save metadata file
    test_metadata_path = dataset_dir / "meta.csv"
    df.to_csv(test_metadata_path, index=False)
    
    # Monkeypatch the metadata_path in the module
    monkeypatch.setattr('utils.dataset.inthewild_extractor.metadata_path', str(test_metadata_path))
    
    return dataset_dir, test_metadata_path

def test_extract_labels(mock_metadata):
    """Test the extract_labels function."""
    dataset_dir, _ = mock_metadata
    
    # Get labels
    labels = extract_labels()
    
    # Verify labels
    assert labels['test1.wav'] == 0  # bona-fide -> 0
    assert labels['test2.wav'] == 1  # spoof -> 1
    assert labels['test3.wav'] == 0  # bona-fide -> 0
    assert len(labels) == 3

def test_extract_audio_files(mock_metadata):
    """Test the extract_audio_files function."""
    dataset_dir, _ = mock_metadata
    
    # Get audio files
    audio_files = extract_audio_files()
    
    # Verify file paths and names
    assert len(audio_files) == 3
    assert os.path.join(dataset_path, 'test1.wav') in audio_files
    assert os.path.join(dataset_path, 'test2.wav') in audio_files
    assert os.path.join(dataset_path, 'test3.wav') in audio_files
    
    # Verify mapping
    for abs_path, filename in audio_files.items():
        assert os.path.basename(abs_path) == filename

def test_extract_labels_empty_file(tmp_path, monkeypatch):
    """Test extract_labels with an empty metadata file."""
    # Create empty metadata file
    dataset_dir = tmp_path / "release_in_the_wild"
    dataset_dir.mkdir()
    test_metadata_path = dataset_dir / "meta.csv"
    pd.DataFrame(columns=['file', 'speaker', 'label']).to_csv(test_metadata_path, index=False)
    
    # Monkeypatch the metadata_path in the module
    monkeypatch.setattr('utils.dataset.inthewild_extractor.metadata_path', str(test_metadata_path))
    
    # Get labels
    labels = extract_labels()
    assert len(labels) == 0

def test_extract_audio_files_empty_file(tmp_path, monkeypatch):
    """Test extract_audio_files with an empty metadata file."""
    # Create empty metadata file
    dataset_dir = tmp_path / "release_in_the_wild"
    dataset_dir.mkdir()
    test_metadata_path = dataset_dir / "meta.csv"
    pd.DataFrame(columns=['file', 'speaker', 'label']).to_csv(test_metadata_path, index=False)
    
    # Monkeypatch the metadata_path in the module
    monkeypatch.setattr('utils.dataset.inthewild_extractor.metadata_path', str(test_metadata_path))
    
    # Get audio files
    audio_files = extract_audio_files()
    assert len(audio_files) == 0
