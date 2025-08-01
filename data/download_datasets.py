#!/usr/bin/env python3
"""
Dataset Download Script for MLOps Drug Repurposing Project

This script downloads the BIOSNAP and BindingDB datasets and prepares them for DVC tracking.
"""

import os
import requests
import pandas as pd
import zipfile
import gzip
import shutil
from pathlib import Path
from urllib.parse import urlparse
import hashlib

def download_file(url, destination, chunk_size=8192):
    """Download a file from URL to destination with progress."""
    print(f"Downloading {url} -> {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    
    print(f"\n✅ Downloaded: {destination}")
    return destination

def verify_file_integrity(file_path, expected_hash=None):
    """Verify file integrity using SHA256 hash."""
    if not expected_hash:
        return True
    
    print(f"Verifying integrity of {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual_hash = sha256_hash.hexdigest()
    if actual_hash == expected_hash:
        print("✅ File integrity verified")
        return True
    else:
        print(f"❌ File integrity check failed. Expected: {expected_hash}, Got: {actual_hash}")
        return False

def extract_archive(archive_path, extract_to):
    """Extract zip or gzip archives."""
    print(f"Extracting {archive_path} to {extract_to}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix == '.gz':
        with gzip.open(archive_path, 'rb') as f_in:
            output_file = extract_to / archive_path.stem
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    print(f"✅ Extracted to: {extract_to}")

def download_biosnap_data():
    """Download BIOSNAP drug-target interaction data."""
    print("\n🧬 Downloading BIOSNAP dataset...")
    
    biosnap_dir = Path("data/raw/biosnap")
    biosnap_dir.mkdir(parents=True, exist_ok=True)
    
    # BIOSNAP drug-target interaction dataset
    biosnap_urls = {
        "drug_target_interactions.csv": "https://snap.stanford.edu/biodata/datasets/10017/files/ChG-Miner_miner-chem-gene.tsv.gz",
        "drug_metadata.csv": "https://snap.stanford.edu/biodata/datasets/10017/files/ChG-Miner_miner-chemical.tsv.gz",
        "target_metadata.csv": "https://snap.stanford.edu/biodata/datasets/10017/files/ChG-Miner_miner-gene.tsv.gz"
    }
    
    for filename, url in biosnap_urls.items():
        destination = biosnap_dir / filename.replace('.csv', '.tsv.gz')
        
        if not destination.exists():
            try:
                download_file(url, destination)
                # Extract if compressed
                if destination.suffix == '.gz':
                    extract_archive(destination, biosnap_dir)
            except Exception as e:
                print(f"⚠️  Failed to download {url}: {e}")
                # Create placeholder file for now
                placeholder_content = f"# Placeholder for {filename}\n# Original URL: {url}\n# Please download manually if automatic download fails\n"
                with open(biosnap_dir / filename, 'w') as f:
                    f.write(placeholder_content)
        else:
            print(f"✅ Already exists: {destination}")
    
    return biosnap_dir

def download_bindingdb_data():
    """Download BindingDB dataset."""
    print("\n💊 Downloading BindingDB dataset...")
    
    bindingdb_dir = Path("data/raw/bindingdb")
    bindingdb_dir.mkdir(parents=True, exist_ok=True)
    
    # BindingDB dataset (using a subset for demo purposes)
    # Note: Full BindingDB is very large, so we'll create a placeholder
    bindingdb_files = {
        "binding_data.tsv": "# BindingDB binding affinity data\n# Download from: https://www.bindingdb.org/bind/chemsearch/marvin/SDFdownload.jsp\n# This is a placeholder - please download the actual dataset\n",
        "compounds.tsv": "# BindingDB compound metadata\n# Download from: https://www.bindingdb.org/bind/downloads.jsp\n# This is a placeholder - please download the actual dataset\n",
        "targets.tsv": "# BindingDB target metadata\n# Download from: https://www.bindingdb.org/bind/downloads.jsp\n# This is a placeholder - please download the actual dataset\n"
    }
    
    for filename, content in bindingdb_files.items():
        file_path = bindingdb_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"📝 Created placeholder: {file_path}")
        else:
            print(f"✅ Already exists: {file_path}")
    
    return bindingdb_dir

def create_sample_data():
    """Create sample datasets for immediate use."""
    print("\n🔬 Creating sample datasets...")
    
    # Create sample drug-target interaction data
    sample_dir = Path("data/raw/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample drug-target interactions
    sample_interactions = pd.DataFrame({
        'drug_id': ['CHEMBL25', 'CHEMBL53', 'CHEMBL85', 'CHEMBL123', 'CHEMBL456'],
        'target_id': ['ENSP001', 'ENSP002', 'ENSP003', 'ENSP001', 'ENSP002'],
        'interaction_type': ['inhibitor', 'agonist', 'antagonist', 'inhibitor', 'agonist'],
        'binding_affinity': [7.2, 6.8, 5.9, 8.1, 6.5],
        'confidence_score': [0.95, 0.87, 0.76, 0.92, 0.84]
    })
    
    # Sample drug metadata
    sample_drugs = pd.DataFrame({
        'drug_id': ['CHEMBL25', 'CHEMBL53', 'CHEMBL85', 'CHEMBL123', 'CHEMBL456'],
        'drug_name': ['Aspirin', 'Ibuprofen', 'Caffeine', 'Paracetamol', 'Morphine'],
        'smiles': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C(=O)O)C',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(=O)NC1=CC=C(C=C1)O',
            'CN1CCC23C4C(=O)CCC2(C1CC5=C3C(=C(C=C5)O)O4)O'
        ],
        'molecular_weight': [180.16, 206.28, 194.19, 151.16, 285.34],
        'logp': [1.19, 3.97, -0.07, 0.46, 0.89]
    })
    
    # Sample target metadata  
    sample_targets = pd.DataFrame({
        'target_id': ['ENSP001', 'ENSP002', 'ENSP003'],
        'target_name': ['COX-1', 'COX-2', 'Adenosine receptor A2A'],
        'uniprot_id': ['P23219', 'P35354', 'P29274'],
        'target_class': ['enzyme', 'enzyme', 'GPCR'],
        'organism': ['Homo sapiens', 'Homo sapiens', 'Homo sapiens']
    })
    
    # Save sample datasets
    sample_interactions.to_csv(sample_dir / 'drug_target_interactions.csv', index=False)
    sample_drugs.to_csv(sample_dir / 'drug_metadata.csv', index=False)
    sample_targets.to_csv(sample_dir / 'target_metadata.csv', index=False)
    
    print(f"✅ Created sample datasets in {sample_dir}")
    return sample_dir

def main():
    """Main function to download all datasets."""
    print("🚀 Starting dataset download for MLOps Drug Repurposing project")
    
    # Create base directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download datasets
        biosnap_dir = download_biosnap_data()
        bindingdb_dir = download_bindingdb_data()
        sample_dir = create_sample_data()
        
        print("\n📊 Dataset download summary:")
        print(f"✅ BIOSNAP data: {biosnap_dir}")
        print(f"✅ BindingDB data: {bindingdb_dir}")
        print(f"✅ Sample data: {sample_dir}")
        
        print("\n🔄 Next steps:")
        print("1. Add datasets to DVC tracking: dvc add data/raw/")
        print("2. Commit to git: git add data/raw/*.dvc .gitignore")
        print("3. Push to DVC remote: dvc push")
        print("4. Run data validation: python scripts/validate_data.py")
        
    except Exception as e:
        print(f"❌ Error during dataset download: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())