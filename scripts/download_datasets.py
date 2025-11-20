#!/usr/bin/env python3
"""
Dataset downloader CLI for BioGNN.

Usage:
    # List all available datasets
    python scripts/download_datasets.py --list

    # Download specific dataset
    python scripts/download_datasets.py --dataset lfw --root ./datasets

    # Download multiple datasets
    python scripts/download_datasets.py --dataset lfw socofing librispeech --root ./datasets

    # Download with specific options
    python scripts/download_datasets.py --dataset librispeech --subset dev-clean --root ./datasets

    # Force re-download
    python scripts/download_datasets.py --dataset lfw --force

    # Show instructions for manual datasets
    python scripts/download_datasets.py --dataset casia-webface --show-instructions
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from biognn.data.downloaders import (
    get_downloader,
    DATASET_REGISTRY,
    LFWDownloader,
    LibriSpeechDownloader,
    SOCOFingDownloader,
    VoxCelebDownloader,
    UBIRISDownloader,
    CASIAIrisDownloader,
    FVC2004Downloader,
)


def list_datasets():
    """List all available datasets with information."""
    print("=" * 80)
    print("Available Biometric Datasets")
    print("=" * 80)
    print()

    datasets_info = {
        'Face Datasets': {
            'lfw': {
                'name': 'Labeled Faces in the Wild',
                'size': '~200MB',
                'download': 'Automatic',
                'license': 'Non-commercial research'
            },
            'celeba': {
                'name': 'CelebA',
                'size': '~1.3GB',
                'download': 'Automatic (Google Drive)',
                'license': 'Non-commercial research'
            },
            'casia-webface': {
                'name': 'CASIA-WebFace',
                'size': '~2.6GB',
                'download': 'Manual (registration required)',
                'license': 'Research purposes'
            },
        },
        'Fingerprint Datasets': {
            'socofing': {
                'name': 'SOCOFing',
                'size': '~1GB',
                'download': 'Kaggle API (requires credentials)',
                'license': 'Open access'
            },
            'fvc2004': {
                'name': 'FVC2004',
                'size': '~500MB',
                'download': 'Manual (registration required)',
                'license': 'Academic research'
            },
        },
        'Iris Datasets': {
            'ubiris': {
                'name': 'UBIRIS.v2',
                'size': '~3GB',
                'download': 'Manual (registration required)',
                'license': 'Research purposes'
            },
            'casia-iris': {
                'name': 'CASIA-Iris-V4',
                'size': '~10GB',
                'download': 'Manual (registration required)',
                'license': 'Research purposes'
            },
        },
        'Voice Datasets': {
            'librispeech': {
                'name': 'LibriSpeech',
                'size': '340MB - 60GB',
                'download': 'Automatic (multiple subsets)',
                'license': 'CC BY 4.0'
            },
            'voxceleb': {
                'name': 'VoxCeleb1/2',
                'size': '40GB - 150GB',
                'download': 'Manual (registration required)',
                'license': 'Research purposes'
            },
        },
    }

    for category, datasets in datasets_info.items():
        print(f"\n{category}")
        print("-" * 80)
        for dataset_id, info in datasets.items():
            print(f"\n  {dataset_id}")
            print(f"    Name: {info['name']}")
            print(f"    Size: {info['size']}")
            print(f"    Download: {info['download']}")
            print(f"    License: {info['license']}")

    print("\n" + "=" * 80)
    print("\nUsage:")
    print("  python scripts/download_datasets.py --dataset <dataset_id> --root ./datasets")
    print("\nFor datasets requiring manual download, use --show-instructions flag")
    print("=" * 80)
    print()


def download_dataset(dataset_name: str, root: str, force: bool = False, **kwargs):
    """
    Download a specific dataset.

    Args:
        dataset_name: Name of the dataset
        root: Root directory to download to
        force: Force re-download
        **kwargs: Additional dataset-specific arguments
    """
    try:
        # Handle special cases with custom arguments
        if dataset_name == 'librispeech':
            subset = kwargs.get('subset', 'dev-clean')
            downloader = LibriSpeechDownloader(root=root, subset=subset)
            print(f"Downloading LibriSpeech subset: {subset}")
        elif dataset_name == 'lfw':
            aligned = kwargs.get('aligned', True)
            downloader = LFWDownloader(root=root, aligned=aligned)
        elif dataset_name == 'socofing':
            downloader = SOCOFingDownloader(root=root)
        elif dataset_name == 'voxceleb':
            version = kwargs.get('version', 1)
            downloader = VoxCelebDownloader(root=root, version=version)
            downloader.show_instructions()
            return
        elif dataset_name == 'ubiris':
            version = kwargs.get('ubiris_version', 'v2')
            downloader = UBIRISDownloader(root=root, version=version)
            downloader.show_instructions()
            return
        elif dataset_name == 'casia-iris':
            version = kwargs.get('casia_version', 'V4')
            downloader = CASIAIrisDownloader(root=root, version=version)
            downloader.show_instructions()
            return
        elif dataset_name == 'fvc2004':
            database = kwargs.get('database', 'DB1_B')
            downloader = FVC2004Downloader(root=root, database=database)
            downloader.show_instructions()
            return
        else:
            downloader = get_downloader(dataset_name, root=root)

        # Download the dataset
        dataset_path = downloader.download(force=force)

        # Offer to clean up downloads
        if hasattr(downloader, 'download_dir') and downloader.download_dir.exists():
            response = input("\nClean up downloaded archives to save space? [y/N]: ")
            if response.lower() == 'y':
                downloader.clean_downloads()

        return dataset_path

    except Exception as e:
        print(f"\n❌ Error downloading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download biometric datasets for BioGNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        help='Dataset(s) to download (e.g., lfw, celeba, socofing, librispeech)'
    )

    parser.add_argument(
        '--root',
        type=str,
        default='./datasets',
        help='Root directory to download datasets to (default: ./datasets)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )

    parser.add_argument(
        '--show-instructions',
        action='store_true',
        help='Show manual download instructions for datasets requiring registration'
    )

    # Dataset-specific options
    parser.add_argument(
        '--subset',
        type=str,
        default='dev-clean',
        help='LibriSpeech subset (dev-clean, test-clean, train-clean-100, etc.)'
    )

    parser.add_argument(
        '--aligned',
        type=bool,
        default=True,
        help='Download aligned version of LFW (default: True)'
    )

    parser.add_argument(
        '--version',
        type=int,
        choices=[1, 2],
        default=1,
        help='VoxCeleb version (1 or 2)'
    )

    parser.add_argument(
        '--ubiris-version',
        type=str,
        choices=['v1', 'v2'],
        default='v2',
        help='UBIRIS version'
    )

    parser.add_argument(
        '--casia-version',
        type=str,
        choices=['V1', 'V2', 'V3', 'V4', 'V5'],
        default='V4',
        help='CASIA-Iris version'
    )

    parser.add_argument(
        '--database',
        type=str,
        choices=['DB1_B', 'DB2_B', 'DB3_B', 'DB4_B'],
        default='DB1_B',
        help='FVC2004 database'
    )

    args = parser.parse_args()

    # Show list of datasets
    if args.list:
        list_datasets()
        return

    # Check if dataset is specified
    if not args.dataset:
        parser.print_help()
        print("\nError: Please specify --dataset or use --list to see available datasets")
        sys.exit(1)

    # Download datasets
    print("=" * 80)
    print("BioGNN Dataset Downloader")
    print("=" * 80)
    print()

    for dataset_name in args.dataset:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}\n")

        kwargs = {
            'subset': args.subset,
            'aligned': args.aligned,
            'version': args.version,
            'ubiris_version': args.ubiris_version,
            'casia_version': args.casia_version,
            'database': args.database,
        }

        download_dataset(dataset_name, args.root, args.force, **kwargs)

    print("\n" + "=" * 80)
    print("✓ All requested datasets processed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
