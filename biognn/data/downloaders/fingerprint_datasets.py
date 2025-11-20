"""
Fingerprint dataset downloaders.
"""

from pathlib import Path
from typing import Optional
from .base import BaseDatasetDownloader


class SOCOFingDownloader(BaseDatasetDownloader):
    """
    Downloader for SOCOFing (Sokoto Coventry Fingerprint) dataset.

    Dataset info:
        - 6,000 fingerprint images from 600 subjects
        - 10 fingers per subject (left/right: thumb, index, middle, ring, little)
        - Real and synthetically altered versions
        - Size: ~1GB
        - License: Open access for research
        - URL: https://www.kaggle.com/datasets/ruizgara/socofing

    Note:
        Downloads from Kaggle. Requires Kaggle API credentials.
        Install: pip install kaggle
        Setup: https://github.com/Kaggle/kaggle-api#api-credentials

    Example:
        >>> downloader = SOCOFingDownloader(root='./datasets')
        >>> dataset_path = downloader.download()
    """

    KAGGLE_DATASET = 'ruizgara/socofing'

    def __init__(self, root: str = './datasets'):
        """
        Initialize SOCOFing downloader.

        Args:
            root: Root directory to download to
        """
        super().__init__(name='socofing', urls=[], root=root)

    def download(self, force: bool = False) -> Path:
        """
        Download SOCOFing dataset from Kaggle.

        Requires Kaggle API credentials in ~/.kaggle/kaggle.json
        """
        if self._check_exists() and not force:
            print(f"✓ Dataset 'socofing' already exists at {self.extract_to}")
            return self.extract_to

        try:
            import kaggle
        except ImportError:
            print("❌ Kaggle API not installed.")
            print("\n📥 Installation:")
            print("  pip install kaggle")
            print("\n🔑 Setup API credentials:")
            print("  1. Create Kaggle account: https://www.kaggle.com")
            print("  2. Go to: https://www.kaggle.com/settings/account")
            print("  3. Create API token (downloads kaggle.json)")
            print("  4. Place kaggle.json in ~/.kaggle/")
            print("  5. chmod 600 ~/.kaggle/kaggle.json")
            raise

        print(f"Downloading SOCOFing dataset from Kaggle...")
        print(f"Dataset: {self.KAGGLE_DATASET}")
        print(f"Destination: {self.extract_to}")
        print("-" * 60)

        # Download using Kaggle API
        self.extract_to.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(
            self.KAGGLE_DATASET,
            path=str(self.extract_to),
            unzip=True
        )

        print("-" * 60)
        print(f"✓ SOCOFing dataset downloaded successfully!")
        print(f"\n📊 Dataset Statistics:")
        print(f"  - Total subjects: 600")
        print(f"  - Total images: 6,000")
        print(f"  - Images per subject: 10 (all fingers)")
        print(f"  - Image format: BMP")
        print(f"  - Resolution: 96 DPI")
        print(f"\n📂 Structure:")
        print(f"  {self.extract_to}/Real/ - Original fingerprints")
        print(f"  {self.extract_to}/Altered/ - Altered fingerprints")

        return self.extract_to


class FVC2004Downloader(BaseDatasetDownloader):
    """
    Placeholder for FVC2004 (Fingerprint Verification Competition) dataset.

    Dataset info:
        - 4 databases (DB1, DB2, DB3, DB4)
        - 800 impressions from 100 fingers per database
        - Different sensor types
        - Size: ~500MB
        - License: Academic research only
        - URL: http://bias.csr.unibo.it/fvc2004/

    Note:
        FVC2004 requires registration and manual download from the official website.
        This class provides instructions only.

    Example:
        >>> downloader = FVC2004Downloader(root='./datasets')
        >>> downloader.show_instructions()
    """

    def __init__(self, root: str = './datasets', database: str = 'DB1_B'):
        """
        Initialize FVC2004 downloader.

        Args:
            root: Root directory
            database: Which database to download (DB1_B, DB2_B, DB3_B, DB4_B)
        """
        self.database = database
        super().__init__(name=f'fvc2004_{database.lower()}', urls=[], root=root)

    def download(self, force: bool = False) -> Path:
        """Show download instructions."""
        self.show_instructions()
        return self.extract_to

    def show_instructions(self):
        """Display manual download instructions."""
        print("=" * 70)
        print(f"FVC2004 {self.database} - Manual Download Required")
        print("=" * 70)
        print("\n📋 Dataset Information:")
        print("  - Images: 800 impressions (100 fingers × 8 impressions)")
        print("  - Resolution: 500 DPI")
        print("  - Format: TIFF")
        print("  - Size: ~100-150MB per database")
        print(f"  - Database: {self.database}")
        print("\n📥 Download Instructions:")
        print("1. Visit: http://bias.csr.unibo.it/fvc2004/")
        print("2. Go to 'Downloads' section")
        print("3. Register for academic access")
        print(f"4. Download {self.database} database")
        print(f"5. Extract to: {self.extract_to}")
        print("\n📂 Database Descriptions:")
        print("  - DB1_B: Optical sensor")
        print("  - DB2_B: Optical sensor")
        print("  - DB3_B: Thermal sweeping sensor")
        print("  - DB4_B: Synthetic fingerprints (SFinGe)")
        print("\n📂 Expected Structure:")
        print(f"  {self.extract_to}/")
        print("    ├── 1_1.tif  (finger 1, impression 1)")
        print("    ├── 1_2.tif  (finger 1, impression 2)")
        print("    ├── ...")
        print("    └── 100_8.tif  (finger 100, impression 8)")
        print("=" * 70)
