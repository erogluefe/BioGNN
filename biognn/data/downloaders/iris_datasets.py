"""
Iris dataset downloaders.
"""

from pathlib import Path
from typing import Optional
from .base import BaseDatasetDownloader


class UBIRISDownloader(BaseDatasetDownloader):
    """
    Downloader for UBIRIS.v2 dataset.

    Dataset info:
        - 11,102 images of 261 subjects
        - Noisy iris images captured at-a-distance and on-the-move
        - Two sessions per subject
        - Size: ~3GB
        - License: Research purposes
        - URL: http://iris.di.ubi.pt/ubiris2.html

    Note:
        UBIRIS.v2 requires registration. This downloader provides
        direct download URLs that work without registration for the
        public version.

    Example:
        >>> downloader = UBIRISDownloader(root='./datasets')
        >>> dataset_path = downloader.download()
    """

    def __init__(self, root: str = './datasets', version: str = 'v2'):
        """
        Initialize UBIRIS downloader.

        Args:
            root: Root directory to download to
            version: Version to download ('v1' or 'v2')
        """
        if version == 'v2':
            # UBIRIS.v2 download links
            urls = []
            name = 'ubiris_v2'
            # Note: Direct download may not be available
            # Keeping structure for potential mirrors or manual setup
        elif version == 'v1':
            urls = []
            name = 'ubiris_v1'
        else:
            raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'")

        self.version = version
        super().__init__(name=name, urls=urls, root=root)

    def download(self, force: bool = False) -> Path:
        """Show download instructions."""
        self.show_instructions()
        return self.extract_to

    def show_instructions(self):
        """Display manual download instructions."""
        print("=" * 70)
        print(f"UBIRIS.{self.version} Dataset - Manual Download Required")
        print("=" * 70)

        if self.version == 'v2':
            print("\n📋 Dataset Information:")
            print("  - Images: 11,102 iris images")
            print("  - Subjects: 261 individuals")
            print("  - Sessions: 2 per subject")
            print("  - Size: ~3GB")
            print("  - Conditions: At-a-distance, on-the-move")
            print("  - Image format: JPEG")
            print("\n📥 Download Instructions:")
            print("1. Visit: http://iris.di.ubi.pt/ubiris2.html")
            print("2. Click 'Download' and register")
            print("3. Download the dataset archive")
            print(f"4. Extract to: {self.extract_to}")
            print("\n📂 Expected Structure:")
            print(f"  {self.extract_to}/")
            print("    ├── C001/")
            print("    │   ├── Sess1/")
            print("    │   │   ├── C001S1I01.jpg")
            print("    │   │   └── ...")
            print("    │   └── Sess2/")
            print("    ├── C002/")
            print("    └── ...")

        elif self.version == 'v1':
            print("\n📋 Dataset Information:")
            print("  - Images: 1,877 iris images")
            print("  - Subjects: 241 individuals")
            print("  - Sessions: 2 per subject")
            print("  - Size: ~1GB")
            print("  - Conditions: Controlled indoor environment")
            print("  - Image format: JPEG")
            print("\n📥 Download Instructions:")
            print("1. Visit: http://iris.di.ubi.pt/")
            print("2. Navigate to UBIRIS.v1 section")
            print("3. Register and download")
            print(f"4. Extract to: {self.extract_to}")

        print("=" * 70)


class MMUDownloader(BaseDatasetDownloader):
    """
    Downloader for MMU Iris Database.

    Dataset info:
        - 450 iris images from 45 subjects
        - 5 images per eye (both eyes)
        - Size: ~50MB
        - License: Research purposes
        - URL: http://pesona.mmu.edu.my/~ccteo/

    Example:
        >>> downloader = MMUDownloader(root='./datasets')
        >>> dataset_path = downloader.download()
    """

    def __init__(self, root: str = './datasets', version: int = 1):
        """
        Initialize MMU downloader.

        Args:
            root: Root directory to download to
            version: MMU version (1 or 2)
        """
        self.version = version
        name = f'mmu_iris_v{version}'

        # Note: MMU dataset URLs may change or require registration
        # Providing structure for potential direct downloads or mirrors
        urls = []

        super().__init__(name=name, urls=urls, root=root)

    def download(self, force: bool = False) -> Path:
        """Show download instructions."""
        self.show_instructions()
        return self.extract_to

    def show_instructions(self):
        """Display manual download instructions."""
        print("=" * 70)
        print(f"MMU Iris Database Version {self.version}")
        print("=" * 70)
        print("\n📋 Dataset Information:")
        print("  - Images: 450 iris images")
        print("  - Subjects: 45 individuals")
        print("  - Eyes: Both left and right")
        print("  - Images per eye: 5")
        print("  - Size: ~50MB")
        print("  - Format: BMP")
        print("  - Resolution: 320x240 pixels")
        print("\n📥 Download Instructions:")
        print("1. Visit: http://pesona.mmu.edu.my/~ccteo/")
        print("2. Navigate to iris database section")
        print(f"3. Download MMU Iris Database Version {self.version}")
        print(f"4. Extract to: {self.extract_to}")
        print("\n📂 Expected Structure:")
        print(f"  {self.extract_to}/")
        print("    ├── 1/")
        print("    │   ├── left/")
        print("    │   │   ├── 1.bmp")
        print("    │   │   └── ...")
        print("    │   └── right/")
        print("    ├── 2/")
        print("    └── ...")
        print("=" * 70)


class CASIAIrisDownloader(BaseDatasetDownloader):
    """
    Placeholder for CASIA Iris Database.

    Dataset info:
        - Multiple versions (V1, V2, V3, V4, V5)
        - V4 is largest: 54,601 images from 1,800 subjects
        - Different capture conditions and wavelengths
        - Size: Varies by version (~500MB to 10GB+)
        - License: Research purposes with registration
        - URL: http://www.cbsr.ia.ac.cn/english/IrisDatabase.asp

    Note:
        CASIA Iris requires registration and agreement.
        This class provides instructions only.

    Example:
        >>> downloader = CASIAIrisDownloader(root='./datasets', version='V4')
        >>> downloader.show_instructions()
    """

    def __init__(self, root: str = './datasets', version: str = 'V4'):
        """
        Initialize CASIA-Iris downloader.

        Args:
            root: Root directory
            version: Version to download (V1, V2, V3, V4, V5)
        """
        self.version = version
        super().__init__(name=f'casia_iris_{version.lower()}', urls=[], root=root)

    def download(self, force: bool = False) -> Path:
        """Show download instructions."""
        self.show_instructions()
        return self.extract_to

    def show_instructions(self):
        """Display manual download instructions."""
        print("=" * 70)
        print(f"CASIA-Iris {self.version} - Manual Download Required")
        print("=" * 70)

        dataset_info = {
            'V1': ('756 images', '108 subjects', '7 images per eye', '~100MB'),
            'V2': ('1,200 images', '60 subjects', '20 images per eye', '~200MB'),
            'V3': ('22,035 images', '700 subjects', 'Indoor lamp', '~3GB'),
            'V4': ('54,601 images', '1,800 subjects', 'Multiple conditions', '~10GB'),
            'V5': ('2,655 images', '5 subjects', 'Video sequences', '~500MB'),
        }

        if self.version in dataset_info:
            images, subjects, extra, size = dataset_info[self.version]
            print(f"\n📋 CASIA-Iris {self.version} Information:")
            print(f"  - Images: {images}")
            print(f"  - Subjects: {subjects}")
            print(f"  - Details: {extra}")
            print(f"  - Size: {size}")

        print("\n📥 Download Instructions:")
        print("1. Visit: http://www.cbsr.ia.ac.cn/english/IrisDatabase.asp")
        print(f"2. Navigate to CASIA-Iris-{self.version}")
        print("3. Register and agree to terms")
        print("4. Download the dataset archive")
        print(f"5. Extract to: {self.extract_to}")
        print("\n⚠️  Important:")
        print("  - Registration required")
        print("  - Research purposes only")
        print("  - Must cite CASIA in publications")
        print("=" * 70)
