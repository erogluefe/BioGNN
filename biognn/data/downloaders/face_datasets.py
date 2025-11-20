"""
Face dataset downloaders.
"""

from pathlib import Path
from typing import Optional
from .base import BaseDatasetDownloader


class LFWDownloader(BaseDatasetDownloader):
    """
    Downloader for Labeled Faces in the Wild (LFW) dataset.

    Dataset info:
        - 13,233 images of 5,749 people
        - Images are aligned and cropped
        - Size: ~200MB
        - License: Non-commercial research purposes
        - URL: http://vis-www.cs.umass.edu/lfw/

    Example:
        >>> downloader = LFWDownloader(root='./datasets')
        >>> dataset_path = downloader.download()
    """

    def __init__(self, root: str = './datasets', aligned: bool = True):
        """
        Initialize LFW downloader.

        Args:
            root: Root directory to download to
            aligned: If True, download aligned version (lfw-deepfunneled)
        """
        if aligned:
            urls = [
                (
                    'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz',
                    'lfw-deepfunneled.tgz',
                    '68331da3eb755a505a502b5aacb3c201'  # MD5 checksum
                ),
            ]
            name = 'lfw-deepfunneled'
        else:
            urls = [
                (
                    'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
                    'lfw.tgz',
                    'a17d05bd522c52d84eca14327a23d494'
                ),
            ]
            name = 'lfw'

        super().__init__(name=name, urls=urls, root=root)

    def download(self, force: bool = False) -> Path:
        """Download and extract LFW dataset."""
        dataset_path = super().download(force=force)

        # The extracted folder is inside the extract_to directory
        # Move contents up one level for cleaner structure
        extracted_dir = dataset_path / self.name
        if extracted_dir.exists():
            # Move all contents to parent directory
            for item in extracted_dir.iterdir():
                item.rename(dataset_path / item.name)
            extracted_dir.rmdir()

        print(f"\n📊 LFW Dataset Statistics:")
        print(f"  - Total subjects: ~5,749")
        print(f"  - Total images: ~13,233")
        print(f"  - Image format: JPEG")
        print(f"  - Image size: 250x250 pixels")

        return dataset_path


class CelebADownloader(BaseDatasetDownloader):
    """
    Downloader for CelebA dataset via Google Drive.

    Dataset info:
        - 202,599 face images of 10,177 celebrities
        - 40 binary attribute annotations per image
        - 5 landmark locations per image
        - Size: ~1.3GB (images only)
        - License: Non-commercial research purposes
        - URL: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Note:
        Downloads from Google Drive. Due to Google Drive limitations,
        very large files may require manual download.

    Example:
        >>> downloader = CelebADownloader(root='./datasets')
        >>> dataset_path = downloader.download()
    """

    # Google Drive file IDs for CelebA
    GDRIVE_FILE_IDS = {
        'img_align_celeba': '0B7EVK8r0v71pZjFTYXZWM3FlRnM',  # Aligned images
        'list_attr_celeba': '0B7EVK8r0v71pblRyaVFSWGxPY0U',   # Attributes
        'identity_CelebA': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',  # Identity labels
    }

    def __init__(self, root: str = './datasets', download_attrs: bool = True):
        """
        Initialize CelebA downloader.

        Args:
            root: Root directory to download to
            download_attrs: If True, also download attribute files
        """
        # Note: Google Drive direct download URLs
        urls = [
            (
                f'https://drive.google.com/uc?id={self.GDRIVE_FILE_IDS["img_align_celeba"]}&export=download',
                'img_align_celeba.zip',
                None  # MD5 not provided for Google Drive
            ),
        ]

        if download_attrs:
            urls.append(
                (
                    f'https://drive.google.com/uc?id={self.GDRIVE_FILE_IDS["list_attr_celeba"]}&export=download',
                    'list_attr_celeba.txt',
                    None
                )
            )
            urls.append(
                (
                    f'https://drive.google.com/uc?id={self.GDRIVE_FILE_IDS["identity_CelebA"]}&export=download',
                    'identity_CelebA.txt',
                    None
                )
            )

        super().__init__(name='celeba', urls=urls, root=root)

    def download(self, force: bool = False) -> Path:
        """
        Download and extract CelebA dataset.

        Note:
            Google Drive may require manual download for large files.
            If automatic download fails, please download manually from:
            http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        """
        print("⚠️  Note: CelebA is a large dataset (~1.3GB)")
        print("   Google Drive may require manual download if automatic fails.")
        print("   Manual download: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n")

        try:
            dataset_path = super().download(force=force)

            print(f"\n📊 CelebA Dataset Statistics:")
            print(f"  - Total subjects: 10,177 celebrities")
            print(f"  - Total images: 202,599")
            print(f"  - Image format: JPEG")
            print(f"  - Image size: 218x178 pixels (aligned)")
            print(f"  - Attributes: 40 binary attributes per image")

            return dataset_path

        except Exception as e:
            print(f"\n❌ Automatic download failed: {e}")
            print("\n📥 Manual Download Instructions:")
            print("1. Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            print("2. Download 'Align&Cropped Images' (img_align_celeba.zip)")
            print("3. Optionally download attribute files")
            print(f"4. Extract to: {self.extract_to}")
            raise


class CASIAWebFaceDownloader(BaseDatasetDownloader):
    """
    Placeholder for CASIA-WebFace dataset.

    Dataset info:
        - 494,414 images of 10,575 subjects
        - Size: ~2.6GB
        - License: Research purposes
        - URL: http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html

    Note:
        CASIA-WebFace requires registration and manual download from the official website.
        This class provides instructions only.

    Example:
        >>> downloader = CASIAWebFaceDownloader(root='./datasets')
        >>> downloader.show_instructions()
    """

    def __init__(self, root: str = './datasets'):
        super().__init__(name='casia-webface', urls=[], root=root)

    def download(self, force: bool = False) -> Path:
        """Show download instructions."""
        self.show_instructions()
        return self.extract_to

    def show_instructions(self):
        """Display manual download instructions."""
        print("=" * 70)
        print("CASIA-WebFace Dataset - Manual Download Required")
        print("=" * 70)
        print("\n📋 Dataset Information:")
        print("  - Images: 494,414 face images")
        print("  - Subjects: 10,575 individuals")
        print("  - Size: ~2.6GB")
        print("  - License: Research purposes only")
        print("\n📥 Download Instructions:")
        print("1. Visit: http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html")
        print("2. Register and request download access")
        print("3. Download the dataset archive")
        print(f"4. Extract to: {self.extract_to}")
        print("\n📂 Expected Structure:")
        print(f"  {self.extract_to}/")
        print("    ├── 0000001/")
        print("    │   ├── 001.jpg")
        print("    │   ├── 002.jpg")
        print("    │   └── ...")
        print("    ├── 0000002/")
        print("    └── ...")
        print("=" * 70)
