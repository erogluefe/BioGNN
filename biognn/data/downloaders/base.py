"""
Base downloader class for biometric datasets.
"""

import os
import hashlib
import tarfile
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download operations."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar.

        Args:
            b: Number of blocks transferred so far
            bsize: Size of each block (in bytes)
            tsize: Total size (in bytes)
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class BaseDatasetDownloader:
    """
    Base class for dataset downloaders.

    Attributes:
        name: Name of the dataset
        urls: List of (url, filename, md5) tuples to download
        root: Root directory for the dataset
        extract_to: Directory to extract files to (default: root/dataset_name)
    """

    def __init__(
        self,
        name: str,
        urls: List[Tuple[str, str, Optional[str]]],
        root: str = './datasets',
        extract_to: Optional[str] = None,
    ):
        """
        Initialize downloader.

        Args:
            name: Dataset name
            urls: List of (url, filename, md5_checksum) tuples
            root: Root directory to download to
            extract_to: Directory to extract to (if None, uses root/name)
        """
        self.name = name
        self.urls = urls
        self.root = Path(root)
        self.extract_to = Path(extract_to) if extract_to else self.root / name
        self.download_dir = self.root / 'downloads'

    def _check_exists(self) -> bool:
        """Check if dataset already exists."""
        return self.extract_to.exists() and any(self.extract_to.iterdir())

    def _check_md5(self, filepath: Path, md5: str) -> bool:
        """
        Check if file matches MD5 checksum.

        Args:
            filepath: Path to file
            md5: Expected MD5 checksum

        Returns:
            True if checksum matches, False otherwise
        """
        if not filepath.exists():
            return False

        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest() == md5

    def _download_file(self, url: str, filename: str, md5: Optional[str] = None) -> Path:
        """
        Download a file from URL.

        Args:
            url: URL to download from
            filename: Filename to save as
            md5: Optional MD5 checksum to verify

        Returns:
            Path to downloaded file
        """
        self.download_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.download_dir / filename

        # Check if file already exists and is valid
        if filepath.exists():
            if md5 is None or self._check_md5(filepath, md5):
                print(f"✓ {filename} already downloaded and verified")
                return filepath
            else:
                print(f"⚠ {filename} exists but checksum mismatch, re-downloading...")
                filepath.unlink()

        # Download file
        print(f"Downloading {filename} from {url}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urlretrieve(url, filepath, reporthook=t.update_to)

        # Verify checksum
        if md5 is not None:
            if self._check_md5(filepath, md5):
                print(f"✓ Checksum verified for {filename}")
            else:
                filepath.unlink()
                raise RuntimeError(f"Checksum mismatch for {filename}")

        return filepath

    def _extract_archive(self, filepath: Path, extract_to: Optional[Path] = None) -> None:
        """
        Extract archive file.

        Args:
            filepath: Path to archive file
            extract_to: Directory to extract to (default: self.extract_to)
        """
        if extract_to is None:
            extract_to = self.extract_to

        extract_to.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {filepath.name}...")

        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filepath.suffix in ['.tar', '.gz', '.tgz', '.bz2']:
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {filepath.suffix}")

        print(f"✓ Extracted to {extract_to}")

    def download(self, force: bool = False) -> Path:
        """
        Download and extract the dataset.

        Args:
            force: If True, re-download even if dataset exists

        Returns:
            Path to extracted dataset directory
        """
        if self._check_exists() and not force:
            print(f"✓ Dataset '{self.name}' already exists at {self.extract_to}")
            print("  Use force=True to re-download")
            return self.extract_to

        print(f"Downloading dataset: {self.name}")
        print(f"Destination: {self.extract_to}")
        print("-" * 60)

        # Download all files
        downloaded_files = []
        for url, filename, md5 in self.urls:
            filepath = self._download_file(url, filename, md5)
            downloaded_files.append(filepath)

        # Extract archives
        for filepath in downloaded_files:
            if filepath.suffix in ['.zip', '.tar', '.gz', '.tgz', '.bz2']:
                self._extract_archive(filepath)

        print("-" * 60)
        print(f"✓ Dataset '{self.name}' downloaded successfully!")
        print(f"  Location: {self.extract_to}")

        return self.extract_to

    def clean_downloads(self) -> None:
        """Remove downloaded archive files to save space."""
        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)
            print(f"✓ Cleaned download directory: {self.download_dir}")

    def remove(self) -> None:
        """Remove the entire dataset."""
        if self.extract_to.exists():
            shutil.rmtree(self.extract_to)
            print(f"✓ Removed dataset: {self.extract_to}")
