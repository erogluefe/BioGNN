"""
Voice/Speaker dataset downloaders.
"""

from pathlib import Path
from typing import Optional
from .base import BaseDatasetDownloader


class LibriSpeechDownloader(BaseDatasetDownloader):
    """
    Downloader for LibriSpeech ASR corpus.

    Dataset info:
        - Large-scale corpus of read English speech
        - Multiple subsets (dev-clean, test-clean, train-clean-100, etc.)
        - Size: Varies by subset (340MB to 60GB)
        - License: CC BY 4.0
        - URL: https://www.openslr.org/12

    Example:
        >>> downloader = LibriSpeechDownloader(root='./datasets', subset='dev-clean')
        >>> dataset_path = downloader.download()
    """

    SUBSETS = {
        'dev-clean': {
            'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
            'size': '337MB',
            'speakers': 40,
            'md5': '42e2234ba48799c1f50f24a7926300a1'
        },
        'dev-other': {
            'url': 'https://www.openslr.org/resources/12/dev-other.tar.gz',
            'size': '314MB',
            'speakers': 33,
            'md5': 'c8d0bcc9cca99d4f8b62fcc847357931'
        },
        'test-clean': {
            'url': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
            'size': '346MB',
            'speakers': 40,
            'md5': '32fa31d27d2e1cad72775fee3f4849a9'
        },
        'test-other': {
            'url': 'https://www.openslr.org/resources/12/test-other.tar.gz',
            'size': '328MB',
            'speakers': 33,
            'md5': 'fb5a50374b501bb3bac4815ee91d3135'
        },
        'train-clean-100': {
            'url': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
            'size': '6.3GB',
            'speakers': 251,
            'md5': '2a93770f6d5c6c964bc36631d331a522'
        },
        'train-clean-360': {
            'url': 'https://www.openslr.org/resources/12/train-clean-360.tar.gz',
            'size': '23GB',
            'speakers': 921,
            'md5': 'c0e676e450a7ff2f54aeade5171606fa'
        },
    }

    def __init__(self, root: str = './datasets', subset: str = 'dev-clean'):
        """
        Initialize LibriSpeech downloader.

        Args:
            root: Root directory to download to
            subset: Subset to download (dev-clean, test-clean, train-clean-100, etc.)
        """
        if subset not in self.SUBSETS:
            available = ', '.join(self.SUBSETS.keys())
            raise ValueError(f"Unknown subset '{subset}'. Available: {available}")

        self.subset = subset
        subset_info = self.SUBSETS[subset]

        urls = [
            (subset_info['url'], f'{subset}.tar.gz', subset_info.get('md5'))
        ]

        super().__init__(name=f'librispeech_{subset}', urls=urls, root=root)
        self.subset_info = subset_info

    def download(self, force: bool = False) -> Path:
        """Download and extract LibriSpeech subset."""
        dataset_path = super().download(force=force)

        print(f"\n📊 LibriSpeech {self.subset} Statistics:")
        print(f"  - Size: {self.subset_info['size']}")
        print(f"  - Speakers: {self.subset_info['speakers']}")
        print(f"  - Format: FLAC")
        print(f"  - Sample rate: 16kHz")
        print(f"\n📂 Structure:")
        print(f"  {dataset_path}/LibriSpeech/{self.subset}/")
        print(f"    ├── speaker_id_1/")
        print(f"    │   ├── chapter_1/")
        print(f"    │   │   ├── audio_1.flac")
        print(f"    │   │   ├── audio_2.flac")
        print(f"    │   │   └── ...")
        print(f"    └── ...")

        return dataset_path


class VoxCelebDownloader(BaseDatasetDownloader):
    """
    Placeholder for VoxCeleb dataset.

    Dataset info:
        - VoxCeleb1: 100,000+ utterances from 1,251 celebrities
        - VoxCeleb2: 1,000,000+ utterances from 6,112 celebrities
        - Size: VoxCeleb1 ~40GB, VoxCeleb2 ~150GB
        - License: Research purposes with registration
        - URL: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

    Note:
        VoxCeleb requires registration and agreement.
        This class provides instructions only.

    Example:
        >>> downloader = VoxCelebDownloader(root='./datasets', version=1)
        >>> downloader.show_instructions()
    """

    def __init__(self, root: str = './datasets', version: int = 1):
        """
        Initialize VoxCeleb downloader.

        Args:
            root: Root directory
            version: VoxCeleb version (1 or 2)
        """
        if version not in [1, 2]:
            raise ValueError("Version must be 1 or 2")

        self.version = version
        super().__init__(name=f'voxceleb{version}', urls=[], root=root)

    def download(self, force: bool = False) -> Path:
        """Show download instructions."""
        self.show_instructions()
        return self.extract_to

    def show_instructions(self):
        """Display manual download instructions."""
        print("=" * 70)
        print(f"VoxCeleb{self.version} Dataset - Manual Download Required")
        print("=" * 70)

        if self.version == 1:
            print("\n📋 VoxCeleb1 Information:")
            print("  - Utterances: 100,000+")
            print("  - Speakers: 1,251 celebrities")
            print("  - Size: ~40GB")
            print("  - Duration: ~352 hours")
            print("  - Format: m4a, WAV")
        else:
            print("\n📋 VoxCeleb2 Information:")
            print("  - Utterances: 1,000,000+")
            print("  - Speakers: 6,112 celebrities")
            print("  - Size: ~150GB")
            print("  - Duration: ~2,400 hours")
            print("  - Format: m4a, WAV")

        print("\n📥 Download Instructions:")
        print("1. Visit: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
        print(f"2. Navigate to VoxCeleb{self.version} section")
        print("3. Fill in the registration form")
        print("4. Agree to license terms")
        print("5. Download links will be emailed")
        print(f"6. Extract to: {self.extract_to}")
        print("\n⚠️  Important:")
        print("  - Registration required")
        print("  - Academic/research use only")
        print("  - Must cite VoxCeleb in publications")
        print("\n📂 Expected Structure:")
        print(f"  {self.extract_to}/")
        print("    ├── id10001/")
        print("    │   ├── video_1/")
        print("    │   │   ├── 00001.wav")
        print("    │   │   └── ...")
        print("    ├── id10002/")
        print("    └── ...")
        print("=" * 70)


class CommonVoiceDownloader(BaseDatasetDownloader):
    """
    Placeholder for Mozilla Common Voice dataset.

    Dataset info:
        - Multilingual speech corpus
        - Multiple languages and versions
        - Size: Varies by language (1GB to 100GB+)
        - License: CC0 (Public Domain)
        - URL: https://commonvoice.mozilla.org/

    Note:
        Common Voice requires account registration.
        Direct downloads available after login.

    Example:
        >>> downloader = CommonVoiceDownloader(root='./datasets', language='en')
        >>> downloader.show_instructions()
    """

    def __init__(self, root: str = './datasets', language: str = 'en', version: str = '11.0'):
        """
        Initialize Common Voice downloader.

        Args:
            root: Root directory
            language: Language code (en, es, fr, de, etc.)
            version: Dataset version
        """
        self.language = language
        self.version = version
        super().__init__(name=f'common_voice_{language}_v{version}', urls=[], root=root)

    def download(self, force: bool = False) -> Path:
        """Show download instructions."""
        self.show_instructions()
        return self.extract_to

    def show_instructions(self):
        """Display manual download instructions."""
        print("=" * 70)
        print(f"Mozilla Common Voice - {self.language.upper()} v{self.version}")
        print("=" * 70)
        print("\n📋 Dataset Information:")
        print(f"  - Language: {self.language}")
        print(f"  - Version: {self.version}")
        print("  - License: CC0 (Public Domain)")
        print("  - Format: MP3")
        print("  - Sample rate: 48kHz")
        print("\n📥 Download Instructions:")
        print("1. Visit: https://commonvoice.mozilla.org/")
        print("2. Create free account or login")
        print("3. Go to 'Datasets' page")
        print(f"4. Select language: {self.language}")
        print(f"5. Download version {self.version}")
        print(f"6. Extract to: {self.extract_to}")
        print("\n📂 Expected Structure:")
        print(f"  {self.extract_to}/")
        print("    ├── clips/")
        print("    │   ├── common_voice_en_1.mp3")
        print("    │   └── ...")
        print("    ├── train.tsv")
        print("    ├── dev.tsv")
        print("    ├── test.tsv")
        print("    └── validated.tsv")
        print("=" * 70)
