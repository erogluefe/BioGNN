#!/usr/bin/env python3
"""
LUTBIO icin ornek veri olusturucu
Demo amacli sentetik biyometrik veriler olusturur
"""

import os
import numpy as np
from PIL import Image
import wave
import struct
from pathlib import Path


def generate_face_image(subject_id: int, sample_id: int, output_path: str):
    """Sentetik yuz goruntusu olustur"""
    np.random.seed(subject_id * 100 + sample_id)

    # 224x224 RGB goruntu
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200

    # Cilt tonu (kisiye ozel)
    skin_base = [200 + (subject_id * 10) % 55,
                 160 + (subject_id * 7) % 40,
                 140 + (subject_id * 5) % 30]

    # Yuz oval
    center_y, center_x = 112, 112
    for y in range(224):
        for x in range(224):
            dist_y = (y - center_y) / 75
            dist_x = (x - center_x) / 55
            if dist_x**2 + dist_y**2 < 1:
                noise = np.random.randint(-5, 5, 3)
                img[y, x] = np.clip(np.array(skin_base) + noise, 0, 255)

    # Gozler
    eye_y = 90 + np.random.randint(-3, 3)
    for eye_x in [80 + np.random.randint(-3, 3), 144 + np.random.randint(-3, 3)]:
        for dy in range(-10, 11):
            for dx in range(-15, 16):
                if dx**2/(15**2) + dy**2/(10**2) < 1:
                    if dy**2 + (dx*0.5)**2 < 30:
                        # Goz bebegi
                        img[eye_y + dy, eye_x + dx] = [40 + subject_id*5, 30 + subject_id*3, 20]
                    else:
                        # Goz akı
                        img[eye_y + dy, eye_x + dx] = [250, 250, 250]

    # Burun
    nose_width = 6 + subject_id % 4
    for y in range(95, 145):
        for x in range(112 - nose_width, 112 + nose_width):
            if abs(x - 112) < nose_width - (y - 95) * 0.02:
                img[y, x] = np.clip(np.array(skin_base) - 20, 0, 255)

    # Agiz
    mouth_y = 155 + np.random.randint(-2, 2)
    mouth_width = 20 + subject_id % 10
    for y in range(mouth_y - 5, mouth_y + 8):
        for x in range(112 - mouth_width, 112 + mouth_width):
            dist = ((x - 112)/mouth_width)**2 + ((y - mouth_y)/6)**2
            if dist < 1:
                img[y, x] = [150 + subject_id*5, 80, 80]

    # Kaydet
    pil_img = Image.fromarray(img)
    pil_img.save(output_path)


def generate_fingerprint_image(subject_id: int, sample_id: int, output_path: str):
    """Sentetik parmak izi goruntusu olustur"""
    np.random.seed(subject_id * 200 + sample_id)

    # 224x224 grayscale
    img = np.ones((224, 224), dtype=np.uint8) * 220

    # Parmak izi deseni (kisiye ozel)
    pattern_type = subject_id % 3  # 0: loop, 1: whorl, 2: arch

    # Cizgiler
    num_lines = 12 + subject_id % 5
    for i in range(num_lines):
        y_base = i * (200 // num_lines) + 12

        if pattern_type == 0:  # Loop
            amplitude = 25 + np.random.randint(-5, 10)
            frequency = 0.03 + (subject_id % 5) * 0.005
            phase = subject_id * 0.5
        elif pattern_type == 1:  # Whorl
            amplitude = 30 + np.random.randint(-5, 10)
            frequency = 0.025 + (subject_id % 5) * 0.003
            phase = subject_id * 0.3 + i * 0.2
        else:  # Arch
            amplitude = 15 + np.random.randint(-3, 5)
            frequency = 0.02 + (subject_id % 5) * 0.002
            phase = subject_id * 0.2

        for x in range(15, 209):
            y = int(y_base + amplitude * np.sin(frequency * x + phase))
            if 0 <= y < 224:
                thickness = 2 + np.random.randint(0, 2)
                for dy in range(-thickness, thickness + 1):
                    if 0 <= y + dy < 224:
                        img[y + dy, x] = 60 + np.random.randint(-10, 10)

    # Merkez desen
    center_x, center_y = 112, 100 + subject_id % 20
    if pattern_type == 1:  # Whorl
        for r in range(8, 50, 6):
            for angle in range(0, 360, 3):
                x = int(center_x + r * np.cos(np.radians(angle + r*2)))
                y = int(center_y + r * np.sin(np.radians(angle + r*2)))
                if 0 <= x < 224 and 0 <= y < 224:
                    img[y, x] = 70

    # BMP olarak kaydet
    pil_img = Image.fromarray(img)
    pil_img.save(output_path)


def generate_voice_wav(subject_id: int, sample_id: int, output_path: str):
    """Sentetik ses dosyasi olustur"""
    np.random.seed(subject_id * 300 + sample_id)

    # Parametreler
    sample_rate = 16000
    duration = 2.0
    num_samples = int(sample_rate * duration)

    # Temel frekans (kisiye ozel)
    base_freq = 100 + subject_id * 20 + np.random.randint(-10, 10)

    # Zaman
    t = np.linspace(0, duration, num_samples)

    # Ses olustur
    signal = np.zeros(num_samples)

    # Harmonikler
    for harmonic in range(1, 6):
        amplitude = 0.5 / harmonic
        freq = base_freq * harmonic * (1 + np.random.randn() * 0.01)
        signal += amplitude * np.sin(2 * np.pi * freq * t)

    # Formantlar (sesli harf benzeri)
    formants = [300 + subject_id * 30, 800 + subject_id * 50, 2500 + subject_id * 100]
    for formant in formants:
        signal += 0.1 * np.sin(2 * np.pi * formant * t) * np.exp(-t * 2)

    # Modulasyon
    mod_freq = 4 + subject_id % 3
    signal = signal * (1 + 0.3 * np.sin(2 * np.pi * mod_freq * t))

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.7

    # Gurultu ekle
    signal += np.random.randn(num_samples) * 0.02

    # 16-bit PCM'e cevir
    signal_int = np.int16(signal * 32767)

    # WAV dosyasi yaz
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int.tobytes())


def main():
    """Ana fonksiyon"""
    base_dir = Path('/home/user/BioGNN/datasets/lutbio')

    # 6 kisi icin veri olustur
    num_subjects = 6
    face_samples = 3
    finger_samples = 4
    voice_samples = 2

    print("LUTBIO ornek veri olusturuluyor...")

    for subject_id in range(1, num_subjects + 1):
        subject_str = f"{subject_id:03d}"
        subject_dir = base_dir / subject_str

        # Dizinleri olustur
        (subject_dir / 'face').mkdir(parents=True, exist_ok=True)
        (subject_dir / 'finger').mkdir(parents=True, exist_ok=True)
        (subject_dir / 'voice').mkdir(parents=True, exist_ok=True)

        gender = 'male' if subject_id % 2 == 1 else 'female'
        age = 25 + subject_id * 5

        # Yuz goruntuları
        for sample_id in range(1, face_samples + 1):
            filename = f"{subject_str}_{gender}_{age}_face_{sample_id:02d}.jpg"
            filepath = subject_dir / 'face' / filename
            generate_face_image(subject_id, sample_id, str(filepath))
            print(f"  Olusturuldu: {filepath.name}")

        # Parmak izi goruntuları
        for sample_id in range(1, finger_samples + 1):
            filename = f"{subject_str}_{gender}_{age}_finger_{sample_id:02d}.bmp"
            filepath = subject_dir / 'finger' / filename
            generate_fingerprint_image(subject_id, sample_id, str(filepath))
            print(f"  Olusturuldu: {filepath.name}")

        # Ses dosyalari
        for sample_id in range(1, voice_samples + 1):
            filename = f"{subject_str}_{gender}_{age}_voice_{sample_id:02d}.wav"
            filepath = subject_dir / 'voice' / filename
            generate_voice_wav(subject_id, sample_id, str(filepath))
            print(f"  Olusturuldu: {filepath.name}")

        print(f"Kisi {subject_str} tamamlandi.")

    print(f"\nToplam {num_subjects} kisi icin ornek veriler olusturuldu.")
    print(f"Konum: {base_dir}")


if __name__ == '__main__':
    main()
