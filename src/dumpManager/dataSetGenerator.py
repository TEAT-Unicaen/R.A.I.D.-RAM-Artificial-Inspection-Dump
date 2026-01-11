import os
import random
import struct

from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import base64
import zlib
import gzip
import cv2
import numpy as np
import json

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
TEXT_EXTENSIONS = {'.txt', '.json', '.xml', '.csv', '.log', '.md'}

class DataProcessor:

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

        # AES avec IV déterministe mais non nul
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=48, 
                         salt=str(seed).encode(), iterations=100000)
        keyMaterial = kdf.derive(b"aes_key_iv")
        self.aes_key = keyMaterial[:32]
        self.static_iv = keyMaterial[32:48]

    def toEncrypted(self, data: bytes) -> bytes:
        padder = padding.PKCS7(128).padder()
        padded = padder.update(data) + padder.finalize()
        encryptor = Cipher(algorithms.AES(self.aes_key), modes.CBC(self.static_iv)).encryptor()
        return encryptor.update(padded) + encryptor.finalize()
    
    def toDecodedImage(self, data: bytes, fragment: bool = True) -> tuple:
        """Retourne (pixels, metadata) ou (None, None)"""
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w, c = img.shape
            rawPxls = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()
            
            metadata = {
                "original_shape": [h, w, c],
                "total_pixels": len(rawPxls)
            }
            
            if fragment and len(rawPxls) > 5000:
                fragSize = self.rng.randint(5000, min(200000, len(rawPxls)))
                start = self.rng.randint(0, len(rawPxls) - fragSize)
                metadata["fragment_start"] = start
                metadata["fragment_size"] = fragSize
                return rawPxls[start : start + fragSize], metadata
            
            metadata["fragment_start"] = 0
            metadata["fragment_size"] = len(rawPxls)
            return rawPxls, metadata
        return None, None
    
    def toBase64(self, data: bytes) -> bytes:
        return base64.b64encode(data)
    
    def toCompressed(self, data: bytes) -> bytes:
        algo = self.rng.choice(['zlib', 'gzip', 'raw_deflate', 'partial'])

        if algo == 'zlib':
            level = self.rng.choice([1, 6, 9])
            return zlib.compress(data, level)
        elif algo == 'gzip':
            level = self.rng.choice([1, 6, 9])
            return gzip.compress(data, compresslevel=level)
        elif algo == 'raw_deflate':
            compressor = zlib.compressobj(wbits=-15)
            return compressor.compress(data) + compressor.flush()
        else:
            compressed = zlib.compress(data)
            if len(compressed) > 10:
                cut = self.rng.randint(len(compressed) // 2, len(compressed) - 5)
                return compressed[:cut]
            return compressed
    
    def generatePointers(self, count: int) -> bytes:
        pointers = [self.rng.getrandbits(48) for _ in range(count)]
        return struct.pack(f'<{count}Q', *pointers)
    
    def generateRandomStrings(self, count: int) -> bytes:
        sample_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-. "
        res = []
        for _ in range(count):
            length = self.rng.randint(10, 50)
            s = "".join(self.rng.choice(sample_chars) for _ in range(length))
            res.append(s.encode() + b'\x00')
        return b"".join(res)
    
class MemoryLayout:

    def __init__(self, totalSize: int, alignment: int = 16):
        self.ram = bytearray(totalSize)
        self.offset = 0
        self.baseAlignement = alignment
        self.metadata = []
        self.rng = None

    def _align(self):
        if self.rng and self.rng.random() > 0.7:
            alignment = self.rng.choice([1, 4, 8, 16, 32, 64])
        else:
            alignment = self.baseAlignement
        
        self.offset = (self.offset + alignment - 1) & ~(alignment - 1)

    def write(self, data: bytes, label: str, originFile: str, extra: dict = None) -> int:
        self._align()
        
        size = len(data)
        magic = self.rng.choice([0x4141, 0xDEAD, 0xBEEF, self.rng.getrandbits(16)])
        flags = self.rng.getrandbits(16)
        header = struct.pack("<IHH", size, flags, magic)
        totalNeeded = len(header) + size

        if self.offset + totalNeeded <= len(self.ram):
            headerStart = self.offset
            self.ram[self.offset : self.offset + 8] = header
            self.offset += 8

            dataStart = self.offset
            self.ram[self.offset: self.offset + size] = data

            entry = {
                "type": label,
                "header_start": headerStart,
                "header_end": dataStart,
                "data_start": dataStart,
                "data_end": dataStart + size,
                "size": size,
                "file": os.path.basename(originFile)
            }

            if extra:
                entry["extra"] = extra

            self.metadata.append(entry)
            self.offset += size
            return totalNeeded
        
        return 0
    
    def addNoise(self, rng: random.Random):
        gap = rng.randint(32, 256)
        if self.offset + gap < len(self.ram):
            noise = bytes([rng.getrandbits(8) if rng.random() > 0.3 else 0x00 for _ in range(gap)])
            self.ram[self.offset : self.offset + gap] = noise
            
            self.metadata.append({
                "type": "NOISE",
                "header_start": self.offset,
                "header_end": self.offset,
                "data_start": self.offset,
                "data_end": self.offset + gap,
                "size": gap,
                "file": "random_noise"
            })
            self.offset += gap

class DumpGenerator:

    def __init__(self, size_mb: int, seed: int):
        self.totalBytes= size_mb * 1024 * 1024
        self.mem = MemoryLayout(self.totalBytes)
        self.proc = DataProcessor(seed)
        self.rng = random.Random(seed)
        self.mem.rng = self.rng
        self.stats = {
            "BINARY_TEXT": 0, "BINARY_IMAGE": 0, "BINARY_OTHER": 0,
            "ENCRYPTED": 0, "DECODED": 0, "BASE64": 0, "COMPRESSED": 0, 
            "SYSTEM": 0, "NOISE": 0
        }

    def _classifyBin(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return "BINARY_IMAGE"
        elif ext in TEXT_EXTENSIONS:
            return "BINARY_TEXT"
        else:
            return "BINARY_OTHER"

    def run(self, files: list, noise: bool = False, noiseLevel: float = 0.5, fragmentation: bool = True, balanceMode: str = "files"):
        self.rng.shuffle(files)

        targetBytesPerType = 0
        if balanceMode == "size":
            targetBytesPerType = self.totalBytes // len([k for k in self.stats.keys() if k not in ["NOISE", "SYSTEM"]])
        
        kernelCount = 0
        for file_path in files:
            if self.mem.offset >= self.totalBytes * 0.95: break # Stop à 95% de la taille max

            ext = os.path.splitext(file_path)[1].lower()
            isImage = ext in IMAGE_EXTENSIONS

            with open(file_path, 'rb') as f:
                fullContent = f.read()

            if fragmentation and len(fullContent) > 1024:
                max_frag = min(500000, len(fullContent))
                frag_size = self.rng.randint(1024, max_frag)
                start = self.rng.randint(0, max(0, len(fullContent) - frag_size))
                content = fullContent[start : start + frag_size]
            else:
                content = fullContent

            tasks = []
            
            binType = self._classifyBin(file_path)
            tasks.append((binType, lambda c=content: (c, {"subtype": "raw"})))
            
            tasks.append(("ENCRYPTED", lambda c=content: (self.proc.toEncrypted(c), {"algorithm": "AES-256-CBC"})))
            
            tasks.append(("BASE64", lambda c=content: (self.proc.toBase64(c), {"encoding": "base64"})))
            
            tasks.append(("COMPRESSED", lambda c=content: (self.proc.toCompressed(c), {"algorithm": "zlib"})))

            # DECODED pour images
            if isImage:
                def decode_task():
                    pixels, meta = self.proc.toDecodedImage(fullContent, fragment=fragmentation)
                    return (pixels, meta) if pixels is not None else (None, None)
                tasks.append(("DECODED", decode_task))

            self.rng.shuffle(tasks)

            for label, func in tasks:
                if balanceMode == "files" or self.stats[label] < targetBytesPerType:
                    data, extra_info = func()
                    if data:
                        written = self.mem.write(data, label, file_path, extra_info)
                        self.stats[label] += written
            
            if self.rng.random() > 0.7:
                sysType = self.rng.choice(['POINTERS', 'STRINGS'])
                sysDat = self.proc.generatePointers(20) if sysType == "POINTERS" else self.proc.generateRandomStrings(10)
                written = self.mem.write(sysDat, "SYSTEM", f"kernel_memory_{kernelCount}", {"system_type": sysType})
                self.stats["SYSTEM"] += written
                kernelCount += 1

            if noise and self.rng.random() > noiseLevel:
                self.mem.addNoise(self.rng)
                self.stats["NOISE"] += self.mem.metadata[-1]["data_end"] - self.mem.metadata[-1]["data_start"]

        return self.mem.ram[:self.mem.offset], self.mem.metadata

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data")
    output_path = os.path.join(base_dir, "output")

    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Erreur : Le dossier {data_path} est introuvable.")
    else:
        files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                 if os.path.isfile(os.path.join(data_path, f))]
        
        print(f"Fichiers sources trouvés : {len(files)}")

        generator = DumpGenerator(size_mb=2, seed=42)
        ram_bin, metadata = generator.run(files, noise=True, noiseLevel=0.2, balanceMode="size")

        bin_file = os.path.join(output_path, "ram_dump.bin")
        meta_file = os.path.join(output_path, "metadata.json")

        with open(bin_file, "wb") as f:
            f.write(ram_bin)
        
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"\n--- RAPPORT DE GÉNÉRATION ---")
        print(f"Fichier Dump : {bin_file} ({len(ram_bin)/1024/1024:.2f} MB)")
        print(f"Fichier Meta : {meta_file}")
        print(f"Segments totaux : {len(metadata)}")
        print("\nRépartition par type (Octets) :")
        for label, val in sorted(generator.stats.items()):
            if val > 0:
                percentage = (val / len(ram_bin) * 100) if len(ram_bin) > 0 else 0
                print(f"- {label:15} : {val/1024:>8.2f} KB ({percentage:.1f}%)")
        
        print("\nStructure RAM (5 premiers segments) :")
        for entry in metadata[:5]:
            extra = f" | {entry.get('extra', {})}" if 'extra' in entry else ""
            print(f"  [H:{entry['header_start']:>8}-{entry['header_end']:>8}] "
                  f"[D:{entry['data_start']:>8}-{entry['data_end']:>8}] | "
                  f"{entry['type']:15} | {entry['file']}{extra}")