import os
import random
from typing import Tuple, List, Set
import cv2
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64

# Extensions d'images supportées
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def createDump(
    size_mb: int, 
    fileTypes: Set[str], 
    seed: int, 
    filePath: str, 
    failRate: float = 0.05,
    export: bool = True, 
    autoReduce: bool = True
) -> Tuple[bytearray, List[Tuple[str, int, int, str]]]:
    """
    Simule le chargement de fichiers dans un buffer RAM, avec décodage d'image 
    et chiffrement AES optionnels.
    """

    # 1. Allocation de la mémoire (RAM)
    total_bytes = size_mb * 1024 * 1024
    ram = bytearray(total_bytes)
    ram_size = len(ram)
    
    offset = 0
    rng = random.Random(seed)

    # 2. Chargement de la liste des fichiers
    files = loadFilesByType(list(fileTypes), filePath)
    if not files:
        print(f"Warning: No files found in {filePath}")
        return bytearray(), []

    rng.shuffle(files)

    # 3. Préparation du chiffrement
    seed_bytes = str(seed).encode('utf-8')
    salt = b"simulation_static_salt" 
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    aes_key = kdf.derive(seed_bytes) # Clé brute de 32 bytes
    
    # On crée un IV fixe (ou prédictible) pour la simulation
    # ATTENTION : Jamais en prod, mais parfait pour une simulation reproductible
    static_iv = b'\x00' * 16

    currentFailCount = 0
    filePosition = [] # Format: (filename, start, end, type)

    print(f"Starting RAM Dump simulation ({size_mb} MB)...")

    for file in files:
        try:
            file_size_on_disk = os.path.getsize(file)

            # --- ÉTAPE A : Vérifier la place pour le fichier brut ---
            if offset + file_size_on_disk > ram_size:
                currentFailCount += 1
                if currentFailCount / len(files) > failRate:
                    print(f"Fail rate ({failRate}) exceeded. Stopping load.")
                    break
                continue

            # --- ÉTAPE B : Charger le binaire brut (Raw Binary) ---
            with open(file, 'rb') as f:
                f.readinto(memoryview(ram)[offset : offset + file_size_on_disk])
            
            raw_start = offset
            raw_end = offset + file_size_on_disk
            filePosition.append((file, raw_start, raw_end, "BINARY"))
            
            offset += file_size_on_disk

            # --- ÉTAPE C : Chiffrement AES Déterministe ---
            data_to_encrypt = bytes(ram[raw_start:raw_end])
            
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data_to_encrypt) + padder.finalize()
            
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(static_iv))
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            encrypted_len = len(encrypted_data)

            if offset + encrypted_len <= ram_size:
                ram[offset : offset + encrypted_len] = encrypted_data
                filePosition.append((f"{file}", offset, offset + encrypted_len, "ENCRYPTED"))
                offset += encrypted_len
            else:
                print(f"Not enough RAM for encrypted version of {os.path.basename(file)}")

            # --- ÉTAPE D : Décodage d'Image (Optionnel) ---
            _, ext = os.path.splitext(file)
            if ext.lower() in IMAGE_EXTENSIONS:
                np_buffer = np.frombuffer(ram[raw_start:raw_end], dtype=np.uint8)
                img_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    pixel_data = img_rgb.tobytes()
                    data_size = len(pixel_data)

                    if offset + data_size <= ram_size:
                        ram[offset : offset + data_size] = pixel_data
                        filePosition.append((f"{file}", offset, offset + data_size, "DECODED"))
                        offset += data_size
                    else:
                        print(f"Skipping decode for {os.path.basename(file)}: Not enough RAM ({data_size//1024} KB needed)")
                else:
                    print(f"Error: cv2 failed to decode {os.path.basename(file)}")

        except OSError as e:
            print(f"Error reading file {file}: {e}")
            continue

    # --- ÉTAPE E :  Nettoyage final et Export
    if autoReduce:
        ram = ram[:offset]

    if export:
        output_filename = 'ram_dump.bin'
        try:
            with open(output_filename, 'wb') as f:
                f.write(ram)
            print(f"Dump saved: {output_filename} ({len(ram)/1024/1024:.2f} MB)")
        except IOError as e:
            print(f"Failed to export dump: {e}")

    return ram, filePosition

def loadFilesByType(fileTypes: List[str], filePath: str) -> List[str]:
    if not os.path.exists(filePath):
        return []

    matching_files = []
    allowed_extensions = set(ext.lower() for ext in fileTypes)

    for root, _, files in os.walk(filePath):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext and ext.lower() in allowed_extensions:
                matching_files.append(os.path.join(root, file))

    return matching_files

if __name__ == "__main__":
    # --- Configuration du test ---
    size_mb = 50 # Taille du buffer simulé
    fileTypes = {'.txt'} | IMAGE_EXTENSIONS
    seed = 42
    filePath = './data'
    
    if not os.path.exists(filePath):
        os.makedirs(filePath)
        print(f"Dossier {filePath} créé. Mettez-y des images pour tester.")

    ram, filePosition = createDump(
        size_mb=size_mb, 
        fileTypes=fileTypes, 
        seed=seed, 
        filePath=filePath, 
        failRate=0.1,        # 10% d'échec autorisé avant arrêt
    )
    
    # Affichage du rapport
    print(f"\n--- RAM MAP SUMMARY ---")
    print(f"Total objects in RAM: {len(filePosition)} : {len(ram)/1024/1024:.2f} MB used\n")
    # En-tête du tableau
    print("┌" + "─" * 12 + "┬" + "─" * 17 + "┬" + "─" * 15 + "┐")
    print(f"│ {'TYPE':10} │ {'FILENAME':15} │ {'SIZE':13} │")
    print("├" + "─" * 12 + "┼" + "─" * 17 + "┼" + "─" * 15 + "┤")
    
    for name, start, end, type_ in filePosition[:15]:  # Affiche les 15 premiers
        size_bytes = end - start
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes/1024:.2f} KB"
        else:
            size_str = f"{size_bytes/1024/1024:.2f} MB"
        
        filename = os.path.basename(name)
        if len(filename) > 33:
            filename = filename[:30] + "..."
        
        print(f"│ {type_:10} │ {filename:15} │ {size_str:>13} │")
    
    print("└" + "─" * 12 + "┴" + "─" * 17 + "┴" + "─" * 15 + "┘")