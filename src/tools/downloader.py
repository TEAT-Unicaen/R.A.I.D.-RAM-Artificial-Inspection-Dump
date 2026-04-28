import asyncio
import aiohttp
import os
import random
import time

class AsyncGigaDownloader:
    def __init__(self, base_path="./data", target_mb_per_type=500):
        self.base_path = base_path
        self.target_bytes = target_mb_per_type * 1024 * 1024
        self.current_bytes = {"text": 0, "pdf": 0, "image": 0}
        self.done = {"text": False, "pdf": False, "image": False}

        self.global_counter = 0
        self.counter_lock = asyncio.Lock()
        
        for ftype in ["text", "pdf", "image"]:
            os.makedirs(os.path.join(base_path, ftype), exist_ok=True)

    def get_url(self, ftype):
        if ftype == "text":
            bid = random.randint(1, 70000)
            return f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
        elif ftype == "pdf":
            y, m = random.choice(["23", "24"]), f"{random.randint(1, 12):02d}"
            idx = f"{random.randint(1, 3000):04d}"
            return f"https://arxiv.org/pdf/{y}{m}.{idx}.pdf"
        elif ftype == "image":
            rand_id = random.randint(1, 1000)
            return f"https://picsum.photos/seed/{rand_id}/2560/1440"

    async def download_worker(self, session, ftype):
        while not self.done[ftype]:
            url = self.get_url(ftype)
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.read()
                        if len(content) < 2000:
                            continue 
                            
                        async with self.counter_lock:
                            self.global_counter += 1
                            hex_name = hex(self.global_counter)[2:]

                        ext = "txt" if ftype == "text" else ("pdf" if ftype == "pdf" else "jpg")
                        fname = f"{ftype}/{hex_name}.{ext}"
                        
                        filepath = os.path.join(self.base_path, fname)
                        with open(filepath, "wb") as f:
                            f.write(content)
                        
                        self.current_bytes[ftype] += len(content)
                        
                        total_type_mb = self.current_bytes[ftype] / (1024**2)
                        print(f"[+] {ftype.upper()} | +{len(content)/1024:.1f}KB | Progress: {total_type_mb:.1f} / {self.target_bytes/(1024**2):.1f} MB")

                        if self.current_bytes[ftype] >= self.target_bytes:
                            self.done[ftype] = True
                            print(f"--- QUOTA ATTEINT POUR {ftype.upper()} ---")
                    
                    delay = 0.1 if ftype == "image" else 0.4
                    await asyncio.sleep(delay)

            except Exception:
                continue

    async def run(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        print(f"--- Démarrage Fibre (Cible: {self.target_bytes/(1024**2):.1f}MB par catégorie) ---")
        
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = []
            for ftype in ["text", "pdf", "image"]:
                for _ in range(5):
                    tasks.append(self.download_worker(session, ftype))
            
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    downloader = AsyncGigaDownloader(target_mb_per_type=500)
    try:
        asyncio.run(downloader.run())
    except KeyboardInterrupt:
        print("\nSTOP MANUEL - Arrêt en cours...")
    print("\nOutput: ./data")