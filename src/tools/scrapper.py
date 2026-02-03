import os
import requests
import random
import time

def setup_test_data(base_path="data", count_per_type=10):
    os.makedirs(base_path, exist_ok=True)
    
    print(f"Téléchargement de {count_per_type} images...")
    for i in range(count_per_type):
        try:
            img_data = requests.get(f"https://loremflickr.com/800/600?lock={i}", timeout=10).content
            with open(os.path.join(base_path, f"sample_img_{i}.jpg"), "wb") as f:
                f.write(img_data)
        except Exception as e: print(f"Erreur image {i}: {e}")

    print(f"Téléchargement de {count_per_type//10} PDFs...")
    for i in range(count_per_type//10):
        try:
            arxiv_id = f"230{random.randint(1,9)}.{random.randint(10000, 11000)}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_data = requests.get(pdf_url, timeout=15).content
            if len(pdf_data) > 1000:
                with open(os.path.join(base_path, f"sample_doc_{i}.pdf"), "wb") as f:
                    f.write(pdf_data)
            time.sleep(0.2)
        except Exception as e: print(f"Erreur PDF {i}: {e}")

    print(f"Récupération de {count_per_type} articles Wikipédia...")

    headers = {'User-Agent': 'RAID-Bot/1.0 (contact@exemple.com)'}
    wiki_api_url = "https://fr.wikipedia.org/w/api.php"
    
    success_count = 0
    attempts = 0
    max_attempts = count_per_type * 2

    while success_count < count_per_type and attempts < max_attempts:
        attempts += 1
        
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "prop": "extracts",
            "explaintext": True,
            "exlimit": 1
        }

        try:
            response = requests.get(wiki_api_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                
                if pages:
                    page_id = list(pages.keys())[0]
                    content = pages[page_id].get('extract', '')
                    
                    if len(content) > 200:
                        filename = f"wiki_{success_count}.txt"
                        with open(os.path.join(base_path, filename), "w", encoding="utf-8") as f:
                            f.write(content)
                        success_count += 1
                        if success_count % 10 == 0:
                            print(f"Articles récupérés : {success_count}/{count_per_type}")
                
            elif response.status_code == 429:
                print("Trop de requêtes... pause de 5 secondes.")
                time.sleep(5)
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Erreur lors de l'appel Wiki : {e}")
            time.sleep(1)

if __name__ == "__main__":
    setup_test_data(base_path="./data", count_per_type=100)