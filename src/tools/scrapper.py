import os
import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor

def download_one_image(i, base_path):
    try:
        img_data = requests.get(f"https://picsum.photos/800/600", timeout=10).content
        with open(os.path.join(base_path, f"sample_img_{i}.jpg"), "wb") as f:
            f.write(img_data)
    except Exception as e:
        print(f"Erreur image {i}: {e}")

def download_one_pdf(i, base_path):
    try:
        arxiv_id = f"230{random.randint(1,9)}.{random.randint(10000, 11000)}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_data = requests.get(pdf_url, timeout=15).content
        if len(pdf_data) > 1000:
            with open(os.path.join(base_path, f"sample_doc_{i}.pdf"), "wb") as f:
                f.write(pdf_data)
        time.sleep(0.5)
    except Exception as e:
        print(f"Erreur PDF {i}: {e}")

def download_one_wiki(i, base_path, headers, wiki_api_url):
    params = {
        "action": "query", "format": "json", "generator": "random",
        "grnnamespace": 0, "prop": "extracts", "explaintext": True, "exlimit": 1
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
                    filename = f"wiki_{i}.txt"
                    with open(os.path.join(base_path, filename), "w", encoding="utf-8") as f:
                        f.write(content)
        elif response.status_code == 429:
            time.sleep(2)
    except Exception as e:
        print(f"Erreur Wiki {i}: {e}")

def setup_test_data_parallel(base_path="data", target_mb=100):
    target_mb = target_mb * 3 # Avec ces valeurs ont obtient environ 100MB a la generation du dataset
    os.makedirs(base_path, exist_ok=True)
    headers = {'User-Agent': 'RAID'}
    wiki_api_url = "https://fr.wikipedia.org/w/api.php"

    print(f"Démarrage du téléchargement parallèle...")
    
    with ThreadPoolExecutor(max_workers=13) as executor:
        
        print(f"Lancement : Images (4 workers), PDFs (4 workers), Wiki (8 workers)")
        
        with ThreadPoolExecutor(max_workers=4) as img_pool:
            img_pool.map(lambda i: download_one_image(i, base_path), range(target_mb * 4))
            
        with ThreadPoolExecutor(max_workers=1) as pdf_pool:
            pdf_pool.map(lambda i: download_one_pdf(i, base_path), range(target_mb // 5))
            
        with ThreadPoolExecutor(max_workers=8) as wiki_pool:
            wiki_pool.map(lambda i: download_one_wiki(i, base_path, headers, wiki_api_url), range(target_mb * 20))

if __name__ == "__main__":
    setup_test_data_parallel(target_mb=100)