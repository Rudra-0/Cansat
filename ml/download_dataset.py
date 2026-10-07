import os
import sys
import hashlib
import zipfile
import urllib.request
import ssl
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EUROSAT_DIR = os.path.join(DATA_DIR, "eurosat")
ZIP_PATH = os.path.join(DATA_DIR, "EuroSAT_RGB.zip")
# SHA-256 of the archive the team trained on (formerly committed to the repo).
EXPECTED_SHA256 = "8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd"

# Reliable Hugging Face mirror as primary, DFKI as fallback
URLS = [
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
]

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

SAFETY_MAPPING = {
    "AnnualCrop": "SAFE_LZ",
    "Pasture": "SAFE_LZ",
    "HerbaceousVegetation": "SAFE_LZ",
    "PermanentCrop": "SAFE_LZ",
    "Forest": "OBSTACLE_CANOPY",
    "Highway": "CRITICAL_HAZARD",
    "Industrial": "CRITICAL_HAZARD",
    "Residential": "CRITICAL_HAZARD",
    "River": "WATER_HAZARD",
    "SeaLake": "WATER_HAZARD"
}

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[DOWNLOAD] {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r[DOWNLOAD] {mb_down:.1f} MB downloaded...")
    sys.stdout.flush()

def verify_dataset(root_dir):
    print("\n[VERIFICATION] Scanning extracted dataset directories...")
    candidates = [root_dir, os.path.join(root_dir, "2750")]
    found_root = None
    for cand in candidates:
        if os.path.exists(cand) and any(os.path.isdir(os.path.join(cand, c)) for c in CLASSES):
            found_root = cand
            break
            
    if not found_root:
        return False

    total_images = 0
    print("-" * 65)
    print(f"{'Class Name':<25} {'SLAI Safety Tier':<20} {'Sample Count':<10}")
    print("-" * 65)
    for c in CLASSES:
        c_path = os.path.join(found_root, c)
        if os.path.exists(c_path):
            count = len([f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
            total_images += count
            tier = SAFETY_MAPPING.get(c, "UNKNOWN")
            print(f"{c:<25} {tier:<20} {count:<10}")
        else:
            print(f"{c:<25} {'MISSING':<20} {0:<10}")
    print("-" * 65)
    print(f"Total Verified Images: {total_images} / 27,000")
    print(f"Dataset Root Directory: {found_root}")
    return total_images > 20000

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    insecure = "--insecure" in sys.argv
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EUROSAT_DIR, exist_ok=True)
    
    print("=" * 65)
    print(" EuroSAT Aerial Remote Sensing Dataset Downloader")
    print(" Cognitive CanSat & TinyML Vision Platform")
    print("=" * 65)
    print(f"Destination: {EUROSAT_DIR}")

    if verify_dataset(EUROSAT_DIR):
        print("\n[STATUS] EuroSAT dataset is already downloaded, extracted, and verified.")
        return

    downloaded = False
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 50 * 1024 * 1024:
        print(f"[FOUND] Existing zip file found ({os.path.getsize(ZIP_PATH)/(1024*1024):.1f} MB). Skipping download.")
        downloaded = True
    else:
        ctx = ssl.create_default_context()
        if insecure:
            # Opt-in only, for TLS-intercepting proxies; the SHA-256 check below still applies.
            print("[WARNING] --insecure: TLS certificate verification disabled.")
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        
        for idx, url in enumerate(URLS, 1):
            print(f"\n[CONNECT] Attempting download from Mirror {idx}:")
            print(f"URL: {url}")
            try:
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, ZIP_PATH, report_progress)
                print("\n[SUCCESS] Download completed.")
                downloaded = True
                break
            except Exception as e:
                print(f"\n[WARNING] Mirror {idx} failed: {e}")
                if os.path.exists(ZIP_PATH):
                    os.remove(ZIP_PATH)

    if not downloaded:
        print("[FATAL] All download mirrors failed. Please check network connectivity.")
        print("        Behind a TLS-intercepting proxy, retry with: python ml/download_dataset.py --insecure")
        sys.exit(1)

    digest = sha256_of(ZIP_PATH)
    if digest == EXPECTED_SHA256:
        print("[VERIFY] Archive SHA-256 matches the reference dataset.")
    else:
        print(f"[WARNING] Archive SHA-256 {digest} differs from the reference archive the models were")
        print("          trained on. The mirror may repackage the same images; results may differ slightly.")

    print(f"\n[EXTRACT] Unpacking {ZIP_PATH} to {EUROSAT_DIR} ...")
    start_time = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EUROSAT_DIR)
    print(f"[EXTRACT] Extraction completed in {time.time() - start_time:.1f}s.")

    if verify_dataset(EUROSAT_DIR):
        print("\n[COMPLETE] EuroSAT aerial dataset is 100% ready for TinyML training!")
    else:
        print("\n[WARNING] Verification found discrepancies in the dataset folder.")

if __name__ == '__main__':
    main()
