import os
import sys
import time
import requests
from tqdm import tqdm

DEST_DIR = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master"
FILE_NAME = "model.safetensors"
TARGET_PATH = os.path.join(DEST_DIR, FILE_NAME)
INCOMPLETE_PATH = os.path.join(DEST_DIR, FILE_NAME + ".incomplete")

URLS = [
    f"https://hf-mirror.com/deepseek-ai/deepseek-vl-1.3b-chat/resolve/main/{FILE_NAME}",
    f"https://www.modelscope.cn/api/v1/models/deepseek-ai/deepseek-vl-1.3b-chat/repo?Revision=master&FilePath={FILE_NAME}"
]

def download_file():
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(TARGET_PATH) and os.path.getsize(TARGET_PATH) > 2_000_000_000:
        print(f"{TARGET_PATH} already exists and is {os.path.getsize(TARGET_PATH):,} bytes. Complete!")
        return True

    for url in URLS:
        print(f"Trying URL: {url}")
        try:
            head = requests.head(url, timeout=10, allow_redirects=True)
            total_size = int(head.headers.get("content-length", 0))
            print(f"File size from header: {total_size:,} bytes ({total_size / (1024*1024):.1f} MB)")

            initial_pos = 0
            headers = {}
            if os.path.exists(INCOMPLETE_PATH):
                initial_pos = os.path.getsize(INCOMPLETE_PATH)
                if initial_pos < total_size:
                    headers["Range"] = f"bytes={initial_pos}-"
                    print(f"Resuming from {initial_pos:,} bytes...")
                else:
                    initial_pos = 0

            resp = requests.get(url, headers=headers, stream=True, timeout=15)
            if resp.status_code not in (200, 206):
                print(f"Unexpected status code {resp.status_code}, trying next source...")
                continue

            mode = "ab" if initial_pos > 0 else "wb"
            with open(INCOMPLETE_PATH, mode) as f, tqdm(
                total=total_size,
                initial=initial_pos,
                unit="B",
                unit_scale=True,
                desc="DeepSeek-VL-1.3B",
                ncols=80
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            if os.path.getsize(INCOMPLETE_PATH) >= total_size:
                if os.path.exists(TARGET_PATH):
                    os.remove(TARGET_PATH)
                os.rename(INCOMPLETE_PATH, TARGET_PATH)
                print(f"\nDownload complete! Saved to {TARGET_PATH}")
                return True
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
            continue

    return False

if __name__ == "__main__":
    success = download_file()
    sys.exit(0 if success else 1)
