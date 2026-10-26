import os
import sys
import zipfile
import io
import requests

URLS = [
    "https://ghproxy.net/https://github.com/deepseek-ai/DeepSeek-VL/archive/refs/heads/main.zip",
    "https://github.com/deepseek-ai/DeepSeek-VL/archive/refs/heads/main.zip"
]

TARGET_DIR = r"C:\workspace\Electronic-Component-Detection-with-Open-Source-VLMs\src"

def fetch_and_extract():
    for url in URLS:
        print(f"Fetching {url}...")
        try:
            resp = requests.get(url, timeout=30)
            print(f"Status: {resp.status_code}, Length: {len(resp.content)}")
            if resp.status_code == 200 and len(resp.content) > 10000:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    for name in z.namelist():
                        if "deepseek_vl/" in name:
                            # Strip the leading folder DeepSeek-VL-main/
                            parts = name.split("deepseek_vl/", 1)
                            if len(parts) == 2:
                                rel_path = os.path.join("deepseek_vl", parts[1])
                                target_path = os.path.join(TARGET_DIR, rel_path)
                                if name.endswith("/"):
                                    os.makedirs(target_path, exist_ok=True)
                                else:
                                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                    with open(target_path, "wb") as f:
                                        f.write(z.read(name))
                print(f"Successfully extracted deepseek_vl package to {TARGET_DIR}!")
                return True
        except Exception as e:
            print(f"Error: {e}")
    return False

if __name__ == "__main__":
    success = fetch_and_extract()
    sys.exit(0 if success else 1)
