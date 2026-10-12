import os
import time
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# The camera ID from your DevTools URL
CAMERA_ID = "56de42f611f398ec0c481298"
BASE_URL = f"https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id={CAMERA_ID}&bg=black&h=200&w=300"

OUTPUT_DIR = "ly_thai_to_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()

# Full browser request headers to avoid 403 Forbidden
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Referer": "https://giaothong.hochiminhcity.gov.vn/",
    "Origin": "https://giaothong.hochiminhcity.gov.vn",
    "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# 1. Warm up the session by visiting the main portal page first (obtains necessary cookies/tokens)
try:
    session.get("https://giaothong.hochiminhcity.gov.vn/", headers=HEADERS, timeout=10, verify=False)
except Exception as e:
    print(f"Warmup notice: {e}")

# 2. Capture snapshots
NUM_FRAMES = 100
INTERVAL = 3

for i in range(NUM_FRAMES):
    timestamp_ms = int(time.time() * 1000)
    url = f"{BASE_URL}&t={timestamp_ms}"
    
    try:
        response = session.get(url, headers=HEADERS, timeout=10, verify=False)
        if response.status_code == 200 and len(response.content) > 1000:
            filepath = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.jpg")
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"[{i+1}/{NUM_FRAMES}] Saved {filepath} ({len(response.content)} bytes)")
        else:
            print(f"Skipped frame {i}: status {response.status_code}")
    except Exception as err:
        print(f"Frame {i} error: {err}")
        
    time.sleep(INTERVAL)