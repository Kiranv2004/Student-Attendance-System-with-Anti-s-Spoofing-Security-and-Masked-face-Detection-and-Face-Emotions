import base64
import json
import sys
from pathlib import Path

import cv2
import requests


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    # Note: Sample images folder has been removed - update this path to your test image
    image_path = project_root / "src" / "data" / "student_images" / "test_image.jpg"
    
    # Alternatively, accept image path as command line argument
    import sys
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"[ERROR] Test image not found at {image_path}")
        print(f"[INFO] Please provide a test image path as argument: python {__file__} path/to/image.jpg")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print("[ERROR] Failed to load test image with OpenCV")
        return 1

    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        print("[ERROR] Failed to encode test image as JPEG")
        return 1

    image_b64 = base64.b64encode(buffer).decode("ascii")

    payload = {
        "image": f"data:image/jpeg;base64,{image_b64}",
        "subject": "AutomatedTest",
        "class": "Automation",
        "session_id": "auto-1",
    }

    url = "http://127.0.0.1:5000/process_attendance"
    print(f"[TEST] Sending request to {url}")

    resp = requests.post(url, json=payload, timeout=30)
    print(f"[TEST] Status: {resp.status_code}")

    try:
        resp_json = resp.json()
        print("[TEST] Response JSON:")
        print(json.dumps(resp_json, indent=2))
    except ValueError:
        print("[WARN] Response was not valid JSON:")
        print(resp.text)
        return 1

    return 0 if resp.ok else 2


if __name__ == "__main__":
    sys.exit(main())

