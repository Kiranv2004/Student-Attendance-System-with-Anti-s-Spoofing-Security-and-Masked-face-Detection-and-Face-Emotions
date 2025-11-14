"""Automated registration smoke test that verifies masked encodings are stored.

Run while the Flask server is up (`python -X utf8 fixed_integrated_attendance_system.py`).
Usage:
    python scripts\test_masked_registration.py --image path/to/your/test/image.jpg --usn TEST001
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import requests
from pymongo import MongoClient


def load_image_as_data_url(image_path: Path, quality: int = 90) -> str:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image as JPEG")

    image_b64 = base64.b64encode(buffer).decode('ascii')
    return f"data:image/jpeg;base64,{image_b64}"


def register_student(base_url: str, usn: str, name: str, image_data_url: str) -> None:
    payload = {
        'usn': usn,
        'name': name,
        'semester': '1',
        'branch': 'CSE',
        'section': 'A',
        'photo_0': image_data_url,
    }

    response = requests.post(f"{base_url}/registration", data=payload, timeout=30)
    if response.status_code not in (200, 302):
        raise RuntimeError(f"Registration failed: status={response.status_code}, body={response.text[:200]}")


def fetch_student(usn: str):
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    db = client.attendance_system
    doc = db.students.find_one({'usn': usn})
    client.close()
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated masked encoding registration test")
    parser.add_argument('--image', type=Path, required=True, help='Path to face image for registration test')
    parser.add_argument('--usn', type=str, required=True, help='Temporary USN to register')
    parser.add_argument('--name', type=str, default='Automation Tester', help='Display name for registration')
    parser.add_argument('--base-url', type=str, default='http://127.0.0.1:5000', help='Base URL of running Flask app')
    parser.add_argument('--cleanup', action='store_true', help='Remove the test student after validation')
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return 1

    print(f"[INFO] Preparing image {image_path}")
    image_data_url = load_image_as_data_url(image_path)

    print(f"[INFO] Registering student {args.usn} at {args.base_url}")
    register_student(args.base_url, args.usn, args.name, image_data_url)

    print("[INFO] Fetching student record from MongoDB")
    student_doc = fetch_student(args.usn)
    if not student_doc:
        print("[ERROR] Student document not found in database")
        return 2

    has_baseline = isinstance(student_doc.get('face_encoding'), list) and len(student_doc['face_encoding']) == 128
    has_masked = isinstance(student_doc.get('face_encoding_masked'), list) and len(student_doc['face_encoding_masked']) == 128

    print(json.dumps({
        'usn': student_doc.get('usn'),
        'name': student_doc.get('name'),
        'has_face_encoding': has_baseline,
        'has_face_encoding_masked': has_masked
    }, indent=2))

    if not has_masked:
        print("[ERROR] Masked face encoding is missing or invalid")
        return 3

    if args.cleanup:
        print("[INFO] Cleaning up test student from database")
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        db.students.delete_one({'usn': args.usn})
        client.close()

    print("[SUCCESS] Masked face encoding stored successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())

