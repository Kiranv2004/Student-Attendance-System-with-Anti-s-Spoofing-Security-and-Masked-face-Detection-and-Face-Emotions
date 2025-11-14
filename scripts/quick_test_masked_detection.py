"""
Quick test script to verify masked face detection is working
Run this while the Flask app is running to test masked face recognition
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import base64
import requests
import json

def test_masked_face_recognition(image_path, student_usn=None):
    """Test masked face recognition via the Flask API"""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return False
    
    # Encode image to base64
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        print("❌ Failed to encode image")
        return False
    
    image_b64 = base64.b64encode(buffer).decode('ascii')
    
    # Prepare request
    url = "http://127.0.0.1:5000/process_attendance"
    payload = {
        "image": f"data:image/jpeg;base64,{image_b64}",
        "subject": "TestSubject",
        "class": "TestClass",
        "session_id": "test-masked-1"
    }
    
    print(f"📤 Sending test image to {url}")
    print(f"   Image: {image_path}")
    if student_usn:
        print(f"   Expected student: {student_usn}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n📊 Recognition Result:")
            print(json.dumps(result, indent=2))
            
            if result.get('recognized') and result.get('student_id'):
                detected_id = result['student_id']
                detected_name = result.get('student_name', 'Unknown')
                confidence = result.get('confidence', 0.0)
                
                print(f"\n✅ Detected: {detected_name} (ID: {detected_id}) - Confidence: {confidence:.3f}")
                
                if student_usn:
                    if detected_id == student_usn:
                        print(f"✅ SUCCESS: Correctly recognized as {student_usn}")
                        return True
                    else:
                        print(f"❌ FAILED: Expected {student_usn}, got {detected_id}")
                        return False
                else:
                    return True
            else:
                print("\n❌ No face recognized")
                return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/quick_test_masked_detection.py <image_path> [expected_student_usn]")
        print("\nExample:")
        print("  python scripts/quick_test_masked_detection.py test_image.jpg 1VE22CS070")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    student_usn = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    success = test_masked_face_recognition(image_path, student_usn)
    sys.exit(0 if success else 1)

