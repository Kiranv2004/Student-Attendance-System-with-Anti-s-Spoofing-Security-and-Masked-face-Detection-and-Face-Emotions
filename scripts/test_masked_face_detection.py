"""
Automated test script for masked face detection and recognition
Tests the complete pipeline: detection, encoding, and matching
"""

import sys
import os
import cv2
import numpy as np
import base64
import json
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import the face recognition system
from fixed_integrated_attendance_system import (
    FixedWebFaceRecognition,
    detect_faces_with_mediapipe,
    create_upper_face_encoding,
    create_masked_face_encoding,
    MEDIAPIPE_AVAILABLE
)
import face_recognition

def test_face_detection(image_path, description):
    """Test face detection on an image"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return False
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Loaded image: {image.shape}")
    
    # Test 1: Standard face detection
    print("\n[Test 1] Standard face_recognition detection...")
    face_locations = face_recognition.face_locations(rgb_image, model='hog')
    print(f"   Found {len(face_locations)} face(s) with standard method")
    
    # Test 2: MediaPipe detection
    if MEDIAPIPE_AVAILABLE:
        print("\n[Test 2] MediaPipe face detection...")
        mediapipe_locations = detect_faces_with_mediapipe(rgb_image)
        print(f"   Found {len(mediapipe_locations)} face(s) with MediaPipe")
    else:
        print("\n[Test 2] MediaPipe not available")
        mediapipe_locations = []
    
    # Use the best detection result
    if face_locations:
        test_location = face_locations[0]
        detection_method = "standard"
    elif mediapipe_locations:
        test_location = mediapipe_locations[0]
        detection_method = "mediapipe"
    else:
        print("❌ No faces detected with any method")
        return False
    
    print(f"\n✅ Using {detection_method} detection: {test_location}")
    
    # Test 3: Standard encoding
    print("\n[Test 3] Standard face encoding...")
    try:
        standard_encodings = face_recognition.face_encodings(rgb_image, [test_location], num_jitters=1)
        if standard_encodings:
            standard_encoding = standard_encodings[0]
            print(f"   ✅ Standard encoding created: shape={standard_encoding.shape}")
        else:
            print("   ❌ Failed to create standard encoding")
            standard_encoding = None
    except Exception as e:
        print(f"   ❌ Error creating standard encoding: {e}")
        standard_encoding = None
    
    # Test 4: Upper face encoding
    print("\n[Test 4] Upper face encoding (for masked faces)...")
    try:
        upper_encoding = create_upper_face_encoding(rgb_image, test_location)
        if upper_encoding is not None:
            print(f"   ✅ Upper face encoding created: shape={upper_encoding.shape}")
        else:
            print("   ❌ Failed to create upper face encoding")
    except Exception as e:
        print(f"   ❌ Error creating upper face encoding: {e}")
        upper_encoding = None
    
    # Test 5: Masked face encoding
    print("\n[Test 5] Masked face encoding...")
    try:
        masked_encoding = create_masked_face_encoding(rgb_image, test_location)
        if masked_encoding is not None:
            print(f"   ✅ Masked face encoding created: shape={masked_encoding.shape}")
        else:
            print("   ❌ Failed to create masked face encoding")
    except Exception as e:
        print(f"   ❌ Error creating masked face encoding: {e}")
        masked_encoding = None
    
    return {
        'face_location': test_location,
        'standard_encoding': standard_encoding,
        'upper_encoding': upper_encoding,
        'masked_encoding': masked_encoding
    }


def test_encoding_matching(encoding1, encoding2, name1, name2):
    """Test matching between two encodings"""
    if encoding1 is None or encoding2 is None:
        print(f"   ⚠️ Cannot compare: {name1} or {name2} is None")
        return None
    
    try:
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        matches = face_recognition.compare_faces([encoding1], encoding2, tolerance=0.6)
        print(f"   Distance between {name1} and {name2}: {distance:.4f}")
        print(f"   Match (tolerance=0.6): {matches[0]}")
        return distance
    except Exception as e:
        print(f"   ❌ Error comparing encodings: {e}")
        return None


def test_with_registered_student(test_image_path, student_usn):
    """Test recognition against a registered student"""
    print(f"\n{'='*60}")
    print(f"TEST: Recognition against registered student {student_usn}")
    print(f"{'='*60}")
    
    # Initialize face recognition system
    fr_system = FixedWebFaceRecognition()
    
    # Check if student is registered
    student_found = False
    student_encodings = []
    for i, student_id in enumerate(fr_system.known_face_ids):
        if student_id == student_usn:
            student_found = True
            encoding = fr_system.known_face_encodings[i]
            variant = fr_system.known_face_metadata[i].get('variant', 'normal')
            student_encodings.append({
                'encoding': encoding,
                'variant': variant,
                'name': fr_system.known_face_names[i]
            })
            print(f"✅ Found registered encoding: {variant} variant for {fr_system.known_face_names[i]}")
    
    if not student_found:
        print(f"❌ Student {student_usn} not found in registered faces")
        return False
    
    # Load test image
    test_image = cv2.imread(str(test_image_path))
    if test_image is None:
        print(f"❌ Failed to load test image: {test_image_path}")
        return False
    
    rgb_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # Detect face in test image
    face_locations = face_recognition.face_locations(rgb_test, model='hog')
    if not face_locations and MEDIAPIPE_AVAILABLE:
        face_locations = detect_faces_with_mediapipe(rgb_test)
    
    if not face_locations:
        print("❌ No face detected in test image")
        return False
    
    test_location = face_locations[0]
    print(f"✅ Detected face in test image: {test_location}")
    
    # Try standard encoding
    test_encodings = face_recognition.face_encodings(rgb_test, [test_location], num_jitters=1)
    if not test_encodings:
        print("⚠️ Standard encoding failed, trying upper face encoding...")
        upper_encoding = create_upper_face_encoding(rgb_test, test_location)
        if upper_encoding is not None:
            test_encodings = [upper_encoding]
            print("✅ Using upper face encoding")
        else:
            print("❌ Failed to create any encoding")
            return False
    
    test_encoding = test_encodings[0]
    
    # Test recognition
    print("\n[Recognition Test]")
    student_id, student_name, confidence = fr_system._identify_face(test_encoding)
    print(f"   Result: {student_name} (ID: {student_id}) - Confidence: {confidence:.3f}")
    
    # Also try with upper face encoding if standard failed
    if student_id == "Unknown":
        print("\n[Retry with Upper Face Encoding]")
        upper_encoding = create_upper_face_encoding(rgb_test, test_location)
        if upper_encoding is not None:
            student_id2, student_name2, confidence2 = fr_system._identify_face(upper_encoding)
            print(f"   Result: {student_name2} (ID: {student_id2}) - Confidence: {confidence2:.3f}")
            if confidence2 > confidence:
                student_id = student_id2
                student_name = student_name2
                confidence = confidence2
    
    # Compare distances manually
    print("\n[Manual Distance Comparison]")
    for enc_info in student_encodings:
        distance = face_recognition.face_distance([enc_info['encoding']], test_encoding)[0]
        print(f"   Distance to {enc_info['variant']} encoding: {distance:.4f}")
        if enc_info['variant'] == 'masked':
            threshold = 0.9
        else:
            threshold = 0.6
        print(f"   Within threshold ({threshold}): {distance <= threshold}")
    
    success = student_id == student_usn
    if success:
        print(f"\n✅ SUCCESS: Correctly recognized as {student_name}")
    else:
        print(f"\n❌ FAILED: Expected {student_usn}, got {student_id}")
    
    return success


def main():
    """Run all tests"""
    print("="*60)
    print("MASKED FACE DETECTION AUTOMATION TEST")
    print("="*60)
    
    # Test 1: Test with a sample image (if available)
    # Note: Sample images folder has been removed - use your own test images
    # sample_dir = project_root / "face_security" / "images" / "sample"
    # test_images = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
    test_images = []
    
    if test_images:
        test_image = test_images[0]
        result = test_face_detection(test_image, f"Face Detection Test - {test_image.name}")
        
        if result and result.get('standard_encoding') and result.get('upper_encoding'):
            print("\n[Encoding Comparison]")
            test_encoding_matching(
                result['standard_encoding'],
                result['upper_encoding'],
                "Standard",
                "Upper Face"
            )
    
    # Test 2: Test recognition against registered students
    # You can modify this to test with a specific student USN
    print("\n" + "="*60)
    print("To test recognition, run:")
    print("  python scripts/test_masked_face_detection.py <student_usn> <test_image_path>")
    print("="*60)
    
    if len(sys.argv) >= 3:
        student_usn = sys.argv[1]
        test_image_path = Path(sys.argv[2])
        if test_image_path.exists():
            test_with_registered_student(test_image_path, student_usn)
        else:
            print(f"❌ Test image not found: {test_image_path}")
    elif len(sys.argv) == 2:
        # Just test detection on provided image
        test_image_path = Path(sys.argv[1])
        if test_image_path.exists():
            test_face_detection(test_image_path, f"Detection Test - {test_image_path.name}")
        else:
            print(f"❌ Image not found: {test_image_path}")


if __name__ == "__main__":
    main()

