"""
Fixed Integrated Student Attendance System with Web Interface
This fixes all the issues with the web-based attendance system
"""

import os
import cv2
import sys
import time
import hashlib
import numpy as np
import face_recognition
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
import json
import base64
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, session, flash
from flask_compress import Compress
import pandas as pd
from pymongo import MongoClient
from gridfs import GridFS
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
import threading
import webbrowser
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate

# Optional: DeepFace for emotion analysis
EMOTION_AVAILABLE = False
try:
    from deepface import DeepFace  # type: ignore
    EMOTION_AVAILABLE = True
    print("✅ DeepFace available for emotion detection")
except Exception as _e:
    print(f"⚠️ DeepFace not available: {_e}")

# Optional: MediaPipe for better masked face detection
MEDIAPIPE_AVAILABLE = False
mp_face_detection = None
mp_drawing = None
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available for masked face detection")
except Exception as _e:
    print(f"⚠️ MediaPipe not available: {_e}")

# Simple emotion detection fallback using facial landmarks
def detect_emotion_simple(face_crop):
    """Simple emotion detection based on facial features"""
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristic-based emotion detection
        # This is a basic implementation - in practice, you'd use more sophisticated methods
        
        # Calculate brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Simple rules based on image characteristics
        if brightness > 120 and contrast > 30:
            return "happy", 0.7
        elif brightness < 80 and contrast < 20:
            return "sad", 0.6
        elif contrast > 40:
            return "angry", 0.5
        else:
            return "neutral", 0.4
            
    except Exception as e:
        print(f"Simple emotion detection error: {e}")
        return "neutral", 0.3


def _estimate_mask_top(face_location, face_landmarks):
    top, right, bottom, left = face_location
    if face_landmarks:
        nose_bridge = face_landmarks.get('nose_bridge')
        if nose_bridge:
            # Use the upper portion of the nose bridge as the mask starting point
            bridge_points_y = [point[1] for point in nose_bridge[:2]]
            if bridge_points_y:
                return max(top, min(int(sum(bridge_points_y) / len(bridge_points_y)), bottom))
    # Fallback to covering the lower half of the face
    return int(top + (bottom - top) * 0.5)


def synthesize_mask_on_face(rgb_image, face_location):
    """Create a synthetic surgical mask overlay on the detected face"""
    try:
        landmarks_list = face_recognition.face_landmarks(rgb_image, [face_location])
        landmarks = landmarks_list[0] if landmarks_list else {}
    except Exception as landmark_error:
        print(f"⚠️ Face landmarks unavailable for mask synthesis: {landmark_error}")
        landmarks = {}

    top, right, bottom, left = face_location
    mask_top = _estimate_mask_top(face_location, landmarks)

    # Ensure coordinates are within bounds
    mask_top = max(top, min(mask_top, bottom))

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    overlay = bgr_image.copy()

    mask_color = (180, 120, 70)  # BGR color for the synthetic mask
    alpha = 0.85

    # Draw rectangle covering lower face
    cv2.rectangle(overlay, (left, mask_top), (right, bottom), mask_color, thickness=-1)

    # Add a curved edge near the nose bridge for more natural appearance
    center = ((left + right) // 2, mask_top)
    axes = (max(1, (right - left) // 2), max(1, (bottom - mask_top)))
    cv2.ellipse(overlay, center, axes, 0, 0, 180, mask_color, thickness=-1)

    # Blend overlay with original image
    masked_bgr = cv2.addWeighted(overlay, alpha, bgr_image, 1 - alpha, 0)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    return masked_rgb


def extract_upper_face_region(rgb_image, face_location):
    """Extract upper portion of face (eyes, nose bridge, forehead) for masked face recognition"""
    try:
        top, right, bottom, left = face_location
        face_height = bottom - top
        face_width = right - left
        
        # Extract upper 60% of the face (eyes, nose bridge, forehead)
        # This region is typically visible even when wearing a mask
        upper_face_top = top
        upper_face_bottom = top + int(face_height * 0.6)
        upper_face_left = left
        upper_face_right = right
        
        # Ensure coordinates are within image bounds
        h, w = rgb_image.shape[:2]
        upper_face_top = max(0, upper_face_top)
        upper_face_bottom = min(h, upper_face_bottom)
        upper_face_left = max(0, upper_face_left)
        upper_face_right = min(w, upper_face_right)
        
        # Extract the upper face region
        upper_face_region = rgb_image[upper_face_top:upper_face_bottom, upper_face_left:upper_face_right]
        
        # Resize to a standard size for better encoding (at least 150x150)
        if upper_face_region.size > 0:
            min_size = 150
            scale_factor = max(min_size / upper_face_region.shape[0], min_size / upper_face_region.shape[1])
            if scale_factor > 1.0:
                new_h = int(upper_face_region.shape[0] * scale_factor)
                new_w = int(upper_face_region.shape[1] * scale_factor)
                upper_face_region = cv2.resize(upper_face_region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            return upper_face_region, (upper_face_top, upper_face_right, upper_face_bottom, upper_face_left)
        else:
            return None, None
    except Exception as e:
        print(f"⚠️ Error extracting upper face region: {e}")
        return None, None


def create_upper_face_encoding(rgb_image, face_location):
    """Create face encoding from upper face region (for masked face recognition)"""
    try:
        top, right, bottom, left = face_location
        face_height = bottom - top
        face_width = right - left
        
        # Extract upper 65% of the face (eyes, nose bridge, forehead)
        upper_face_top = top
        upper_face_bottom = top + int(face_height * 0.65)
        upper_face_left = left
        upper_face_right = right
        
        # Ensure coordinates are within image bounds
        h, w = rgb_image.shape[:2]
        upper_face_top = max(0, upper_face_top)
        upper_face_bottom = min(h, upper_face_bottom)
        upper_face_left = max(0, upper_face_left)
        upper_face_right = min(w, upper_face_right)
        
        # Extract the upper face region
        upper_face_region = rgb_image[upper_face_top:upper_face_bottom, upper_face_left:upper_face_right]
        
        if upper_face_region.size == 0:
            return None
        
        # Resize to ensure minimum size for face detection (at least 150x150)
        min_size = 150
        region_h, region_w = upper_face_region.shape[:2]
        if region_h < min_size or region_w < min_size:
            scale = max(min_size / region_h, min_size / region_w)
            new_h = int(region_h * scale)
            new_w = int(region_w * scale)
            upper_face_region = cv2.resize(upper_face_region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Method 1: Try to detect face in upper region and encode
        upper_locations = face_recognition.face_locations(upper_face_region, model='hog')
        if upper_locations:
            upper_encodings = face_recognition.face_encodings(upper_face_region, upper_locations, num_jitters=3, model='large')
            if upper_encodings:
                print("✅ Created upper face encoding using face detection in upper region")
                return upper_encodings[0]
        
        # Method 2: Create a padded version to simulate full face
        region_h, region_w = upper_face_region.shape[:2]
        # Create a full-size face by padding the bottom
        padded_face = np.zeros((region_h * 2, region_w, 3), dtype=np.uint8)
        padded_face[:region_h, :, :] = upper_face_region
        # Mirror the bottom part of upper region to fill the lower half
        mirror_start = max(0, region_h - region_h // 3)
        padded_face[region_h:, :, :] = upper_face_region[mirror_start:, :, :]
        
        padded_locations = face_recognition.face_locations(padded_face, model='hog')
        if padded_locations:
            padded_encodings = face_recognition.face_encodings(padded_face, padded_locations, num_jitters=3, model='large')
            if padded_encodings:
                print("✅ Created upper face encoding using padded region")
                return padded_encodings[0]
        
        # Method 3: Use the original face location but extract encoding from upper portion
        # Adjust the face location to only include upper portion
        adjusted_location = (top, right, upper_face_bottom, left)
        try:
            # Try encoding with adjusted location
            adjusted_encodings = face_recognition.face_encodings(rgb_image, [adjusted_location], num_jitters=3, model='large')
            if adjusted_encodings:
                print("✅ Created upper face encoding using adjusted face location")
                return adjusted_encodings[0]
        except Exception:
            pass
        
        # Method 4: Use MediaPipe if available to detect in upper region
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_locations = detect_faces_with_mediapipe(upper_face_region)
                if mp_locations:
                    mp_encodings = face_recognition.face_encodings(upper_face_region, mp_locations, num_jitters=3, model='large')
                    if mp_encodings:
                        print("✅ Created upper face encoding using MediaPipe")
                        return mp_encodings[0]
            except Exception:
                pass
        
        return None
    except Exception as e:
        print(f"⚠️ Error creating upper face encoding: {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_faces_with_mediapipe(rgb_image):
    """Detect faces using MediaPipe (works better with masks)"""
    if not MEDIAPIPE_AVAILABLE:
        return []
    
    try:
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_image)
            
            face_locations = []
            if results.detections:
                h, w = rgb_image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    # Convert MediaPipe normalized coordinates to (top, right, bottom, left)
                    left = int(bbox.xmin * w)
                    top = int(bbox.ymin * h)
                    right = int((bbox.xmin + bbox.width) * w)
                    bottom = int((bbox.ymin + bbox.height) * h)
                    
                    # Ensure coordinates are within bounds
                    left = max(0, left)
                    top = max(0, top)
                    right = min(w, right)
                    bottom = min(h, bottom)
                    
                    # Convert to face_recognition format (top, right, bottom, left)
                    face_locations.append((top, right, bottom, left))
            
            return face_locations
    except Exception as e:
        print(f"⚠️ MediaPipe face detection error: {e}")
        return []


def create_masked_face_encoding(rgb_image, face_location):
    """Create a masked face encoding using upper face region extraction"""
    try:
        # First, try to create encoding from upper face region (best for masked faces)
        upper_encoding = create_upper_face_encoding(rgb_image, face_location)
        if upper_encoding is not None:
            print("✅ Created masked encoding using upper face region")
            return upper_encoding
        
        # Fallback: Use synthetic mask augmentation
        masked_rgb = synthesize_mask_on_face(rgb_image, face_location)
        masked_rgb = np.ascontiguousarray(masked_rgb, dtype=np.uint8)
        # Try using the same face location first
        masked_encoding = face_recognition.face_encodings(masked_rgb, [face_location], num_jitters=1)
        if masked_encoding:
            return masked_encoding[0]

        # If encoding failed, attempt to re-detect on masked image
        masked_locations = face_recognition.face_locations(masked_rgb, model='hog')
        if masked_locations:
            masked_encoding = face_recognition.face_encodings(masked_rgb, masked_locations, num_jitters=1)
            if masked_encoding:
                return masked_encoding[0]

        # Try using the CNN model as a fallback (if available)
        try:
            masked_locations_cnn = face_recognition.face_locations(masked_rgb, model='cnn')
            if masked_locations_cnn:
                masked_encoding = face_recognition.face_encodings(masked_rgb, masked_locations_cnn, num_jitters=1)
                if masked_encoding:
                    return masked_encoding[0]
        except Exception as cnn_error:
            print(f"⚠️ CNN-based masked detection failed: {cnn_error}")
    except Exception as mask_error:
        print(f"⚠️ Failed to create masked face encoding: {mask_error}")

    # Final fallback: return the original encoding to ensure a value exists
    try:
        original_encoding = face_recognition.face_encodings(rgb_image, [face_location], num_jitters=1)
        if original_encoding:
            print("⚠️ Using original face encoding as fallback for masked version")
            return original_encoding[0]
    except Exception as fallback_error:
        print(f"⚠️ Failed to use fallback original encoding: {fallback_error}")

    return None

# Add face_security module to path
face_security_dir = os.path.join(os.path.dirname(__file__), 'face_security')
face_security_src_dir = os.path.join(face_security_dir, 'src')
sys.path.insert(0, face_security_src_dir)

# Try to import face_security modules
ANTISPOOFING_AVAILABLE = False
try:
    # Add face_security root directory to Python path
    # This allows the 'src' imports to work properly
    if face_security_dir not in sys.path:
        sys.path.insert(0, face_security_dir)
    
    # Import face_security components
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
    ANTISPOOFING_AVAILABLE = True
    print("✅ Face security module available")
except ImportError as e:
    print(f"⚠️ Face security module not available: {e}")
    print("Face recognition will work without anti-spoofing")
except Exception as e:
    print(f"⚠️ Face security module initialization error: {e}")
    print("Face recognition will work without anti-spoofing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# EMAIL CONFIGURATION
# =============================================================================

# Email configuration - Update these with your SMTP settings
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'smtp_username': os.getenv('SMTP_USERNAME', 'collegeattendance4@gmail.com'),
    'smtp_password': os.getenv('SMTP_PASSWORD', 'rrun gwlj owjv gqep'),  # Gmail App Password
    'from_email': os.getenv('FROM_EMAIL', 'collegeattendance4@gmail.com'),
    'from_name': os.getenv('FROM_NAME', 'College Attendance System'),
    'teacher_emails': os.getenv('TEACHER_EMAILS', 'teacher1@example.com,teacher2@example.com').split(',')
}

# Print email configuration status on startup
print("=" * 60)
print("EMAIL CONFIGURATION STATUS")
print("=" * 60)
print(f"SMTP Server: {EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}")
print(f"From Email: {EMAIL_CONFIG['from_email']}")
print(f"Username: {EMAIL_CONFIG['smtp_username']}")
print(f"Password: {'*' * len(EMAIL_CONFIG['smtp_password']) if EMAIL_CONFIG['smtp_password'] else 'NOT SET'}")
print(f"Email configured: ✅ Ready to send emails")
print("")
print("⚠️  GMAIL SENDING LIMITS:")
print("   - Standard Gmail: 500 emails per day (rolling 24-hour period)")
print("   - Google Workspace: 2,000 emails per day")
print("   - If limit exceeded, sending will be temporarily suspended")
print("=" * 60)

def send_email(to_email, subject, body_html, body_text=None):
    """Send email using SMTP with improved error handling"""
    try:
        # Validate email configuration
        if not EMAIL_CONFIG['smtp_username'] or EMAIL_CONFIG['smtp_username'] in ['your_email@gmail.com', '']:
            logger.warning("Email not configured. Please set SMTP_USERNAME and SMTP_PASSWORD environment variables.")
            print("❌ Email not configured properly")
            return False
        
        if not EMAIL_CONFIG['smtp_password'] or EMAIL_CONFIG['smtp_password'] in ['your_app_password', '']:
            logger.warning("Email password not configured.")
            print("❌ Email password not configured")
            return False
        
        # Validate password is set (not empty)
        if not EMAIL_CONFIG['smtp_password'] or EMAIL_CONFIG['smtp_password'].strip() == '':
            print("❌ Email password is empty")
            return False
        
        print(f"📧 Attempting to send email to {to_email} from {EMAIL_CONFIG['from_email']}")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{EMAIL_CONFIG['from_name']} <{EMAIL_CONFIG['from_email']}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg['Date'] = formatdate(localtime=True)
        
        # Add body
        if body_text:
            part1 = MIMEText(body_text, 'plain')
            msg.attach(part1)
        
        part2 = MIMEText(body_html, 'html')
        msg.attach(part2)
        
        # Send email with detailed error handling
        print(f"🔗 Connecting to SMTP server: {EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}")
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.set_debuglevel(0)  # Set to 1 for verbose debugging
        
        print(f"🔐 Starting TLS...")
        server.starttls()
        
        print(f"🔑 Logging in as {EMAIL_CONFIG['smtp_username']}...")
        # Remove spaces from App Password if present
        password = EMAIL_CONFIG['smtp_password'].replace(' ', '')
        server.login(EMAIL_CONFIG['smtp_username'], password)
        print("✅ Login successful")
        
        print(f"📤 Sending email to {to_email}...")
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent successfully to {to_email}")
        logger.info(f"Email sent successfully to {to_email}")
        return True
        
    except smtplib.SMTPAuthenticationError as auth_error:
        error_msg = f"SMTP Authentication failed: {auth_error}"
        print(f"❌ {error_msg}")
        print("💡 For Gmail, you need to:")
        print("   1. Enable 2-Factor Authentication")
        print("   2. Generate an App Password (not your regular password)")
        print("   3. Use the App Password in EMAIL_CONFIG")
        logger.error(f"Failed to send email to {to_email}: {error_msg}")
        return False
        
    except smtplib.SMTPException as smtp_error:
        error_msg = f"SMTP error: {smtp_error}"
        print(f"❌ {error_msg}")
        logger.error(f"Failed to send email to {to_email}: {error_msg}")
        return False
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ {error_msg}")
        logger.error(f"Failed to send email to {to_email}: {error_msg}")
        import traceback
        traceback.print_exc()
        return False

def send_attendance_confirmation_email(student_email, student_name, subject, class_name, timestamp):
    """Send attendance confirmation email to student"""
    if not student_email:
        logger.warning(f"No email address for student {student_name}")
        return False
    
    subject_line = f"Attendance Marked - {subject}"
    
    body_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
            .info-box {{ background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #3b82f6; }}
            .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>✅ Attendance Confirmed</h2>
            </div>
            <div class="content">
                <p>Dear <strong>{student_name}</strong>,</p>
                <p>Your attendance has been successfully marked.</p>
                
                <div class="info-box">
                    <p><strong>Subject:</strong> {subject}</p>
                    <p><strong>Class:</strong> {class_name}</p>
                    <p><strong>Date & Time:</strong> {timestamp.strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <p>If you have any questions or concerns, please contact your instructor.</p>
                
                <div class="footer">
                    <p>This is an automated message from the Attendance System.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    body_text = f"""
    Attendance Confirmed
    
    Dear {student_name},
    
    Your attendance has been successfully marked.
    
    Subject: {subject}
    Class: {class_name}
    Date & Time: {timestamp.strftime('%B %d, %Y at %I:%M %p')}
    
    If you have any questions or concerns, please contact your instructor.
    
    This is an automated message from the Attendance System.
    """
    
    return send_email(student_email, subject_line, body_html, body_text)

def send_daily_summary_email(teacher_emails, date, attendance_data):
    """Send daily attendance summary to teachers"""
    if not teacher_emails:
        logger.warning("No teacher emails configured")
        return False
    
    subject_line = f"Daily Attendance Summary - {date.strftime('%B %d, %Y')}"
    
    # Prepare attendance data
    present_count = len([a for a in attendance_data if a.get('status') == 'present'])
    absent_count = len([a for a in attendance_data if a.get('status') == 'absent'])
    total_count = len(attendance_data)
    
    # Create HTML table for present students
    present_table = ""
    if present_count > 0:
        present_table = "<table style='width:100%; border-collapse: collapse; margin: 15px 0;'><tr style='background: #d1fae5;'><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>USN</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Name</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Subject</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Time</th></tr>"
        for record in attendance_data:
            if record.get('status') == 'present':
                present_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>{record.get('usn', 'N/A')}</td><td style='padding: 8px; border: 1px solid #ddd;'>{record.get('name', 'N/A')}</td><td style='padding: 8px; border: 1px solid #ddd;'>{record.get('subject', 'N/A')}</td><td style='padding: 8px; border: 1px solid #ddd;'>{record.get('timestamp', 'N/A')}</td></tr>"
        present_table += "</table>"
    
    # Create HTML table for absent students
    absent_table = ""
    if absent_count > 0:
        absent_table = "<table style='width:100%; border-collapse: collapse; margin: 15px 0;'><tr style='background: #fee2e2;'><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>USN</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Name</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Subject</th></tr>"
        for record in attendance_data:
            if record.get('status') == 'absent':
                absent_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>{record.get('usn', 'N/A')}</td><td style='padding: 8px; border: 1px solid #ddd;'>{record.get('name', 'N/A')}</td><td style='padding: 8px; border: 1px solid #ddd;'>{record.get('subject', 'N/A')}</td></tr>"
        absent_table += "</table>"
    
    body_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
            .stat-box {{ flex: 1; background: white; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-box.present {{ border-top: 4px solid #10b981; }}
            .stat-box.absent {{ border-top: 4px solid #ef4444; }}
            .stat-box.total {{ border-top: 4px solid #3b82f6; }}
            .stat-number {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
            .section {{ margin: 30px 0; }}
            .section-title {{ font-size: 20px; font-weight: bold; margin-bottom: 15px; color: #1e40af; }}
            .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>📊 Daily Attendance Summary</h2>
                <p>{date.strftime('%B %d, %Y')}</p>
            </div>
            <div class="content">
                <div class="stats">
                    <div class="stat-box present">
                        <div>Present</div>
                        <div class="stat-number" style="color: #10b981;">{present_count}</div>
                    </div>
                    <div class="stat-box absent">
                        <div>Absent</div>
                        <div class="stat-number" style="color: #ef4444;">{absent_count}</div>
                    </div>
                    <div class="stat-box total">
                        <div>Total</div>
                        <div class="stat-number" style="color: #3b82f6;">{total_count}</div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">✅ Present Students ({present_count})</div>
                    {present_table if present_table else "<p>No students were marked present today.</p>"}
                </div>
                
                <div class="section">
                    <div class="section-title">❌ Absent Students ({absent_count})</div>
                    {absent_table if absent_table else "<p>No students were marked absent today.</p>"}
                </div>
                
                <div class="footer">
                    <p>This is an automated daily summary from the Attendance System.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    body_text = f"""
    Daily Attendance Summary - {date.strftime('%B %d, %Y')}
    
    Statistics:
    - Present: {present_count}
    - Absent: {absent_count}
    - Total: {total_count}
    
    Present Students:
    {chr(10).join([f"{r.get('usn', 'N/A')} - {r.get('name', 'N/A')} - {r.get('subject', 'N/A')} - {r.get('timestamp', 'N/A')}" for r in attendance_data if r.get('status') == 'present']) if present_count > 0 else 'None'}
    
    Absent Students:
    {chr(10).join([f"{r.get('usn', 'N/A')} - {r.get('name', 'N/A')} - {r.get('subject', 'N/A')}" for r in attendance_data if r.get('status') == 'absent']) if absent_count > 0 else 'None'}
    
    This is an automated daily summary from the Attendance System.
    """
    
    # Send to all teacher emails
    success_count = 0
    for email in teacher_emails:
        if email.strip():
            if send_email(email.strip(), subject_line, body_html, body_text):
                success_count += 1
    
    logger.info(f"Daily summary email sent to {success_count}/{len(teacher_emails)} teachers")
    return success_count > 0

# =============================================================================
# FIXED FACE RECOGNITION SYSTEM
# =============================================================================

class FixedWebFaceRecognition:
    """Fixed face recognition system for web interface"""
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.known_face_metadata = []
        
        # Recognition parameters - Optimized for better detection
        self.recognition_tolerance = 0.6  # face_recognition library tolerance
        self.masked_recognition_tolerance = 0.9  # relaxed tolerance for masked variants
        self.min_confidence = 0.3  # Reasonable threshold for normal faces
        self.max_distance = 0.6  # Standard distance threshold for normal faces
        self.masked_max_distance = 1.2  # Lenient distance threshold for masked encodings
        
        # Initialize face security system
        self.anti_spoof_predictor = None
        self.image_cropper = None
        self.model_dir = None
        self.detection_model_dir = None
        self.debug_dump_count = 0
        
        if ANTISPOOFING_AVAILABLE:
            try:
                print("🚀 Initializing face security system...")
                
                # Set model directory paths
                self.model_dir = os.path.join(face_security_dir, 'resources', 'anti_spoof_models')
                self.detection_model_dir = os.path.join(face_security_dir, 'resources', 'detection_model')
                
                # Check if detection model exists
                deploy_file = os.path.join(self.detection_model_dir, 'deploy.prototxt')
                caffemodel_file = os.path.join(self.detection_model_dir, 'Widerface-RetinaFace.caffemodel')
                
                if not os.path.exists(deploy_file) or not os.path.exists(caffemodel_file):
                    print("❌ Detection model files not found")
                    print(f"   Deploy: {deploy_file} - {os.path.exists(deploy_file)}")
                    print(f"   Caffemodel: {caffemodel_file} - {os.path.exists(caffemodel_file)}")
                    self.anti_spoof_predictor = None
                else:
                    # Initialize face security components with base_dir parameter
                    # The base_dir parameter allows the module to find resources automatically
                    self.anti_spoof_predictor = AntiSpoofPredict(device_id=0, base_dir=face_security_dir)
                        self.image_cropper = CropImage()
                    
                    # Check if anti-spoofing models exist
                    if os.path.exists(self.model_dir):
                        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
                        print(f"✅ Loaded {len(model_files)} anti-spoofing models:")
                        for model_file in model_files:
                            print(f"   - {model_file}")
                    else:
                        print("❌ Anti-spoofing models directory not found")
                        self.anti_spoof_predictor = None
                        
                    print("✅ Face security system initialized successfully")
                
            except Exception as e:
                print(f"❌ Error initializing face security system: {e}")
                import traceback
                traceback.print_exc()
                self.anti_spoof_predictor = None
        
        print("Initializing web face recognition system...")
        self.load_known_faces_from_db()

    def load_known_faces_from_db(self):
        """Load known faces from database"""
        try:
            print("Connecting to MongoDB...")
            # Try to connect to MongoDB
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            students_collection = db.students
            
            # Test connection
            client.admin.command('ping')
            print("MongoDB connection successful")
            
            # Load students from database
            students = list(students_collection.find({}))
            print(f"Found {len(students)} students in database")
            
            if len(students) == 0:
                print("No students found in database. Please register some students first.")
                client.close()
                return
            
            # Reset current known faces before loading
            self.known_face_encodings = []
            self.known_face_ids = []
            self.known_face_names = []
            self.known_face_metadata = []
            
            loaded_count = 0
            for student in students:
                student_id = student.get('usn', '')
                student_name = student.get('name', '')
                
                print(f"Processing student: {student_name} (ID: {student_id})")
                
                base_metadata = {
                                'registered_at': student.get('registered_at', ''),
                    'active': student.get('active', True),
                    'data_source': 'database'
                            }

                # Load baseline (unmasked) encoding
                if 'face_encoding' in student and student['face_encoding']:
                    if self._register_known_encoding(student['face_encoding'], student_id, student_name, variant='normal', metadata=base_metadata):
                            loaded_count += 1
                        else:
                    print(f"⚠️ No baseline face encoding found for {student_name} (ID: {student_id})")

                # Load masked encoding if available
                if 'face_encoding_masked' in student and student['face_encoding_masked']:
                    if self._register_known_encoding(student['face_encoding_masked'], student_id, student_name, variant='masked', metadata=base_metadata):
                        loaded_count += 1

                # Support optional list of additional encodings
                if 'additional_encodings' in student and isinstance(student['additional_encodings'], list):
                    for idx, extra_encoding in enumerate(student['additional_encodings'], start=1):
                        variant_label = f'extra_{idx}'
                        if self._register_known_encoding(extra_encoding, student_id, student_name, variant=variant_label, metadata=base_metadata):
                            loaded_count += 1
            
            print(f"✅ Successfully loaded {loaded_count} face encodings for {len(set(self.known_face_ids))} students")
            
            # Diagnostic: Count variants
            normal_count = sum(1 for meta in self.known_face_metadata if meta.get('variant', 'normal') == 'normal')
            masked_count = sum(1 for meta in self.known_face_metadata if meta.get('variant', 'normal') != 'normal')
            print(f"📊 Encoding variants: {normal_count} normal, {masked_count} masked")
            
            if masked_count == 0:
                print("⚠️ WARNING: No masked encodings found! Masked face recognition may not work properly.")
                print("   Make sure students are registered with masked encodings.")
            
            client.close()
            
        except Exception as e:
            print(f"❌ Error loading faces from database: {e}")
            print("Face recognition will work with empty database")
            import traceback
            traceback.print_exc()

    def recognize_faces_improved(self, frame):
        """Improved face recognition for web interface with anti-spoofing"""
        print(f"🔍 Starting face recognition with {len(self.known_face_encodings)} known faces")
        
        if len(self.known_face_encodings) == 0:
            print("❌ No known face encodings loaded!")
            return [], [], [], []
        
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                print("❌ Invalid input frame")
                return [], [], [], []
            
            print(f"📷 Processing frame: {frame.shape}", flush=True)
            print(f"🧾 Frame dtype: {frame.dtype}", flush=True)

            # Ensure frame has 3 channels (convert grayscale or remove alpha if needed)
            if len(frame.shape) == 2:
                print("ℹ️ Converting grayscale frame to BGR", flush=True)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                print("ℹ️ Converting BGRA frame to BGR (dropping alpha channel)", flush=True)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if not frame.flags['C_CONTIGUOUS']:
                print("ℹ️ Making frame C-contiguous", flush=True)
                frame = np.ascontiguousarray(frame)

            # Ensure frame uses 8-bit unsigned integers as required by dlib
            if frame.dtype != np.uint8:
                print(f"⚠️ Unexpected frame dtype {frame.dtype}, converting to uint8", flush=True)
                frame_min, frame_max = frame.min(), frame.max()
                if frame.dtype in [np.float32, np.float64]:
                    # Assume values are either 0-1 or already 0-255
                    scale = 255.0 if frame_max <= 1.0 else 1.0
                    frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                print(f"✅ Frame converted to uint8 (min={frame.min()}, max={frame.max()})", flush=True)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            print(f"📐 RGB frame info: dtype={rgb_frame.dtype}, shape={rgb_frame.shape}, strides={rgb_frame.strides}, contiguous={rgb_frame.flags['C_CONTIGUOUS']}, owndata={rgb_frame.flags['OWNDATA']}", flush=True)

            if not rgb_frame.flags['OWNDATA']:
                print("ℹ️ Making RGB frame own its data", flush=True)
                rgb_frame = rgb_frame.copy(order='C')
                print(f"📐 RGB frame (copied) info: dtype={rgb_frame.dtype}, shape={rgb_frame.shape}, strides={rgb_frame.strides}, contiguous={rgb_frame.flags['C_CONTIGUOUS']}, owndata={rgb_frame.flags['OWNDATA']}", flush=True)

            print("✅ Frame converted to RGB", flush=True)
            
            # Detect faces with optimized settings for better detection
            print("🔍 Detecting faces...", flush=True)

            try:
            face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
            except RuntimeError as rte:
                print(f"⚠️ face_recognition initial call failed: {rte}", flush=True)

                # Attempt fallback conversion using PIL to ensure compatibility
                try:
                    pil_image = Image.fromarray(rgb_frame.astype(np.uint8), mode='RGB')
                    fallback_rgb = np.array(pil_image, dtype=np.uint8)
                    fallback_rgb = np.ascontiguousarray(fallback_rgb, dtype=np.uint8)
                    print(f"📐 Fallback RGB info: dtype={fallback_rgb.dtype}, shape={fallback_rgb.shape}, strides={fallback_rgb.strides}, contiguous={fallback_rgb.flags['C_CONTIGUOUS']}, owndata={fallback_rgb.flags['OWNDATA']}", flush=True)
                    face_locations = face_recognition.face_locations(fallback_rgb, model="hog", number_of_times_to_upsample=1)
                    rgb_frame = fallback_rgb
                except Exception as fallback_error:
                    # Save debug images for troubleshooting (limited to first few occurrences)
                    try:
                        if self.debug_dump_count < 5:
                            debug_dir = os.path.join(os.path.dirname(__file__), 'debug_frames')
                            os.makedirs(debug_dir, exist_ok=True)
                            timestamp = int(time.time())
                            bgr_path = os.path.join(debug_dir, f'debug_frame_{timestamp}_bgr.png')
                            rgb_path = os.path.join(debug_dir, f'debug_frame_{timestamp}_rgb.png')
                            cv2.imwrite(bgr_path, frame)
                            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                            self.debug_dump_count += 1
                            print(f"🧾 Saved debug frames: {bgr_path}, {rgb_path}", flush=True)
                    except Exception as dump_error:
                        print(f"⚠️ Failed to save debug frames: {dump_error}", flush=True)

                    raise fallback_error

            print(f"📍 Found {len(face_locations)} face locations", flush=True)
            
            # If no faces detected with standard method, try MediaPipe (better for masked faces)
            if not face_locations and MEDIAPIPE_AVAILABLE:
                print("🔄 No faces detected with standard method, trying MediaPipe...", flush=True)
                mediapipe_locations = detect_faces_with_mediapipe(rgb_frame)
                if mediapipe_locations:
                    print(f"✅ MediaPipe detected {len(mediapipe_locations)} face(s)", flush=True)
                    face_locations = mediapipe_locations
            
            if not face_locations:
                print("❌ No faces detected in frame (tried both standard and MediaPipe)")
                return [], [], [], []
            
            # Get face encodings with tolerance parameter
            print("🧠 Extracting face encodings...")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
            print(f"✅ Extracted {len(face_encodings)} face encodings")
            
            # If standard encoding fails, try upper face encoding (for masked faces)
            if not face_encodings and len(face_locations) > 0:
                print("🔄 Standard encoding failed, trying upper face encoding for masked faces...", flush=True)
                upper_face_encodings = []
                for face_location in face_locations:
                    upper_encoding = create_upper_face_encoding(rgb_frame, face_location)
                    if upper_encoding is not None:
                        upper_face_encodings.append(upper_encoding)
                    else:
                        # Fallback: try to get encoding from the full face again with different settings
                        try:
                            fallback_encodings = face_recognition.face_encodings(
                                rgb_frame, [face_location], num_jitters=3, model='large'
                            )
                            if fallback_encodings:
                                upper_face_encodings.append(fallback_encodings[0])
                        except Exception as fallback_e:
                            print(f"⚠️ Fallback encoding also failed: {fallback_e}")
                
                if upper_face_encodings:
                    print(f"✅ Extracted {len(upper_face_encodings)} upper face encodings", flush=True)
                    face_encodings = upper_face_encodings
            
            if not face_encodings:
                print("❌ Failed to extract face encodings (tried standard and upper face methods)")
                return [], [], [], []
            
            recognized_ids = []
            recognized_names = []
            confidences = []
            
            print("🔍 Identifying faces...")
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                print(f"  Processing face {i+1}/{len(face_encodings)}")
                
                # Method 1: Standard recognition (normal encodings) - PRIMARY METHOD
                student_id, student_name, confidence = self._identify_face(face_encoding, is_masked_face=False)
                print(f"  Method 1 (standard): {student_name} - Confidence: {confidence:.3f}")
                
                # Only try masked methods if standard recognition failed or confidence is very low
                if student_id == "Unknown" or confidence < 0.3:
                    print(f"  ⚠️ Standard recognition failed/low confidence, trying masked face methods...", flush=True)
                    
                    # Method 2: Standard encoding with masked face flag (tries masked encodings with lenient thresholds)
                    student_id2, student_name2, confidence2 = self._identify_face(face_encoding, is_masked_face=True)
                    print(f"  Method 2 (standard + masked flag): {student_name2} - Confidence: {confidence2:.3f}")
                    
                    # Use Method 2 if it's better (higher confidence or found a match when Method 1 didn't)
                    if (student_id2 != "Unknown" and student_id == "Unknown") or (confidence2 > confidence and student_id2 != "Unknown"):
                        student_id = student_id2
                        student_name = student_name2
                        confidence = confidence2
                        print(f"  ✅ Method 2 improved recognition", flush=True)
                    
                    # Method 3: Upper face encoding (for masked faces) - only if still not found
                    if student_id == "Unknown" or confidence < 0.3:
                        print(f"  ⚠️ Trying upper face encoding for masked face detection...", flush=True)
                        upper_encoding = create_upper_face_encoding(rgb_frame, face_location)
                        if upper_encoding is not None:
                            student_id3, student_name3, confidence3 = self._identify_face(upper_encoding, is_masked_face=True)
                            print(f"  Method 3 (upper face + masked flag): {student_name3} - Confidence: {confidence3:.3f}")
                            
                            # Use Method 3 if it's better
                            if (student_id3 != "Unknown" and student_id == "Unknown") or (confidence3 > confidence and student_id3 != "Unknown"):
                                student_id = student_id3
                                student_name = student_name3
                                confidence = confidence3
                                print(f"  ✅ Upper face encoding provided best match!", flush=True)
                        else:
                            print(f"  ⚠️ Upper face encoding creation failed", flush=True)
                else:
                    print(f"  ✅ Standard recognition successful - using Method 1 result", flush=True)
                
                recognized_ids.append(student_id)
                recognized_names.append(student_name)
                confidences.append(confidence)
                print(f"  🎯 Final Result: {student_name} (ID: {student_id}) - Confidence: {confidence:.3f}")
            
            print(f"✅ Face recognition complete: {len(recognized_ids)} faces processed")
            return face_locations, recognized_ids, recognized_names, confidences
            
        except Exception as e:
            print(f"❌ Error in face recognition: {str(e)}", flush=True)
            if 'frame' in locals() and isinstance(frame, np.ndarray):
                print(f"🧾 Frame debug -> dtype: {frame.dtype}, shape: {frame.shape}, flags: C_CONTIGUOUS={frame.flags['C_CONTIGUOUS']}, OWNDATA={frame.flags['OWNDATA']}", flush=True)
            if 'rgb_frame' in locals() and isinstance(rgb_frame, np.ndarray):
                print(f"🧾 RGB debug -> dtype: {rgb_frame.dtype}, shape: {rgb_frame.shape}, flags: C_CONTIGUOUS={rgb_frame.flags['C_CONTIGUOUS']}, OWNDATA={rgb_frame.flags['OWNDATA']}", flush=True)
            import traceback
            traceback.print_exc()
            return [], [], [], []

    def check_anti_spoofing(self, frame, face_location):
        """Check if face is real or fake using face security module"""
        print(f"🛡️ Starting face security check...")
        
        if not self.anti_spoof_predictor or not self.image_cropper or not self.model_dir:
            print("⚠️ Face security module not available, assuming real face")
            return True, 0.0  # Assume real if anti-spoofing not available
        
        try:
            # Use face security module's own face detection
            print("🔍 Using face security module face detection...")
            bbox = self.anti_spoof_predictor.get_bbox(frame)
            
            if bbox is None:
                print("❌ No face detected by face security module, assuming real face")
                return True, 0.0
            
            print(f"📍 Face security module bbox: {bbox}")
            
            # Use the helper method for anti-spoofing
            return self._test_anti_spoofing_with_bbox(frame, bbox)
            
        except Exception as e:
            print(f"❌ Error in face security detection: {e}")
            import traceback
            traceback.print_exc()
            return True, 0.0  # Assume real on error

    def _test_anti_spoofing_with_bbox(self, frame, bbox):
        """Test anti-spoofing with a given bbox (helper method)"""
        if not self.anti_spoof_predictor or not self.image_cropper or not self.model_dir:
            print("⚠️ Face security module not available, assuming real face")
            return True, 0.0
        
        try:
            print(f"📍 Testing with bbox: {bbox}")
            
            # Validate bbox
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] >= frame.shape[1] or bbox[1] + bbox[3] >= frame.shape[0]:
                print("❌ Invalid bbox, assuming real face")
                return True, 0.0
            
            # Initialize prediction accumulator (same as original test.py)
            prediction = np.zeros((1, 3))
            model_count = 0
            
            # Use all available models for ensemble prediction (exactly like original test.py)
            for model_name in os.listdir(self.model_dir):
                if not model_name.endswith('.pth'):
                    continue
                
                try:
                    print(f"🤖 Processing with model: {model_name}")
                    
                    # Parse the model name to get parameters (same as original)
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    
                    # Prepare image for prediction using CropImage (same as original)
                    param = {
                        "org_img": frame,
                        "bbox": bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    
                    # Handle scale parameter (same as original test.py)
                    if scale is None:
                        param["crop"] = False
                    
                    # Crop and prepare image
                    img = self.image_cropper.crop(**param)
                    print(f"📷 Cropped image shape: {img.shape}")
                    
                    # Get model path and predict
                    # Use just the filename - the _load_model method will resolve it relative to base_dir
                    model_prediction = self.anti_spoof_predictor.predict(img, model_name)
                    
                    # Accumulate predictions (same as original)
                    prediction += model_prediction
                    model_count += 1
                    
                    print(f"📊 Model {model_name} prediction: {model_prediction}")
                    
                except Exception as model_error:
                    print(f"⚠️ Error with model {model_name}: {model_error}")
                    continue
            
            if model_count == 0:
                print("❌ No models processed successfully, assuming real face")
                return True, 0.0
            
            # Get result using the same logic as test_anti_spoofing.py
            label = np.argmax(prediction)
            value = prediction[0][label] / sum(prediction[0])  # Normalize by sum
            
            # Calculate confidence scores for each class
            real_score = prediction[0][1] / sum(prediction[0]) if sum(prediction[0]) > 0 else 0
            fake_score = prediction[0][0] / sum(prediction[0]) if sum(prediction[0]) > 0 else 0
            spoof_score = prediction[0][2] / sum(prediction[0]) if sum(prediction[0]) > 0 else 0
            
            # Use threshold-based detection (like test_anti_spoofing.py)
            threshold = 0.3  # Lower threshold for better real face detection
            is_real = (label == 1 and value > threshold)
            confidence = float(value)
            
            print(f"🛡️ Face security module result:")
            print(f"   - Models used: {model_count}")
            print(f"   - Raw prediction: {prediction[0]}")
            print(f"   - Label: {label} (0=fake, 1=real, 2=spoof)")
            print(f"   - Value: {value:.4f}")
            print(f"   - Real score: {real_score:.4f}")
            print(f"   - Fake score: {fake_score:.4f}")
            print(f"   - Spoof score: {spoof_score:.4f}")
            print(f"   - Threshold: {threshold}")
            print(f"   - Is Real: {is_real}")
            
            if is_real:
                print(f"✅ Image is Real Face. Score: {value:.2f}")
            else:
                print(f"❌ Image is Fake Face. Score: {value:.2f}")
                print(f"   - Detected as: {'Fake' if label == 0 else 'Spoof' if label == 2 else 'Unknown'}")
                print(f"   - Reason: {'Label not 1' if label != 1 else f'Value {value:.3f} <= threshold {threshold}'}")
            
            return bool(is_real), confidence
            
        except Exception as e:
            print(f"❌ Error in face security detection: {e}")
            import traceback
            traceback.print_exc()
            return True, 0.0  # Assume real on error

    def _identify_face(self, face_encoding, is_masked_face=False):
        """Identify a face with improved matching algorithm
        
        Args:
            face_encoding: The face encoding to match
            is_masked_face: If True, prioritize masked encodings and use more lenient thresholds
        """
        if len(self.known_face_encodings) == 0:
            print("❌ No known face encodings available for matching")
            return "Unknown", "Unknown", 0.0
        
        try:
            # Validate face encoding
            if face_encoding is None or len(face_encoding) != 128:
                print(f"❌ Invalid face encoding: {face_encoding.shape if hasattr(face_encoding, 'shape') else 'None'}")
                return "Unknown", "Unknown", 0.0
            
            print(f"🔍 Matching face encoding (shape: {face_encoding.shape}) against {len(self.known_face_encodings)} known faces")
            if is_masked_face:
                print("   ⚠️ Detected as masked face - using lenient thresholds")
            
            # Calculate distances to all known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            print(f"📊 Calculated distances: {face_distances}")
            
            # If this is a masked face, prioritize masked encodings
            if is_masked_face:
                # First, try to find best match among masked encodings
                masked_indices = [i for i, meta in enumerate(self.known_face_metadata) 
                                if meta.get('variant', 'normal') != 'normal']
                
                if masked_indices:
                    masked_distances = [face_distances[i] for i in masked_indices]
                    best_masked_idx = masked_indices[int(np.argmin(masked_distances))]
                    best_masked_distance = float(face_distances[best_masked_idx])
                    
                    print(f"🎯 Best masked encoding match: index={best_masked_idx}, distance={best_masked_distance:.3f}")
                    
                    # Use very lenient threshold for masked faces - accept almost any reasonable match
                    masked_threshold = 1.2  # Even more lenient
                    masked_confidence = float(max(0, min(1, 1 - (best_masked_distance / masked_threshold))))
                    
                    # Accept if distance is reasonable (even if confidence is low)
                    if best_masked_distance <= masked_threshold:
                        student_id = self.known_face_ids[best_masked_idx]
                        student_name = self.known_face_names[best_masked_idx]
                        # Boost confidence if distance is actually good
                        if best_masked_distance <= 0.8:
                            masked_confidence = max(masked_confidence, 0.5)
                        print(f"✅ Recognized (masked): {student_name} (ID: {student_id}) with confidence {masked_confidence:.3f} (distance: {best_masked_distance:.3f})")
                        return student_id, student_name, masked_confidence

            # Find the best match overall
            best_match_index = int(np.argmin(face_distances)) if len(face_distances) > 0 else -1
            if best_match_index < 0:
                print("❌ No known faces available after distance calculation")
                return "Unknown", "Unknown", 0.0

            best_distance = float(face_distances[best_match_index])
            variant = self.known_face_metadata[best_match_index].get('variant', 'normal') if best_match_index < len(self.known_face_metadata) else 'normal'
            
            # Use appropriate thresholds based on face type and encoding variant
            if is_masked_face:
                # For masked faces, use lenient thresholds
                if variant != 'normal':
                    # Matching masked face to masked encoding - very lenient
                    variant_max_distance = 1.2
                    variant_tolerance = 1.2
                    min_confidence_threshold = 0.1
                else:
                    # Matching masked face to normal encoding - still lenient but less so
                    variant_max_distance = 0.8
                    variant_tolerance = 0.8
                    min_confidence_threshold = 0.2
            else:
                # For normal faces, use standard thresholds
                if variant != 'normal':
                    # Matching normal face to masked encoding - use normal thresholds
                    variant_max_distance = self.max_distance
                    variant_tolerance = self.recognition_tolerance
                    min_confidence_threshold = self.min_confidence
                else:
                    # Matching normal face to normal encoding - standard thresholds
                    variant_max_distance = self.max_distance
                    variant_tolerance = self.recognition_tolerance
                    min_confidence_threshold = self.min_confidence

            confidence_scale = variant_max_distance if variant_max_distance > 0 else self.max_distance
            confidence = float(max(0, min(1, 1 - (best_distance / confidence_scale))))

            print(f"🎯 Best match: index={best_match_index}, variant={variant}, distance={best_distance:.3f}, confidence={confidence:.3f}")
            print(f"📏 Variant thresholds: max_distance={variant_max_distance}, tolerance={variant_tolerance}, min_confidence={min_confidence_threshold}")

            is_within_distance = best_distance <= variant_max_distance
            match_id = self.known_face_ids[best_match_index] if is_within_distance else "Unknown"

            if is_within_distance and confidence >= min_confidence_threshold:
                student_id = match_id
                student_name = self.known_face_names[best_match_index]
                print(f"✅ Recognized: {student_name} (ID: {student_id}) with confidence {confidence:.3f} using {variant} encoding")
                return student_id, student_name, confidence
            else:
                print(f"❌ Face not recognized: distance={best_distance:.3f} > threshold={variant_max_distance} or confidence={confidence:.3f} < min={min_confidence_threshold}")
                return "Unknown", "Unknown", confidence
                
        except Exception as e:
            print(f"❌ Error in face identification: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown", "Unknown", 0.0

    def _register_known_encoding(self, encoding, student_id, student_name, variant='normal', metadata=None):
        """Internal helper to register a single face encoding with metadata"""
        try:
            np_encoding = np.asarray(encoding, dtype=np.float64)
            if np_encoding.shape != (128,):
                print(f"❌ Invalid face encoding shape for {student_name}: {np_encoding.shape}, expected (128,)")
                return False

            self.known_face_encodings.append(np_encoding)
            self.known_face_ids.append(student_id)
            self.known_face_names.append(student_name)

            record = {
                'student_id': student_id,
                'student_name': student_name,
                'variant': variant,
                'registered_at': None,
                'active': True
            }
            if metadata:
                record.update(metadata)

            self.known_face_metadata.append(record)
            print(f"✅ Registered {variant} encoding for {student_name} (ID: {student_id})")
            return True
        except Exception as register_error:
            print(f"❌ Error registering encoding for {student_name}: {register_error}")
            return False

    def add_face(self, face_encoding, student_id, student_name, variant='manual'):
        """Add a new face encoding to the in-memory recognition system"""
        metadata = {
            'added_at': datetime.now(),
            'data_source': 'runtime'
        }
        return self._register_known_encoding(face_encoding, student_id, student_name, variant=variant, metadata=metadata)

    def reload_faces_from_db(self):
        """Reload all faces from database"""
        print("🔄 Reloading faces from database...")
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.known_face_metadata = []
        self.load_known_faces_from_db()
        print(f"✅ Reloaded {len(self.known_face_encodings)} face encodings")
    
    def test_anti_spoofing_with_image(self, image_path):
        """Test anti-spoofing with a specific image file"""
        if not self.anti_spoof_predictor or not self.image_cropper:
            print("❌ Face security module not available for testing")
            return False, 0.0
        
        try:
            print(f"🧪 Testing anti-spoofing with image: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return False, 0.0
            
            print(f"📷 Loaded image: {image.shape}")
            
            # Get face bounding box using face security module
            bbox = self.anti_spoof_predictor.get_bbox(image)
            if bbox is None:
                print("❌ No face detected in image")
                return False, 0.0
            
            print(f"📍 Face bbox: {bbox}")
            
            # Test anti-spoofing using the same method as check_anti_spoofing
            is_real, confidence = self._test_anti_spoofing_with_bbox(image, bbox)
            
            print(f"🧪 Test result: {'REAL' if is_real else 'FAKE'} (confidence: {confidence:.3f})")
            return is_real, confidence
            
        except Exception as e:
            print(f"❌ Error testing anti-spoofing: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0

# =============================================================================
# FIXED ATTENDANCE SYSTEM
# =============================================================================

class FixedWebAttendanceSystem:
    """Fixed attendance system for web interface"""
    
    def __init__(self):
        self.face_recognition = FixedWebFaceRecognition()
        self.sync_interval_seconds = 10
        self._last_sync_signature = None
        self._sync_stop_event = threading.Event()
        self._auto_sync_thread = threading.Thread(target=self._auto_sync_loop, daemon=True)
        try:
            self._last_sync_signature = self._compute_student_signature()
        except Exception as signature_error:
            print(f"⚠️ Unable to compute initial student signature: {signature_error}")
        self._auto_sync_thread.start()
        print("Fixed web attendance system initialized")

    def recognize_face(self, image):
        """Recognize faces in image"""
        try:
            face_locations, student_ids, student_names, confidences = self.face_recognition.recognize_faces_improved(image)
            return face_locations, student_ids, student_names, confidences
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return [], [], [], []

    def _compute_student_signature(self):
        """Compute a hash signature of current student records for auto-sync"""
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        students = list(db.students.find({}, {
            'usn': 1,
            'registered_at': 1,
            'updated_at': 1,
            'active': 1,
            'face_encoding': 1,
            'face_encoding_masked': 1
        }))
        client.close()

        if not students:
            return ""

        digest = hashlib.sha256()
        for student in sorted(students, key=lambda d: d.get('usn', '')):
            payload = json.dumps({
                'usn': student.get('usn', ''),
                'registered_at': str(student.get('registered_at')),
                'updated_at': str(student.get('updated_at')),
                'active': student.get('active', True),
                'encoding_len': len(student.get('face_encoding') or []),
                'masked_len': len(student.get('face_encoding_masked') or [])
            }, sort_keys=True)
            digest.update(payload.encode('utf-8'))

        return digest.hexdigest()

    def _auto_sync_loop(self):
        """Background loop that reloads faces when database changes"""
        while not self._sync_stop_event.wait(self.sync_interval_seconds):
            try:
                signature = self._compute_student_signature()
                if signature != self._last_sync_signature:
                    print("🔄 Detected changes in student database. Reloading face encodings...")
                    self.face_recognition.reload_faces_from_db()
                    self._last_sync_signature = signature
            except Exception as sync_error:
                print(f"⚠️ Auto-sync error: {sync_error}")

# =============================================================================
# DATABASE MODELS (SIMPLIFIED)
# =============================================================================

class Student:
    """Student model for database operations"""
    
    @classmethod
    def count(cls):
        """Count total students"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            count = db.students.count_documents({})
            client.close()
            return count
        except:
            return 0

    @classmethod
    def get_department_counts(cls):
        """Get department counts"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            pipeline = [
                {"$group": {"_id": "$department", "count": {"$sum": 1}}}
            ]
            results = list(db.students.aggregate(pipeline))
            client.close()
            
            counts = {}
            for result in results:
                counts[result['_id']] = result['count']
            return counts
        except:
            return {}

class Attendance:
    """Attendance model for database operations"""
    
    @classmethod
    def get_today_count(cls):
        """Get today's attendance count"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            today = datetime.now().strftime('%Y-%m-%d')
            count = db.attendance.count_documents({'date': today})
            client.close()
            return count
        except:
            return 0

    @classmethod
    def from_recognition(cls, student_id, student_name, extra_data=None):
        """Create attendance record from recognition"""
        now = datetime.now()
        return {
            'student_id': student_id,
            'student_name': student_name,
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'timestamp': now,
            'day_of_week': now.strftime('%A'),
            'subject': extra_data.get('subject') if extra_data else 'General',
            'class_name': extra_data.get('class') if extra_data else 'General',
            'branch': extra_data.get('branch', 'Unknown'),
            'sem': extra_data.get('semester', 'Unknown'),
            'section': extra_data.get('section', 'Unknown'),
            'emotion': extra_data.get('emotion', 'Unknown') if extra_data else 'Unknown',
            'emotion_confidence': extra_data.get('emotion_confidence', 0.0) if extra_data else 0.0
        }

    def save(self):
        """Save attendance record"""
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            result = db.attendance.insert_one(self)
            client.close()
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving attendance: {e}")
            return None

# =============================================================================
# FLASK WEB APPLICATION
# =============================================================================

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'student_attendance_system_secret_key'

# Initialize compression
compress = Compress()
compress.init_app(app)

# Initialize fixed attendance system
attendance_system = FixedWebAttendanceSystem()

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the dashboard page with statistics"""
    try:
        student_count = Student.count()
        todays_attendance = Attendance.get_today_count()
        dept_counts = Student.get_department_counts()
        
        cse_students = dept_counts.get('CSE', 0)
        ece_students = dept_counts.get('ECE', 0)
        eee_students = dept_counts.get('EEE', 0)
        me_students = dept_counts.get('ME', 0)
        
        # Determine system status
        system_status = "Good"
        status_class = "success"
        
        # Check if face recognition is loaded
        if len(attendance_system.face_recognition.known_face_encodings) == 0:
            system_status = "Warning"
            status_class = "warning"
        
        return render_template('dashboard.html',
                              total_students=student_count,
                              todays_attendance=todays_attendance,
                              cse_students=cse_students,
                              ece_students=ece_students,
                              eee_students=eee_students,
                              me_students=me_students,
                              system_status=system_status,
                              status_class=status_class)
    except Exception as e:
        print(f"Error in dashboard: {e}")
        return render_template('dashboard.html',
                              total_students=0,
                              todays_attendance=0,
                              cse_students=0,
                              ece_students=0,
                              eee_students=0,
                              me_students=0,
                              system_status="Error",
                              status_class="danger")

@app.route('/attendance')
def attendance():
    """Render the attendance taking page"""
    return render_template('attendance.html')

@app.route('/get_attendance_stats')
def get_attendance_stats():
    """Get attendance statistics for the dashboard - FRESH DATA (no cache)"""
    try:
        # Force fresh database connection
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        
        # Get fresh counts
        total_students = db.students.count_documents({'active': True})
        today = datetime.now().strftime('%Y-%m-%d')
        todays_attendance = db.attendance.count_documents({'date': today})
        
        # Get department counts
        dept_counts = {}
        pipeline = [
            {'$match': {'active': True}},
            {'$group': {'_id': '$branch', 'count': {'$sum': 1}}}
        ]
        for dept in db.students.aggregate(pipeline):
            dept_counts[dept['_id']] = dept['count']
        
        client.close()
        
        return jsonify({
            'success': True,
            'total_students': total_students,
            'todays_attendance': todays_attendance,
            'department_counts': dept_counts,
            'system_status': 'Active',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error getting attendance stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_todays_attendance_list')
def get_todays_attendance_list():
    """Get today's attendance list with student names - FRESH DATA (no cache)"""
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get all attendance records for today, sorted by time (newest first)
        attendance_records = list(db.attendance.find({'date': today}).sort('time', -1))
        
        # Also check total records in collection for debugging
        total_records = db.attendance.count_documents({})
        
        # Format the data
        attendance_list = []
        for record in attendance_records:
            attendance_list.append({
                'student_id': record.get('student_id', 'Unknown'),
                'student_name': record.get('student_name', 'Unknown'),
                'time': record.get('time', 'Unknown'),
                'subject': record.get('subject', 'General'),
                'emotion': record.get('emotion', 'Unknown'),
                'timestamp': record.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if isinstance(record.get('timestamp'), datetime) else str(record.get('timestamp', ''))
            })
        
        client.close()
        
        return jsonify({
            'success': True,
            'date': today,
            'count': len(attendance_list),
            'attendance': attendance_list,
            'total_records_in_collection': total_records,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error getting today's attendance list: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test_database_connection')
def test_database_connection():
    """Test endpoint to verify database connection and check attendance records"""
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        
        # Test connection
        db.command('ping')
        
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get all attendance records
        all_records = list(db.attendance.find({}).sort('date', -1).limit(10))
        
        # Get today's records
        today_records = list(db.attendance.find({'date': today}))
        
        # Get collection stats
        total_count = db.attendance.count_documents({})
        today_count = db.attendance.count_documents({'date': today})
        
        client.close()
        
        return jsonify({
            'success': True,
            'database_connected': True,
            'today_date': today,
            'total_attendance_records': total_count,
            'today_attendance_count': today_count,
            'recent_records': [
                {
                    'student_id': r.get('student_id'),
                    'student_name': r.get('student_name'),
                    'date': r.get('date'),
                    'time': r.get('time')
                } for r in all_records
            ],
            'today_records': [
                {
                    'student_id': r.get('student_id'),
                    'student_name': r.get('student_name'),
                    'date': r.get('date'),
                    'time': r.get('time')
                } for r in today_records
            ]
        })
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'database_connected': False,
            'error': str(e)
        }), 500

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    """Process attendance from captured image - FIXED VERSION"""
    try:
        # Check if the request contains JSON data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data provided'}), 400
        
        # Get additional context data from the request if available
        subject = data.get('subject', 'General')
        class_name = data.get('class', 'General')
        session_id = data.get('session_id', None)
        
        # Log the attendance attempt
        logger.info(f"Processing attendance for subject: {subject}, class: {class_name}")
        
        # Get image data
        image_data = data['image']
        
        try:
            # Make sure the image data is properly formatted
            if ',' not in image_data:
                return jsonify({'success': False, 'message': 'Invalid image data format'}), 400
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            if not image_bytes:
                return jsonify({'success': False, 'message': 'Empty image data'}), 400
            
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'success': False, 'message': 'Failed to decode image'}), 400
            
            # Check if image is too small
            if image.shape[0] < 100 or image.shape[1] < 100:
                return jsonify({
                    'success': False, 
                    'message': f'Image too small: {image.shape[1]}x{image.shape[0]}, minimum 100x100 required'
                }), 400
            
            # Resize if image is too large (better performance)
            max_size = 1024
            if image.shape[0] > max_size or image.shape[1] > max_size:
                scale = max_size / max(image.shape[0], image.shape[1])
                image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
                
        except Exception as decode_error:
            logger.error(f"Error decoding image: {decode_error}")
            return jsonify({'success': False, 'error': f"Image decoding error: {str(decode_error)}"}), 400

        # Process with attendance system
        try:
            face_locations, student_ids, student_names, confidences = attendance_system.recognize_face(image)
            print(f"Recognition found {len(face_locations)} faces")
        except Exception as recog_error:
            logger.error(f"Face recognition error: {recog_error}")
            face_locations, student_ids, student_names, confidences = [], [], [], []

        # Track students we've already marked today to prevent duplicates
        today = datetime.now().strftime('%Y-%m-%d')
        processed_students = set()
        
        # Check today's attendance records for already marked students
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client.attendance_system
            today_records = list(db.attendance.find({'date': today}))
            for record in today_records:
                processed_students.add(record.get('student_id'))
            client.close()
        except Exception as db_error:
            print(f"Database error: {db_error}")
        
        faces = []
        
        for i, (student_id, student_name, confidence) in enumerate(zip(student_ids, student_names, confidences)):
            # Use the same threshold as the face recognition system
            threshold = 0.2
            
            # Check anti-spoofing for each detected face
            is_real = True
            anti_spoof_score = 0.0
            if i < len(face_locations):
                try:
                    is_real, anti_spoof_score = attendance_system.face_recognition.check_anti_spoofing(image, face_locations[i])
                except Exception as spoof_error:
                    print(f"Anti-spoofing error: {spoof_error}")
                    is_real = True  # Assume real on error
                    anti_spoof_score = 0.0

            # Emotion analysis using DeepFace (optional)
            emotion_label = None
            emotion_conf = 0.0
            
            # Try emotion detection if DeepFace is available
            if EMOTION_AVAILABLE and i < len(face_locations):
                try:
                    # face_locations format: (top, right, bottom, left)
                    (top, right, bottom, left) = face_locations[i]
                    
                    # Increase padding for better emotion detection
                    pad = 30  # Increased from 10 to 30
                    h, w = image.shape[:2]
                    t = max(0, top - pad)
                    l = max(0, left - pad)
                    b = min(h, bottom + pad)
                    r = min(w, right + pad)
                    
                    face_crop = image[t:b, l:r]
                    
                    if face_crop.size > 0:
                        # Ensure minimum face size for better emotion detection
                        min_face_size = 100
                        if face_crop.shape[0] < min_face_size or face_crop.shape[1] < min_face_size:
                            # Resize face crop to minimum size while maintaining aspect ratio
                            scale_factor = max(min_face_size / face_crop.shape[0], min_face_size / face_crop.shape[1])
                            new_h = int(face_crop.shape[0] * scale_factor)
                            new_w = int(face_crop.shape[1] * scale_factor)
                            face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Convert BGR to RGB for DeepFace
                        rgb_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        
                        print(f"🔍 Analyzing emotion for face crop: {rgb_face_crop.shape}")
                        
                        # Use multiple models for better emotion detection
                        try:
                            # Try with different backends for better accuracy
                            emo_result = DeepFace.analyze(
                                rgb_face_crop, 
                                actions=['emotion'], 
                                enforce_detection=False,
                                detector_backend='opencv',  # Use OpenCV detector
                                models={'emotion': 'fer2013'}  # Use FER2013 model
                            )
                        except Exception as model_error:
                            print(f"⚠️ Primary emotion model failed, trying fallback: {model_error}")
                            # Fallback to default settings
                            emo_result = DeepFace.analyze(
                                rgb_face_crop, 
                                actions=['emotion'], 
                                enforce_detection=False
                            )
                        
                        # DeepFace may return list or dict depending on version
                        emo = emo_result[0] if isinstance(emo_result, list) else emo_result
                        
                        # Get emotion scores
                        emotion_scores = emo.get('emotion', {})
                        print(f"📊 Raw emotion scores: {emotion_scores}")
                        
                        # Find dominant emotion with better logic
                        if emotion_scores:
                            # Filter out very low confidence emotions
                            filtered_emotions = {k: v for k, v in emotion_scores.items() if v > 5}
                            
                            if filtered_emotions:
                                dominant = max(filtered_emotions, key=filtered_emotions.get)
                                emotion_conf = float(filtered_emotions[dominant]) / 100.0
                                emotion_label = str(dominant)
                                
                                # If confidence is too low, try alternative approach
                                if emotion_conf < 0.3:
                                    # Get the emotion with highest score regardless of threshold
                                    dominant = max(emotion_scores, key=emotion_scores.get)
                                    emotion_conf = float(emotion_scores[dominant]) / 100.0
                                    emotion_label = str(dominant)
                            else:
                                # If all emotions are very low, use the highest one
                                dominant = max(emotion_scores, key=emotion_scores.get)
                                emotion_conf = float(emotion_scores[dominant]) / 100.0
                                emotion_label = str(dominant)
                        
                        print(f"😊 Emotion detected: {emotion_label} ({emotion_conf:.3f})")
                        
                except Exception as emo_err:
                    print(f"❌ Emotion detection error: {emo_err}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try simple emotion detection as fallback
                    try:
                        print("🔄 Trying simple emotion detection fallback...")
                        emotion_label, emotion_conf = detect_emotion_simple(face_crop)
                        print(f"😊 Fallback emotion detected: {emotion_label} ({emotion_conf:.3f})")
                    except Exception as fallback_err:
                        print(f"❌ Fallback emotion detection also failed: {fallback_err}")
                        emotion_label = "neutral"
                        emotion_conf = 0.3
            
            # If DeepFace is not available, try simple emotion detection
            elif not EMOTION_AVAILABLE and i < len(face_locations):
                try:
                    (top, right, bottom, left) = face_locations[i]
                    pad = 30
                    h, w = image.shape[:2]
                    t = max(0, top - pad)
                    l = max(0, left - pad)
                    b = min(h, bottom + pad)
                    r = min(w, right + pad)
                    face_crop = image[t:b, l:r]
                    
                    if face_crop.size > 0:
                        print("🔄 DeepFace not available, using simple emotion detection...")
                        emotion_label, emotion_conf = detect_emotion_simple(face_crop)
                        print(f"😊 Simple emotion detected: {emotion_label} ({emotion_conf:.3f})")
                except Exception as simple_err:
                    print(f"❌ Simple emotion detection failed: {simple_err}")
                    emotion_label = "neutral"
                    emotion_conf = 0.3
            
            # Debug: Print recognition details
            print(f"🎯 Recognition Check: student_id='{student_id}', confidence={confidence:.3f}, threshold={threshold}")
            print(f"🛡️ Anti-spoof Check: is_real={is_real}, anti_spoof_score={anti_spoof_score:.3f}")
            
            # Recognized with sufficient confidence
            if student_id != "Unknown" and confidence > threshold:
                # Check if student already marked attendance today
                already_marked = student_id in processed_students
                print(f"📅 Already marked today: {already_marked}")
                
                # Only mark attendance if face is real AND confidence >= 65%
                anti_spoof_threshold = 0.65
                accepted = bool(is_real and (anti_spoof_score is not None) and (anti_spoof_score >= anti_spoof_threshold))
                print(f"✅ Attendance accepted: {accepted} (is_real={is_real}, anti_spoof_score={anti_spoof_score:.3f}, threshold={anti_spoof_threshold})")
                
                face_result = {
                    'student_id': str(student_id),
                    'name': str(student_name),
                    'confidence': float(confidence),
                    'anti_spoof_score': float(anti_spoof_score),
                    'is_real': bool(is_real),
                    'accepted': bool(accepted),
                    'marked': False,
                    'already_marked': bool(already_marked),
                    'reason': (
                        f"Recognized with confidence {confidence:.2f}"
                        + ("" if is_real else " - Fake face detected")
                        + ("" if accepted else f" - Anti-spoof score {anti_spoof_score:.2f} < {anti_spoof_threshold:.2f}")
                    ),
                    'emotion': (str(emotion_label) if emotion_label else 'Unknown'),
                    'emotion_confidence': float(emotion_conf)
                }
                
                if not already_marked and accepted:
                    # Record new attendance only for real faces
                    try:
                        print(f"📝 Marking attendance for {student_name} (ID: {student_id})")
                        
                        # Get student details from database
                        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
                        db = client.attendance_system
                        
                        try:
                            # Get student document from database to retrieve email and other details
                        student_doc = db.students.find_one({'usn': student_id})
                            
                            if not student_doc:
                                print(f"⚠️ Student document not found for USN: {student_id}")
                                # Try to find by name as fallback
                                student_doc = db.students.find_one({'name': student_name})
                                if student_doc:
                                    print(f"✅ Found student by name: {student_name}")
                            
                            # Get student email early
                            student_email = None
                            if student_doc:
                                student_email = student_doc.get('email', '').strip()
                                if student_email:
                                    print(f"📧 Found email for {student_name}: {student_email}")
                                else:
                                    print(f"⚠️ No email found in database for {student_name} (USN: {student_id})")
                            else:
                                print(f"⚠️ Student document not found in database for {student_name} (USN: {student_id})")
                            
                            # Prepare extra data with student details and emotion
                        extra_data = {
                            'subject': subject,
                            'class': class_name,
                            'branch': student_doc.get('branch', 'Unknown') if student_doc else 'Unknown',
                            'semester': student_doc.get('semester', 'Unknown') if student_doc else 'Unknown',
                                'section': student_doc.get('section', 'Unknown') if student_doc else 'Unknown',
                                'emotion': str(emotion_label) if emotion_label else 'Unknown',
                                'emotion_confidence': float(emotion_conf)
                        }
                        
                        attendance_data = Attendance.from_recognition(student_id, student_name, extra_data)
                        
                            # Add additional fields to attendance record
                            attendance_data['is_real'] = bool(is_real)
                            attendance_data['anti_spoof_score'] = float(anti_spoof_score)
                            attendance_data['confidence'] = float(confidence)
                            
                            print(f"💾 Saving attendance to database: {attendance_data}")
                            print(f"📅 Date format: {attendance_data.get('date')} (type: {type(attendance_data.get('date'))})")
                            
                            # Ensure all required fields are present
                            required_fields = ['student_id', 'student_name', 'date', 'time', 'timestamp']
                            missing_fields = [field for field in required_fields if field not in attendance_data]
                            if missing_fields:
                                print(f"❌ ERROR: Missing required fields: {missing_fields}")
                                logger.error(f"Missing fields in attendance data: {missing_fields}")
                                face_result['marked'] = False
                                face_result['reason'] = f'Missing fields: {missing_fields}'
                            else:
                                # Insert attendance record with error handling
                                try:
                        result = db.attendance.insert_one(attendance_data)
                                    print(f"📝 Insert result: inserted_id={result.inserted_id}, acknowledged={result.acknowledged}")
                                    
                                    # Verify the insert was successful
                                    if result.inserted_id and result.acknowledged:
                                        # Double-check by querying the database immediately
                                        verify_record = db.attendance.find_one({'_id': result.inserted_id})
                                        if verify_record:
                            processed_students.add(student_id)
                            face_result['marked'] = True
                                            print(f"✅ Successfully marked attendance for {student_name} (Record ID: {result.inserted_id})")
                                            print(f"✅ Verified in database - Date: {verify_record.get('date')}, Student: {verify_record.get('student_name')}")
                                            logger.info(f"Attendance marked successfully for {student_name} (ID: {student_id}) at {attendance_data.get('time')} on {attendance_data.get('date')}")
                                            
                                            # Send email confirmation to student (email already retrieved above)
                                            if student_email:
                                                print(f"📧 Sending attendance confirmation email to {student_email} for {student_name} (USN: {student_id})")
                                                print(f"📧 From: {EMAIL_CONFIG['from_email']} ({EMAIL_CONFIG['from_name']})")
                                                email_sent = send_attendance_confirmation_email(
                                                    student_email,
                                                    student_name,
                                                    subject,
                                                    class_name,
                                                    datetime.now()
                                                )
                                                if email_sent:
                                                    print(f"✅ Email sent successfully to {student_email}")
                                                    logger.info(f"Attendance confirmation email sent to {student_email} for {student_name}")
                                                else:
                                                    print(f"❌ Failed to send email to {student_email} - check error logs above")
                                                    logger.error(f"Failed to send email to {student_email} for {student_name}")
                                            else:
                                                print(f"⚠️ Cannot send email: No email address found for student {student_name} (USN: {student_id})")
                                                print(f"   Please ensure the student is registered with an email address in the database")
                                                logger.warning(f"No email address for student {student_name} (USN: {student_id}) - email not sent")
                                        else:
                                            print(f"❌ ERROR: Attendance record not found after insert! Insert ID: {result.inserted_id}")
                                            logger.error(f"Failed to verify attendance record for {student_name} (ID: {student_id})")
                                            face_result['marked'] = False
                                            face_result['reason'] = 'Database verification failed'
                                    else:
                                        print(f"❌ ERROR: Failed to insert attendance record - no inserted_id or not acknowledged")
                                        logger.error(f"Failed to insert attendance record for {student_name} (ID: {student_id}) - acknowledged={result.acknowledged if result else 'None'}")
                                        face_result['marked'] = False
                                        face_result['reason'] = 'Database insert failed - not acknowledged'
                                except Exception as insert_error:
                                    print(f"❌ Database insert exception: {insert_error}")
                                    logger.error(f"Exception inserting attendance for {student_name} (ID: {student_id}): {insert_error}")
                                    import traceback
                                    traceback.print_exc()
                                    face_result['marked'] = False
                                    face_result['reason'] = f'Database insert exception: {str(insert_error)}'
                                
                        except Exception as db_insert_error:
                            print(f"❌ Database error while saving attendance: {db_insert_error}")
                            logger.error(f"Database error saving attendance for {student_name} (ID: {student_id}): {db_insert_error}")
                            import traceback
                            traceback.print_exc()
                            face_result['marked'] = False
                            face_result['reason'] = f'Database error: {str(db_insert_error)}'
                        finally:
                            client.close()
                    except Exception as save_error:
                        print(f"Error saving attendance: {save_error}")
                        face_result['reason'] = f"Database error: {str(save_error)}"
                elif already_marked:
                    face_result['reason'] = "Already marked today"
                    print(f"❌ Attendance NOT marked: Already marked today for {student_name}")
                elif not accepted:
                    if not is_real:
                        face_result['reason'] = "Fake face detected - attendance not marked"
                        print(f"❌ Attendance NOT marked: Fake face detected for {student_name}")
                    else:
                        face_result['reason'] = f"Anti-spoof score below threshold ({anti_spoof_score:.2f} < 0.65) - attendance not marked"
                        print(f"❌ Attendance NOT marked: Low real-face confidence {anti_spoof_score:.2f} for {student_name}")
                
                faces.append(face_result)
            else:
                # Unrecognized face - still check for anti-spoofing
                print(f"❌ Face NOT recognized: student_id='{student_id}', confidence={confidence:.3f} <= threshold={threshold}")
                is_real_unknown = True
                anti_spoof_score_unknown = 0.0
                if i < len(face_locations):
                    try:
                        is_real_unknown, anti_spoof_score_unknown = attendance_system.face_recognition.check_anti_spoofing(image, face_locations[i])
                    except Exception as spoof_error:
                        print(f"Anti-spoofing error for unknown face: {spoof_error}")
                        is_real_unknown = True
                        anti_spoof_score_unknown = 0.0
                
                faces.append({
                    'student_id': 'Unknown',
                    'name': 'Unknown Person',
                    'confidence': float(confidence),
                    'anti_spoof_score': float(anti_spoof_score_unknown),
                    'is_real': bool(is_real_unknown),
                    'marked': False,
                    'reason': ("Confidence too low" if confidence > 0.2 else "No match found") + ("" if is_real_unknown else " - Fake face detected"),
                    'emotion': (str(emotion_label) if emotion_label else 'Unknown'),
                    'emotion_confidence': float(emotion_conf)
                })

        return jsonify({
            'success': True,
            'faces': faces,
            'total_faces': int(len(face_locations))
        })

    except Exception as e:
        logger.error(f"Error in process_attendance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/registration', methods=['GET', 'POST'])
def registration():
    """Handle student registration"""
    if request.method == 'POST':
        try:
            logger.info('Received POST to /registration')
            # Get form data
            student_info = {
                'usn': request.form['usn'],
                'name': request.form['name'],
                'semester': request.form['semester'],
                'branch': request.form['branch'],
                'section': request.form['section'],
                'phone': request.form.get('phone'),
                'address': request.form.get('address'),
                'email': request.form.get('email', '').strip()
            }
            logger.info(f"Registration form data: usn={student_info.get('usn')}, name={student_info.get('name')}")

            # Process uploaded image (optional)
            photo_key = 'photo_0'
            if photo_key in request.form and request.form.get(photo_key):
                try:
                    # Decode base64 image
                    image_data = request.form.get(photo_key)
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    
                    # Convert to numpy array
                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Convert to RGB for face_recognition
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Find face locations - try standard method first, then MediaPipe
                        face_locations = face_recognition.face_locations(rgb_image)
                        
                        # If no faces detected, try MediaPipe (better for masked faces)
                        if not face_locations and MEDIAPIPE_AVAILABLE:
                            logger.info("No faces detected with standard method, trying MediaPipe...")
                            face_locations = detect_faces_with_mediapipe(rgb_image)
                            if face_locations:
                                logger.info(f"MediaPipe detected {len(face_locations)} face(s)")
                        
                        if face_locations:
                            # Get face encodings
                            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                            
                            if face_encodings:
                                # Use the first face encoding
                                face_encoding = face_encodings[0]
                                student_info['face_encoding'] = face_encoding.tolist()
                                logger.info("Face encoding extracted successfully")

                                masked_encoding = create_masked_face_encoding(rgb_image, face_locations[0])
                                if masked_encoding is None:
                                    logger.warning("Masked encoding generation failed; falling back to base encoding")
                                    masked_encoding = face_encoding
                                try:
                                    student_info['face_encoding_masked'] = np.asarray(masked_encoding, dtype=np.float64).tolist()
                                except Exception as mask_store_error:
                                    logger.error(f"Error converting masked encoding to list: {mask_store_error}")
                                    student_info['face_encoding_masked'] = student_info['face_encoding']
                                logger.info("Masked face encoding stored successfully")
                            else:
                                logger.warning("No face encodings found in image")
                        else:
                            logger.warning("No faces detected in image")
                    else:
                        logger.warning("Could not decode image")
                        
                except Exception as img_error:
                    logger.error(f"Error processing image: {img_error}")
            else:
                logger.info("No photo provided; registering student without face encoding")

            # Ensure masked encoding exists if baseline was captured
            if student_info.get('face_encoding') and not student_info.get('face_encoding_masked'):
                logger.warning("No masked encoding available; duplicating baseline encoding")
                student_info['face_encoding_masked'] = student_info['face_encoding']

            # Save to MongoDB
            try:
                client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
                db = client.attendance_system
                
                # Create student document
                student_doc = {
                    'usn': student_info['usn'],
                    'name': student_info['name'],
                    'semester': student_info['semester'],
                    'branch': student_info['branch'],
                    'section': student_info['section'],
                    'phone': student_info.get('phone'),
                    'address': student_info.get('address'),
                    'email': student_info.get('email', ''),
                    'registered_at': datetime.now(),
                    'face_encoding': student_info.get('face_encoding'),
                    'face_encoding_masked': student_info.get('face_encoding_masked'),
                    'active': True
                }
                
                # Insert into database
                result = db.students.insert_one(student_doc)
                client.close()
                
                if result.inserted_id:
                    logger.info(f'Student saved to DB with id: {result.inserted_id}')
                    
                    # Update face recognition system if face encoding exists
                    if 'face_encoding' in student_info:
                        attendance_system.face_recognition.add_face(
                            student_info['face_encoding'],
                            student_info['usn'],
                            student_info['name'],
                            variant='normal'
                        )
                        flash(f"{student_info['name']} registered successfully with face recognition enabled.", 'success')
                    else:
                        flash(f"{student_info['name']} registered successfully (no face photo provided).", 'success')

                    if 'face_encoding_masked' in student_info:
                        attendance_system.face_recognition.add_face(
                            student_info['face_encoding_masked'],
                            student_info['usn'],
                            student_info['name'],
                            variant='masked'
                        )

                    try:
                        attendance_system._last_sync_signature = attendance_system._compute_student_signature()
                    except Exception as sync_update_error:
                        logger.warning(f"Failed to update sync signature after registration: {sync_update_error}")
                    
                    return redirect(url_for('registration'))
                else:
                    flash("Failed to save student to database", 'danger')
                    
            except Exception as db_error:
                logger.error(f"Database error: {db_error}")
                flash(f"Database error: {str(db_error)}", 'danger')
                
        except Exception as e:
            logger.error(f"Error in registration: {e}")
            flash(f"Error during registration: {str(e)}", 'danger')

    return render_template('registration.html')

@app.route('/send_daily_summary', methods=['POST'])
def send_daily_summary_manual():
    """Manually trigger daily summary email (for testing)"""
    try:
        generate_daily_summary()
        return jsonify({'success': True, 'message': 'Daily summary email sent successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    """Reload faces from database"""
    try:
        attendance_system.face_recognition.reload_faces_from_db()
        try:
            attendance_system._last_sync_signature = attendance_system._compute_student_signature()
        except Exception as sync_error:
            logger.warning(f"Failed to refresh sync signature: {sync_error}")
        return jsonify({
            'success': True,
            'message': f'Successfully reloaded {len(attendance_system.face_recognition.known_face_encodings)} face encodings'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error reloading faces: {str(e)}'}), 500

@app.route('/analytics')
def analytics():
    """Serve the analytics page"""
    today_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('analytics.html', today_date=today_date)

@app.route('/get_attendance')
def get_attendance():
    """Get attendance records with filtering"""
    try:
        # Get filter parameters
        date_filter = request.args.get('date')
        branch_filter = request.args.get('branch')
        semester_filter = request.args.get('semester')
        section_filter = request.args.get('section')
        
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        
        # Build query
        query = {}
        if date_filter:
            query['date'] = date_filter
        if branch_filter:
            query['branch'] = branch_filter
        if semester_filter:
            query['sem'] = semester_filter
        if section_filter:
            query['section'] = section_filter
        
        # Get attendance records
        attendance_records = list(db.attendance.find(query).sort('date', -1).sort('time', -1))
        
        # Get all students to show absent ones
        students_query = {}
        if branch_filter:
            students_query['branch'] = branch_filter
        if semester_filter:
            students_query['semester'] = semester_filter  # Fixed: use 'semester' field from students collection
        if section_filter:
            students_query['section'] = section_filter
            
        all_students = list(db.students.find(students_query))
        
        # Create a comprehensive attendance list
        attendance_data = []
        
        # Add present students
        for record in attendance_records:
            attendance_data.append({
                'student_id': record.get('student_id', 'Unknown'),
                'student_name': record.get('student_name', 'Unknown'),
                'branch': record.get('branch', 'Unknown'),
                'sem': record.get('sem', 'Unknown'),
                'section': record.get('section', 'Unknown'),
                'date': record.get('date', 'Unknown'),
                'time': record.get('time', 'Unknown'),
                'status': 'Present'
            })
        
        # Add absent students (students not in attendance records for the date)
        if date_filter:
            present_student_ids = {record.get('student_id') for record in attendance_records}
            for student in all_students:
                if student.get('usn') not in present_student_ids:
                    attendance_data.append({
                        'student_id': student.get('usn', 'Unknown'),
                        'student_name': student.get('name', 'Unknown'),
                        'branch': student.get('branch', 'Unknown'),
                        'sem': student.get('semester', 'Unknown'),  # Fixed: use 'semester' from students collection
                        'section': student.get('section', 'Unknown'),
                        'date': date_filter,
                        'time': '',
                        'status': 'Absent'
                    })
        
        client.close()
        
        return jsonify(attendance_data)
        
    except Exception as e:
        logger.error(f"Error getting attendance: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def generate_daily_summary():
    """Generate and send daily attendance summary to teachers"""
    try:
        today = datetime.now().date()
        today_str = today.strftime('%Y-%m-%d')
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.attendance_system
        
        # Get all attendance records for today
        attendance_records = list(db.attendance.find({'date': today_str}))
        
        # Get all registered students
        all_students = list(db.students.find({'active': True}, {'usn': 1, 'name': 1, 'branch': 1, 'section': 1}))
        student_dict = {s['usn']: s for s in all_students}
        
        # Create attendance data with present/absent status
        attendance_data = []
        present_student_ids = {r.get('student_id') for r in attendance_records}
        
        # Add present students
        for record in attendance_records:
            student_id = record.get('student_id')
            student = student_dict.get(student_id, {})
            attendance_data.append({
                'usn': student_id,
                'name': student.get('name', 'Unknown'),
                'subject': record.get('subject', 'Unknown'),
                'timestamp': record.get('timestamp', 'N/A'),
                'status': 'present'
            })
        
        # Add absent students (all registered students not in present list)
        for student_id, student in student_dict.items():
            if student_id not in present_student_ids:
                # Get subject from today's records (use most common subject)
                subjects = [r.get('subject', 'Unknown') for r in attendance_records]
                most_common_subject = max(set(subjects), key=subjects.count) if subjects else 'Unknown'
                
                attendance_data.append({
                    'usn': student_id,
                    'name': student.get('name', 'Unknown'),
                    'subject': most_common_subject,
                    'timestamp': 'N/A',
                    'status': 'absent'
                })
        
        client.close()
        
        # Send email to teachers
        if EMAIL_CONFIG.get('teacher_emails'):
            send_daily_summary_email(EMAIL_CONFIG['teacher_emails'], today, attendance_data)
            logger.info(f"Daily summary generated and sent for {today_str}")
        else:
            logger.warning("No teacher emails configured for daily summary")
            
    except Exception as e:
        logger.error(f"Error generating daily summary: {e}")
        import traceback
        traceback.print_exc()

def schedule_daily_summary():
    """Schedule daily summary email to be sent at end of day (e.g., 6 PM)"""
    def run_scheduler():
        while True:
            try:
                now = datetime.now()
                # Schedule for 6 PM (18:00)
                target_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
                
                # If it's already past 6 PM today, schedule for tomorrow
                if now > target_time:
                    target_time += timedelta(days=1)
                
                # Calculate seconds until target time
                wait_seconds = (target_time - now).total_seconds()
                
                logger.info(f"Daily summary email scheduled for {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(wait_seconds)
                
                # Send daily summary
                generate_daily_summary()
                
                # Wait until next day
                time.sleep(3600)  # Wait 1 hour before scheduling next day
                
            except Exception as e:
                logger.error(f"Error in daily summary scheduler: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying
    
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Daily summary email scheduler started")

if __name__ == '__main__':
    print("Starting Fixed Integrated Student Attendance System...")
    print(f"Face recognition loaded: {len(attendance_system.face_recognition.known_face_encodings)} faces")
    
    # Start daily summary email scheduler
    schedule_daily_summary()
    
    # Add test page route
    @app.route('/test_anti_spoofing_page')
    def test_anti_spoofing_page():
        """Serve the anti-spoofing test page"""
        return render_template('test_anti_spoofing.html')
    
    # Add test route for anti-spoofing
    @app.route('/test_anti_spoofing', methods=['POST'])
    def test_anti_spoofing():
        """Test anti-spoofing with uploaded image"""
        try:
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No image selected'}), 400
            
            # Save uploaded image temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Test anti-spoofing
            is_real, confidence = attendance_system.face_recognition.test_anti_spoofing_with_image(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return jsonify({
                'success': True,
                'is_real': is_real,
                'confidence': confidence,
                'result': 'REAL FACE' if is_real else 'FAKE FACE'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Automatically open the web browser
    threading.Timer(1.0, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    
    # Start Flask app
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        import traceback
        traceback.print_exc()
