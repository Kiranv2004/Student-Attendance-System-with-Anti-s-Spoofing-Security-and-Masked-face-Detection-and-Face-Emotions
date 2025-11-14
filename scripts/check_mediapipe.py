"""Check if MediaPipe is installed"""
try:
    import mediapipe
    print(f"OK MediaPipe version: {mediapipe.__version__}")
except ImportError as e:
    print(f"ERROR MediaPipe not installed: {e}")
    print("Install with: pip install mediapipe==0.10.8")

