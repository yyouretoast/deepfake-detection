import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Append project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import predict_video_sequence

class MockVideoCapture:
    def __init__(self, *args, **kwargs):
        self.frame_count = 0
        self.max_frames = 20

    def isOpened(self):
        return self.frame_count < self.max_frames

    def get(self, propId):
        import cv2
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return self.max_frames
        return 0

    def read(self):
        if self.frame_count < self.max_frames:
            self.frame_count += 1
            # Return a valid RGB image simulating a frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            # Draw a simulated "face" so MTCNN/CenterCrop has something
            import cv2
            cv2.rectangle(frame, (200, 150), (400, 350), (255, 200, 200), -1)
            return True, frame
        return False, None
        
    def grab(self):
        if self.frame_count < self.max_frames:
            self.frame_count += 1
            return True
        return False

    def release(self):
        pass

@patch("app.cv2.VideoCapture", MockVideoCapture)
def test_predict_video_sequence_e2e():
    """
    E2E integration test: Mocks VideoCapture to supply solid-color frames, 
    passes them through MTCNN (or fallback crop), preprocesses them, 
    and validates model batched forward inference via predict_video_sequence.
    """
    # Use a dummy path
    res = predict_video_sequence("dummy_video.mp4", enable_gradcam=False)
    
    assert res is not None, "predict_video_sequence returned None"
    assert "final_label" in res
    assert "final_conf" in res
    assert "sample_outputs" in res
    
    # We should have evaluated something
    assert res["real_frames"] + res["fake_frames"] > 0
