import os
import tempfile
import cv2
import numpy as np
import pytest
from unittest.mock import patch
from scripts.prepare_dataset import process_video, main
import sys

def create_synthetic_mp4(filepath, frames=10, size=(100, 100)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 30.0, size)
    for _ in range(frames):
        frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        out.write(frame)
    out.release()

def test_process_video():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "test.mp4")
        create_synthetic_mp4(input_video, frames=5)
        
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)
        
        extracted = process_video(input_video, output_dir, max_frames=2, img_size=128)
        assert extracted == 2
        assert len(os.listdir(output_dir)) == 2

def test_main_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        
        create_synthetic_mp4(os.path.join(input_dir, "test1.mp4"), frames=5)
        create_synthetic_mp4(os.path.join(input_dir, "test2.mp4"), frames=5)
        
        test_args = [
            "prepare_dataset.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--max_frames", "2",
            "--img_size", "128",
            "--num_workers", "2"
        ]
        
        with patch.object(sys, 'argv', test_args):
            main()
            
        assert len(os.listdir(output_dir)) == 4
