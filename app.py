#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of OCR Butterfly.
# Based on MLX-Video-OCR-DeepSeek-Apple-Silicon by matica0902 (AGPL-3.0).
# Copyright (C) 2025 MLX DeepSeek-OCR contributors
# Copyright (C) 2026 Ricardo Kupper
# See the LICENSE file in the project root for full license text.

import os
import gc
import re
import traceback
import atexit
import tempfile
import uuid
import fitz
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import threading
import io
import base64
import multiprocessing
import queue
import cv2
import numpy as np
import zipfile
import shutil

import mlx.core as mx
from mlx_vlm import load, generate

os.environ["HF_HOME"] = str(Path.home() / "hf_cache")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB limit for video uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

MODEL_NAME = "mlx-community/DeepSeek-OCR-2-8bit"
MODEL_DISPLAY_NAME = "DeepSeek-OCR-2 (8-bit)"
model_loaded_status = multiprocessing.Value('b', False)
pdf_tasks = {}
preprocess_tasks = {}
video_tasks = {}

UPLOAD_FOLDER = tempfile.gettempdir()
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# ==============================================================================
# 9 Categories × 5 Complexity Preprocessing Configuration
# ==============================================================================

PREPROCESSING_CONFIG = {
    'Document': {
        'Academic': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Business': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Content': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Table': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Handwritten': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Complex': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        }
    },
    'Scene': {
        'Street': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Photo': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Objects': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 512},
            'Small': {'image_size': (640, 640), 'max_tokens': 1024},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 2048},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 4096},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 8192}
        },
        'Verification': {
            'Tiny': {'image_size': (512, 512), 'max_tokens': 256},
            'Small': {'image_size': (640, 640), 'max_tokens': 512},
            'Medium': {'image_size': (1024, 1024), 'max_tokens': 1024},
            'Large': {'image_size': (1280, 1280), 'max_tokens': 2048},
            'Gundam': {'image_size': (1024, 1024), 'max_tokens': 4096}
        }
    }
}

# Old 5 modes to new categories quick mapping
MODE_QUICK_MAP = {
    'basic': {'content_type': 'Document', 'subcategory': 'Academic', 'complexity': 'Medium'},
    'table': {'content_type': 'Document', 'subcategory': 'Table', 'complexity': 'Large'},
    'markdown': {'content_type': 'Document', 'subcategory': 'Content', 'complexity': 'Medium'},
    'formula': {'content_type': 'Document', 'subcategory': 'Academic', 'complexity': 'Large'},
    'product': {'content_type': 'Scene', 'subcategory': 'Photo', 'complexity': 'Small'}
}

prompts = {
    'product': '<image>\nExtract all printed text from this product packaging exactly as it appears. Preserve the original language.',
    'basic': '<image>\nExtract all text from the image. Keep the original language and formatting.',
    'table': '<image>\nConvert this table to clean Markdown format.',
    'markdown': '<image>\nConvert this document page to clean Markdown format.',
    'formula': '<image>\nExtract all mathematical formulas using LaTeX format.'
}

# ==============================================================================
# Image Preprocessing Functions
# ==============================================================================

def remove_background_pil(image):
    """Remove background using rembg"""
    try:
        from rembg import remove
        result = remove(image)
        return result
    except ImportError:
        print("⚠️ rembg not available, using simple background removal")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = img_array
        rgba[:, :, 3] = mask
        
        return Image.fromarray(rgba, 'RGBA')
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        return image

def auto_rotate_image(image):
    """Auto-rotate image"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = []
            for rho, theta in lines[:20]:
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 1:
                    (h, w) = img_array.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_REPLICATE)
                    return Image.fromarray(rotated)
    except Exception as e:
        print(f"⚠️ Auto-rotate failed: {e}")
    
    return image

def enhance_image(image):
    """Image enhancement processing"""
    img_array = np.array(image)

    # Contrast enhancement
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return Image.fromarray(sharpened)

def remove_shadows(image):
    """Remove shadows"""
    img_array = np.array(image)
    rgb_planes = cv2.split(img_array)
    
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    
    result = cv2.merge(result_planes)
    return Image.fromarray(result)

def binarize_image(image):
    """Binarization processing"""
    img_array = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary, 'L')

PREPROCESS_PRESETS = {
    'scan_optimize': {
        'auto_rotate': True,
        'enhance': True,
        'remove_shadows': True,
        'binarize': True,
        'remove_bg': False
    },
    'photo_optimize': {
        'auto_rotate': True,
        'enhance': True,
        'remove_shadows': True,
        'binarize': False,
        'remove_bg': False
    },
    'enhance_blurry': {
        'auto_rotate': False,
        'enhance': True,
        'remove_shadows': False,
        'binarize': False,
        'remove_bg': False
    },
    'remove_background_only': {
        'auto_rotate': False,
        'enhance': False,
        'remove_shadows': False,
        'binarize': False,
        'remove_bg': True
    }
}

def preprocess_single_image(image, settings):
    """Preprocess single image"""
    processed = image.copy()
    intermediate_images = []

    try:
        if settings.get('auto_rotate', False):
            new_img = auto_rotate_image(processed)
            if new_img is not processed:
                intermediate_images.append(processed)
                processed = new_img

        if settings.get('enhance', False):
            new_img = enhance_image(processed)
            if new_img is not processed:
                intermediate_images.append(processed)
                processed = new_img

        if settings.get('remove_shadows', False):
            new_img = remove_shadows(processed)
            if new_img is not processed:
                intermediate_images.append(processed)
                processed = new_img

        if settings.get('binarize', False):
            new_img = binarize_image(processed)
            if new_img is not processed:
                intermediate_images.append(processed)
                processed = new_img

        if settings.get('remove_bg', False):
            new_img = remove_background_pil(processed)
            if new_img is not processed:
                intermediate_images.append(processed)
                processed = new_img

        # Successfully processed, cleanup intermediate images
        for img in intermediate_images:
            if img is not None and img is not processed:
                try:
                    img.close()
                except:
                    pass

        return processed

    except Exception as e:
        # On error, cleanup all intermediate images and processing results
        print(f"⚠️ Image preprocessing failed: {e}")
        for img in intermediate_images:
            if img is not None:
                try:
                    img.close()
                except:
                    pass
        if processed is not None and processed is not image:
            try:
                processed.close()
            except:
                pass
        raise

# ==============================================================================
# Video Frame Extraction Functions
# ==============================================================================

def extract_frames_from_video(video_path, output_dir, method='fixed_count', interval=5, total_frames=1000, sensitivity=0.5):
    """Extract frames from video"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / fps if fps > 0 else 0

    print(f"📹 Video info: {total_video_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")

    frames = []

    if method == 'fixed_interval':
        # Fixed interval (N frames per second)
        frame_interval = max(1, int(fps / interval))
        if frame_interval == 0:
            frame_interval = 1
        
        for frame_idx in range(0, total_video_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(output_dir, f"frame_{len(frames):06d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames.append(frame_path)
            
            if len(frames) >= total_frames:
                break
    
    elif method == 'fixed_count':
        # Fixed count
        frame_interval = max(1, total_video_frames // total_frames)
        
        for frame_idx in range(0, total_video_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(output_dir, f"frame_{len(frames):06d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames.append(frame_path)
            
            if len(frames) >= total_frames:
                break
    
    elif method == 'scene_change':
        # Scene change detection (simplified)
        prev_frame = None
        # Convert sensitivity (0.1-1.0) to threshold: higher sensitivity = lower threshold = detect more changes
        # sensitivity=0.1(low) → threshold≈0.73, sensitivity=1.0(high) → threshold≈0.40
        # Threshold range: 0.4~0.75 (significantly increased range for better discrimination, max 5 frames per second)
        scene_change_threshold = (1.0 - sensitivity) * 0.35 + 0.4
        print(f"🎬 Scene change detection: sensitivity={sensitivity:.2f}, threshold={scene_change_threshold:.3f}")
        
        for frame_idx in range(total_video_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, frame)
                non_zero_count = np.count_nonzero(diff)
                total_pixels = diff.shape[0] * diff.shape[1] * diff.shape[2]
                change_ratio = non_zero_count / total_pixels
                
                if change_ratio > scene_change_threshold:
                    frame_path = os.path.join(output_dir, f"frame_{len(frames):06d}.jpg")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frames.append(frame_path)
            
            prev_frame = frame.copy()
            
            if len(frames) >= total_frames:
                break
    
    cap.release()
    print(f"✅ Extraction complete: {len(frames)} frames")
    return frames

# ==============================================================================
# Model Loading and OCR Related Functions
# ==============================================================================

_model_instance = None
_processor_instance = None
_model_instance_lock = threading.Lock()

def _patch_transformers_llama_compat():
    """
    Some recent transformers builds removed LlamaFlashAttention2.
    DeepSeek remote modeling code still imports it, so provide a safe alias.
    """
    try:
        from transformers.models.llama import modeling_llama
        if hasattr(modeling_llama, "LlamaFlashAttention2"):
            return
        if hasattr(modeling_llama, "LlamaAttention"):
            modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
            print(f"[{os.getpid()}] ⚠️ Added compatibility alias: LlamaFlashAttention2 -> LlamaAttention")
    except Exception as e:
        print(f"[{os.getpid()}] ⚠️ Failed to apply transformers compatibility patch: {e}")

def _load_model_for_subprocess():
    global _model_instance, _processor_instance, _model_instance_lock
    with _model_instance_lock:
        if _model_instance is not None and _processor_instance is not None:
            return True
        try:
            _patch_transformers_llama_compat()
            print(f"[{os.getpid()}] 🚀 Loading OCR model in subprocess...")
            model_path = MODEL_NAME
            _model_instance, _processor_instance = load(model_path, trust_remote_code=True)
            print(f"[{os.getpid()}] ✅ Model loaded successfully in subprocess!")
            print(f"[{os.getpid()}] 🔊 Metal available: {mx.metal.is_available()}")
            return True
        except Exception as e:
            print(f"[{os.getpid()}] ❌ Error loading model in subprocess: {e}")
            traceback.print_exc()
            _model_instance = None
            _processor_instance = None
            return False

def _run_ocr_in_process(image_bytes, prompt, max_tokens, output_queue):
    if not _load_model_for_subprocess():
        output_queue.put({'error': 'Model load failed in subprocess'})
        return
    try:
        t_load_start = time.time()
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        t_load_end = time.time()
        print(f"[{os.getpid()}] 📸 Image loaded: {img.size}, mode: {img.mode}, time: {t_load_end - t_load_start:.2f}s")

        print(f"[{os.getpid()}] 🔍 Starting OCR in subprocess...")
        print(f"[{os.getpid()}] 🔍 Image object type: {type(img)}")
        print(f"[{os.getpid()}] 🔍 Image size: {img.size}, mode: {img.mode}")

        # Ensure device configuration is correct
        # Ensure device configuration is correct
        if mx.metal.is_available():
            # mx.set_default_device(mx.gpu) # Default is usually GPU on Apple Silicon
            print(f"[{os.getpid()}] 🔧 Metal available, using default device (GPU)")
        else:
            mx.set_default_device(mx.cpu)
            print(f"[{os.getpid()}] 🔧 Set default device to CPU")
        
        print(f"[{os.getpid()}] 🔍 Prompt: {prompt[:100]}...")
        print(f"[{os.getpid()}] 🔍 Max tokens: {max_tokens}")

        # Ensure model and processor are correctly initialized
        if _model_instance is None or _processor_instance is None:
            raise RuntimeError("Model or processor is None - model not properly loaded")
        
        t_ocr_start = time.time()

        # Fix mlx-vlm 0.3.5 bug: stream_generate() may return empty generator, causing last_response to be None
        # Solution: Use smaller max_tokens value and add retry mechanism

        max_retries = 3
        retry_tokens = [min(max_tokens, 2048), min(max_tokens, 512), 256]
        res = None

        for attempt in range(max_retries):
            try:
                current_max_tokens = retry_tokens[attempt]
                print(f"[{os.getpid()}] 🔍 Attempt {attempt + 1}/{max_retries}: max_tokens={current_max_tokens}")

                res = generate(
                    model=_model_instance,
                    processor=_processor_instance,
                    image=img,
                    prompt=prompt,
                    max_tokens=current_max_tokens,
                    temperature=0.0,
                    use_cache=False
                )

                # If successful, break out of loop
                if res is not None:
                    break

            except AttributeError as e:
                if "'NoneType' object has no attribute 'token'" in str(e):
                    print(f"[{os.getpid()}] ⚠️ Attempt {attempt + 1} failed: mlx-vlm bug (last_response=None)")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry (time already imported at file beginning)
                        continue
                    else:
                        raise RuntimeError("mlx-vlm generate() failed: stream_generate() produced no response. This may be a known issue in CPU mode.")
                else:
                    raise
            except Exception as e:
                print(f"[{os.getpid()}] ⚠️ Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry (time already imported at file beginning)
                    continue
                else:
                    raise

        t_ocr_end = time.time()

        # Check if result is valid
        if res is None:
            raise RuntimeError("generate() still returned None after all retries")
        
        text = res.text if hasattr(res, 'text') else str(res)
        text = re.sub(r'<\|grounding\|>|\[\[.*?\]\]', '', text).strip()
        
        print(f"[{os.getpid()}] ✅ OCR completed in {t_ocr_end - t_ocr_start:.2f}s, text length: {len(text)}")
        output_queue.put({'success': True, 'text': text, 'timing': {
            'load': t_load_end - t_load_start,
            'inference': t_ocr_end - t_ocr_start
        }})
    except Exception as e:
        traceback.print_exc()
        output_queue.put({'error': f'OCR processing failed in subprocess: {str(e)}'})
    finally:
        if 'img' in locals() and img is not None:
            img.close()
        gc.collect()

def generate_with_timeout_and_process(image, prompt, max_tokens=8192, timeout=160):
    t_serialize_start = time.time()
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()
    t_serialize_end = time.time()
    
    print(f"📦 Image serialized: {len(image_bytes) / 1024:.1f}KB (JPEG), time: {t_serialize_end - t_serialize_start:.2f}s")
    
    output_queue = multiprocessing.Queue()
    t_process_start = time.time()
    
    process = multiprocessing.Process(
        target=_run_ocr_in_process,
        args=(image_bytes, prompt, max_tokens, output_queue)
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        print(f"[{os.getpid()}] ⏰ OCR processing timed out. Terminating subprocess {process.pid}.")
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        raise TimeoutError("OCR processing timeout")

    if process.exitcode != 0:
        print(f"[{os.getpid()}] ❌ OCR subprocess exited with code {process.exitcode}")
        try:
            result = output_queue.get_nowait()
            if 'error' in result:
                raise RuntimeError(result['error'])
        except queue.Empty:
            pass
        raise RuntimeError(f"OCR subprocess exited unexpectedly (code {process.exitcode})")

    try:
        result = output_queue.get(timeout=1)
        t_process_end = time.time()
        
        if 'success' in result:
            timing = result.get('timing', {})
            print(f"⏱️ Total time: {t_process_end - t_process_start:.2f}s (serialization: {t_serialize_end - t_serialize_start:.2f}s, inference: {timing.get('inference', 0):.2f}s)")
            return result['text']
        else:
            raise RuntimeError(result.get('error', 'Unknown OCR error in subprocess'))
    except queue.Empty:
        raise RuntimeError("OCR subprocess finished but no result in queue.")

# ==============================================================================
# Preprocessing Functions
# ==============================================================================

def get_preprocessing_config(content_type, subcategory, complexity):
    """Get preprocessing configuration based on 3D classification"""
    try:
        config = PREPROCESSING_CONFIG[content_type][subcategory][complexity]
        return config
    except KeyError:
        return None

def preprocess_image_by_config(image, image_size):
    """Resize image to specified size"""
    img_processed = image.copy()
    img_processed.thumbnail(image_size, Image.Resampling.LANCZOS)
    print(f"🖼️ Image preprocessed to {image_size}, actual size: {img_processed.size}")
    return img_processed

# ==============================================================================
# Routes and Main Application Logic
# ==============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def is_model_healthy():
    return model_loaded_status.value

def preload_model_main_process():
    print("🔧 Setting model preloaded status for main process...")
    try:
        if not mx.metal.is_available():
            print("WARNING: Metal is not available, MLX might not perform well.")
        
        model_loaded_status.value = True
        print("✅ Model preloaded status set successfully for main process.")
        return True
    except Exception as e:
        print(f"❌ Failed to set model preloaded status: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': model_loaded_status.value,
        'model_healthy': is_model_healthy(),
        'model_name': MODEL_NAME,
        'model_display_name': MODEL_DISPLAY_NAME
    })

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded_status.value,
        'model_healthy': is_model_healthy(),
        'model_name': MODEL_NAME,
        'model_display_name': MODEL_DISPLAY_NAME,
        'active_tasks': len(pdf_tasks),
        'preprocess_tasks': len(preprocess_tasks),
        'video_tasks': len(video_tasks)
    })

# ==============================================================================
# Image Preprocessing API
# ==============================================================================

@app.route('/api/preprocess/upload', methods=['POST'])
def preprocess_upload():
    """Upload photos for preprocessing"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    task_id = str(uuid.uuid4())
    task_dir = Path(UPLOAD_FOLDER) / f"preprocess_{task_id}"
    raw_dir = task_dir / "raw"
    processed_dir = task_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    
    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            raw_path = raw_dir / filename
            file.save(raw_path)
            
            img = None
            try:
                # Generate thumbnail preview
                img = Image.open(raw_path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                thumb_b64 = base64.b64encode(buffered.getvalue()).decode()

                image_files.append({
                    'filename': filename,
                    'raw_path': str(raw_path),
                    'thumb_b64': f"data:image/png;base64,{thumb_b64}",
                    'processed_path': None,
                    'status': 'pending'
                })
            except Exception as e:
                print(f"⚠️ Error generating thumbnail for {filename}: {e}")
                # Still add to list even if thumbnail fails
                image_files.append({
                    'filename': filename,
                    'raw_path': str(raw_path),
                    'thumb_b64': None,
                    'processed_path': None,
                    'status': 'pending'
                })
            finally:
                if img:
                    img.close()
    
    if not image_files:
        return jsonify({'error': 'No valid image files'}), 400
    
    preprocess_tasks[task_id] = {
        'task_dir': str(task_dir),
        'images': image_files,
        'settings': {},
        'created_at': datetime.now()
    }
    
    return jsonify({
        'success': True,
        'task_id': task_id,
        'images': image_files,
        'total_images': len(image_files)
    })

@app.route('/api/preprocess/process', methods=['POST'])
def preprocess_process():
    """Execute photo preprocessing"""
    data = request.get_json()
    task_id = data.get('task_id')
    settings = data.get('settings', {})
    
    if task_id not in preprocess_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = preprocess_tasks[task_id]
    task['settings'] = settings
    
    processed_dir = Path(task['task_dir']) / "processed"
    
    results = []
    
    for img_info in task['images']:
        original_img = None
        processed_img = None
        thumb_img = None
        
        try:
            raw_path = img_info['raw_path']
            filename = img_info['filename']

            # Process image
            original_img = Image.open(raw_path)
            processed_img = preprocess_single_image(original_img, settings)

            # Save processed image
            output_path = processed_dir / filename
            if processed_img.mode == 'RGBA':
                output_path = output_path.with_suffix('.png')
                processed_img.save(output_path, 'PNG')
            else:
                processed_img.save(output_path, 'JPEG', quality=95)

            # Generate processed thumbnail
            thumb_img = processed_img.copy()
            thumb_img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            thumb_img.save(buffered, format="PNG")
            processed_thumb_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            img_info.update({
                'processed_path': str(output_path),
                'processed_thumb_b64': f"data:image/png;base64,{processed_thumb_b64}",
                'status': 'completed'
            })

            results.append({
                'filename': filename,
                'status': 'completed',
                'processed_path': str(output_path),
                'processed_thumb_b64': f"data:image/png;base64,{processed_thumb_b64}"
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_info['filename']}: {e}")
            traceback.print_exc()
            img_info['status'] = 'failed'
            results.append({
                'filename': img_info['filename'],
                'status': 'failed',
                'error': str(e)
            })
        
        finally:
            # Ensure all images are closed
            if original_img:
                try:
                    original_img.close()
                except:
                    pass
            if processed_img and processed_img is not original_img:
                try:
                    processed_img.close()
                except:
                    pass
            if thumb_img:
                try:
                    thumb_img.close()
                except:
                    pass
    
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/api/preprocess/download', methods=['POST'])
def preprocess_download():
    """Download processed photos"""
    data = request.get_json()
    task_id = data.get('task_id')

    if task_id not in preprocess_tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = preprocess_tasks[task_id]
    processed_dir = Path(task['task_dir']) / "processed"

    # Create ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_info in task['images']:
            if img_info['status'] == 'completed' and img_info['processed_path']:
                file_path = Path(img_info['processed_path'])
                if file_path.exists():
                    zip_file.write(file_path, file_path.name)
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"processed_images_{task_id}.zip",
        mimetype='application/zip'
    )

@app.route('/api/preprocess/to-ocr', methods=['POST'])
def preprocess_to_ocr():
    """Send processed photos to OCR"""
    data = request.get_json()
    task_id = data.get('task_id')
    
    if task_id not in preprocess_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = preprocess_tasks[task_id]

    # Return processed images info, frontend can send these images to OCR
    processed_images = []
    for img_info in task['images']:
        if img_info['status'] == 'completed' and img_info['processed_path']:
            processed_images.append({
                'filename': img_info['filename'],
                'processed_path': img_info['processed_path'],
                'thumb_b64': img_info['processed_thumb_b64']
            })
    
    return jsonify({
        'success': True,
        'processed_images': processed_images
    })

# ==============================================================================
# Video Frame Extraction API
# ==============================================================================

@app.route('/api/video/upload', methods=['POST'])
def video_upload():
    """Upload video for frame extraction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_video_file(file.filename):
        return jsonify({'error': 'Invalid video format'}), 400

    task_id = str(uuid.uuid4())
    task_dir = Path(UPLOAD_FOLDER) / f"video_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save video
    video_filename = secure_filename(file.filename)
    video_path = task_dir / video_filename
    file.save(video_path)

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video file'}), 400
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()

    video_tasks[task_id] = {
        'task_dir': str(task_dir),
        'video_path': str(video_path),
        'video_info': {
            'filename': video_filename,
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'resolution': f"{width}x{height}"
        },
        'frames': [],
        'settings': {},
        'created_at': datetime.now()
    }

    return jsonify({
        'success': True,
        'task_id': task_id,
        'video_info': video_tasks[task_id]['video_info']
    })

@app.route('/api/video/extract', methods=['POST'])
def video_extract():
    """Extract frames from video"""
    data = request.get_json()
    task_id = data.get('task_id')
    settings = data.get('settings', {})
    
    if task_id not in video_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = video_tasks[task_id]
    task['settings'] = settings
    
    frames_dir = Path(task['task_dir']) / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    method = settings.get('method', 'fixed_count')
    interval = settings.get('interval', 5)
    total_frames = settings.get('total_frames', 1000)
    sensitivity = settings.get('sensitivity', 0.5)
    output_format = settings.get('format', 'jpg')
    
    try:
        frames = extract_frames_from_video(
            task['video_path'],
            str(frames_dir),
            method=method,
            interval=interval,
            total_frames=total_frames,
            sensitivity=sensitivity
        )

        # Generate frame thumbnail previews
        frame_previews = []
        for i, frame_path in enumerate(frames):
            img = None
            try:
                img = Image.open(frame_path)
                img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                thumb_b64 = base64.b64encode(buffered.getvalue()).decode()
                
                frame_previews.append({
                    'index': i + 1,
                    'path': frame_path,
                    'thumb_b64': f"data:image/png;base64,{thumb_b64}",
                    'selected': True
                })
            except Exception as e:
                print(f"⚠️ Error generating thumbnail for frame {i}: {e}")
                frame_previews.append({
                    'index': i + 1,
                    'path': frame_path,
                    'thumb_b64': None,
                    'selected': True
                })
            finally:
                if img:
                    try:
                        img.close()
                    except:
                        pass

        task['frames'] = frame_previews

        return jsonify({
            'success': True,
            'total_frames': len(frames),
            'frames': frame_previews
        })

    except Exception as e:
        print(f"❌ Video extraction failed: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Frame extraction failed: {str(e)}'}), 500

@app.route('/api/video/download', methods=['POST'])
def video_download():
    """Download video frames"""
    data = request.get_json()
    task_id = data.get('task_id')
    selected_frames = data.get('selected_frames', [])
    
    if task_id not in video_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = video_tasks[task_id]

    # Create ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for frame_info in task['frames']:
            if frame_info['index'] in selected_frames or not selected_frames:
                frame_path = Path(frame_info['path'])
                if frame_path.exists():
                    zip_file.write(frame_path, frame_path.name)
    
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"video_frames_{task_id}.zip",
        mimetype='application/zip'
    )

# ==============================================================================
# OCR Routes (unchanged)
# ==============================================================================

@app.route('/api/ocr', methods=['POST'])
def ocr():
    """Single image OCR - supports new 3D classification and old quick mode"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format not supported'}), 400

    # Determine whether to use new classification or old mode
    old_mode = request.form.get('mode')
    if old_mode and old_mode in MODE_QUICK_MAP:
        # Quick map old mode
        mapping = MODE_QUICK_MAP[old_mode]
        content_type = mapping['content_type']
        subcategory = mapping['subcategory']
        complexity = mapping['complexity']
    else:
        # Use new classification
        content_type = request.form.get('content_type', 'Document')
        subcategory = request.form.get('subcategory', 'Academic')
        complexity = request.form.get('complexity', 'Medium')

    # Get preprocessing configuration
    config = get_preprocessing_config(content_type, subcategory, complexity)
    if not config:
        return jsonify({'error': f'Invalid configuration: {content_type}/{subcategory}/{complexity}'}), 400
    
    if not is_model_healthy():
        return jsonify({'error': 'Model not ready. Please check server status.'}), 500
    
    tmp_path = None
    img = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        if tmp_path.lower().endswith('.pdf'):
            return jsonify({'error': 'Use PDF batch API for PDF files'}), 400

        img = Image.open(tmp_path).convert('RGB')

        # Preprocess based on configuration
        img_processed = preprocess_image_by_config(img, config['image_size'])

        # Use old mode prompt (if provided)
        if old_mode and old_mode in prompts:
            prompt = prompts[old_mode]
        else:
            prompt = prompts['basic']

        print(f"📋 Processing with: content_type={content_type}, subcategory={subcategory}, complexity={complexity}")
        print(f"   Image size: {config['image_size']}, max_tokens: {config['max_tokens']}")

        text = generate_with_timeout_and_process(
            image=img_processed,
            prompt=prompt,
            max_tokens=config['max_tokens'],
            timeout=160
        )

        print(f"✅ OCR completed, text length: {len(text)}")
        return jsonify({
            'success': True,
            'text': text,
            'config': {
                'content_type': content_type,
                'subcategory': subcategory,
                'complexity': complexity,
                'image_size': config['image_size'],
                'max_tokens': config['max_tokens']
            }
        })
    
    except TimeoutError:
        return jsonify({'error': 'OCR processing timeout'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500
    finally:
        if img:
            img.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        gc.collect()

# ==============================================================================
# PDF Processing (unchanged)
# ==============================================================================

def cleanup_old_tasks():
    now = datetime.now()

    # Cleanup PDF tasks
    expired = [tid for tid, t in pdf_tasks.items() if now - t['created_at'] > timedelta(minutes=30)]
    for tid in expired:
        pdf_file_path = pdf_tasks[tid].get('pdf_path')
        if pdf_file_path and os.path.exists(pdf_file_path):
            try:
                os.remove(pdf_file_path)
                print(f"🗑️ Removed expired PDF file: {pdf_file_path}")
            except Exception as e:
                print(f"❌ Error removing expired PDF file {pdf_file_path}: {e}")
        del pdf_tasks[tid]

    # Cleanup preprocessing tasks
    expired = [tid for tid, t in preprocess_tasks.items() if now - t['created_at'] > timedelta(minutes=30)]
    for tid in expired:
        task_dir = preprocess_tasks[tid].get('task_dir')
        if task_dir and os.path.exists(task_dir):
            try:
                shutil.rmtree(task_dir)
                print(f"🗑️ Removed expired preprocess task: {task_dir}")
            except Exception as e:
                print(f"❌ Error removing preprocess task {task_dir}: {e}")
        del preprocess_tasks[tid]

    # Cleanup video tasks
    expired = [tid for tid, t in video_tasks.items() if now - t['created_at'] > timedelta(minutes=30)]
    for tid in expired:
        task_dir = video_tasks[tid].get('task_dir')
        if task_dir and os.path.exists(task_dir):
            try:
                shutil.rmtree(task_dir)
                print(f"🗑️ Removed expired video task: {task_dir}")
            except Exception as e:
                print(f"❌ Error removing video task {task_dir}: {e}")
        del video_tasks[tid]

    if expired:
        print(f"🧹 Cleaned up {len(expired)} expired tasks")

@app.route('/api/pdf/init', methods=['POST'])
def init_pdf_task():
    cleanup_old_tasks()
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    # Determine whether to use new classification or old mode
    old_mode = request.form.get('mode')
    if old_mode and old_mode in MODE_QUICK_MAP:
        mapping = MODE_QUICK_MAP[old_mode]
        content_type = mapping['content_type']
        subcategory = mapping['subcategory']
        complexity = mapping['complexity']
    else:
        content_type = request.form.get('content_type', 'Document')
        subcategory = request.form.get('subcategory', 'Academic')
        complexity = request.form.get('complexity', 'Medium')

    pdf_save_path = None
    thumbnails = []

    try:
        task_id = str(uuid.uuid4())
        pdf_filename = secure_filename(file.filename)
        pdf_save_path = Path(UPLOAD_FOLDER) / f"{task_id}_{pdf_filename}"
        file.save(pdf_save_path)
        print(f"📄 Saved PDF for task {task_id} to: {pdf_save_path}")

        doc = fitz.open(pdf_save_path)
        total_pages = len(doc)
        print(f"📄 Processing PDF with {total_pages} pages for thumbnails")

        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            img_thumb = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            img_thumb.thumbnail((200, 200), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            img_thumb.save(buffered, format="PNG")
            thumb_b64 = base64.b64encode(buffered.getvalue()).decode()
            thumbnails.append(f"data:image/png;base64,{thumb_b64}")
            img_thumb.close()
            del pix

            if page_num % 10 == 0:
                print(f"📊 Generated thumbnails for {page_num + 1}/{total_pages} pages")

        doc.close()

        pdf_tasks[task_id] = {
            'pdf_path': str(pdf_save_path),
            'thumbnails': thumbnails,
            'content_type': content_type,
            'subcategory': subcategory,
            'complexity': complexity,
            'created_at': datetime.now(),
            'total_pages': total_pages
        }

        print(f"✅ PDF task initialized: {task_id}, pages: {total_pages}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'total_pages': total_pages,
            'thumbnails': thumbnails
        })
    except Exception as e:
        traceback.print_exc()
        if pdf_save_path and os.path.exists(pdf_save_path):
            os.remove(pdf_save_path)
        return jsonify({'error': f'PDF initialization failed: {str(e)}'}), 500
    finally:
        gc.collect()

@app.route('/api/pdf/extract-pages', methods=['POST'])
def extract_pdf_pages():
    """Extract all PDF pages as image files for preprocessing"""
    data = request.get_json()
    task_id = data.get('task_id')

    if not task_id:
        return jsonify({'success': False, 'error': 'Task ID is required'}), 400

    if task_id not in pdf_tasks:
        return jsonify({'success': False, 'error': 'Task expired or not found'}), 404

    task = pdf_tasks[task_id]
    pdf_path = task.get('pdf_path')
    total_pages = task.get('total_pages')

    if not pdf_path:
        return jsonify({'success': False, 'error': 'PDF path not found in task'}), 400

    if not total_pages or total_pages <= 0:
        return jsonify({'success': False, 'error': 'Invalid total pages'}), 400

    # Check if PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return jsonify({'success': False, 'error': f'PDF file not found: {pdf_path}'}), 404

    # Create temporary directory to save extracted images
    extract_dir = Path(UPLOAD_FOLDER) / f"pdf_extract_{task_id}"
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to create extract directory: {str(e)}'}), 500

    image_files = []
    doc = None

    try:
        doc = fitz.open(str(pdf_path))
        print(f"📄 Extracting {total_pages} pages from PDF for preprocessing...")

        upload_folder_path = Path(UPLOAD_FOLDER).resolve()

        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save as PNG file
                filename = f"page_{page_num + 1}.png"
                file_path = extract_dir / filename
                img.save(file_path, 'PNG')

                # Generate thumbnail
                thumb = img.copy()
                thumb.thumbnail((200, 200), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                thumb.save(buffered, format="PNG")
                thumb_b64 = base64.b64encode(buffered.getvalue()).decode()

                # Calculate relative path for API access
                try:
                    file_path_resolved = file_path.resolve()
                    relative_path = str(file_path_resolved.relative_to(upload_folder_path))
                except ValueError:
                    # If cannot calculate relative path, use filename
                    relative_path = f"pdf_extract_{task_id}/{filename}"

                image_files.append({
                    'filename': filename,
                    'file_path': str(file_path),
                    'file_url': f"/api/files/{relative_path}",
                    'thumb_b64': f"data:image/png;base64,{thumb_b64}",
                    'page_number': page_num + 1
                })

                img.close()
                thumb.close()
                del pix

                if (page_num + 1) % 10 == 0:
                    print(f"📊 Extracted {page_num + 1}/{total_pages} pages")
            except Exception as e:
                print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                traceback.print_exc()
                # Continue processing next page
                continue

        if doc:
            doc.close()
            doc = None

        # Save extracted file paths to task
        task['extracted_images'] = image_files
        task['extract_dir'] = str(extract_dir)

        if len(image_files) == 0:
            return jsonify({'success': False, 'error': 'No pages were extracted successfully'}), 500

        print(f"✅ Extracted {len(image_files)}/{total_pages} pages from PDF")
        
        return jsonify({
            'success': True,
            'images': image_files,
            'total_images': len(image_files)
        })
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        print(f"❌ Error in extract_pdf_pages: {error_msg}")
        return jsonify({'success': False, 'error': f'Failed to extract pages: {error_msg}'}), 500
    finally:
        if doc:
            try:
                doc.close()
            except:
                pass
        gc.collect()

@app.route('/api/files/<path:filepath>')
def serve_file(filepath):
    """Serve file access service"""
    try:
        # Parse file path to prevent path traversal attacks
        file_path = Path(UPLOAD_FOLDER) / filepath
        file_path = file_path.resolve()
        upload_folder_path = Path(UPLOAD_FOLDER).resolve()

        # Ensure file is in UPLOAD_FOLDER directory
        try:
            file_path.relative_to(upload_folder_path)
        except ValueError:
            return jsonify({'error': 'Access denied: File outside upload folder'}), 403
        
        if not file_path.exists() or not file_path.is_file():
            return jsonify({'error': 'File not found'}), 404

        return send_file(file_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/pdf/preview-page', methods=['POST'])
def preview_page():
    data = request.get_json()
    task_id = data.get('task_id')
    page_number = data.get('page_number', 1)

    if task_id not in pdf_tasks:
        return jsonify({'error': 'Task expired or not found'}), 404

    task = pdf_tasks[task_id]
    pdf_path = task['pdf_path']
    total_pages = task['total_pages']

    if page_number < 1 or page_number > total_pages:
        return jsonify({'error': 'Page out of range'}), 400

    img_b64 = None
    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        img.close()
        del pix

        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_b64}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate preview: {str(e)}'}), 500
    finally:
        if doc:
            doc.close()
        gc.collect()

@app.route('/api/pdf/process-batch', methods=['POST'])
def process_pdf_batch():
    data = request.get_json()
    task_id = data.get('task_id')
    batch_index = data.get('batch_index', 0)
    batch_size = data.get('batch_size', 2)
    processed_images = data.get('processed_images', {})  # ===== Fix: Receive processed image path mapping ====
    
    if task_id not in pdf_tasks:
        return jsonify({'error': 'Task not found or expired'}), 404

    task = pdf_tasks[task_id]
    pdf_path = task['pdf_path']
    content_type = task['content_type']
    subcategory = task['subcategory']
    complexity = task['complexity']
    total_pages = task['total_pages']

    # Get preprocessing configuration
    config = get_preprocessing_config(content_type, subcategory, complexity)
    if not config:
        return jsonify({'error': f'Invalid configuration: {content_type}/{subcategory}/{complexity}'}), 400

    # Use default prompt
    prompt = prompts.get('basic', '<image>\nExtract all text from the image.')

    start_page_idx = batch_index * batch_size
    end_page_idx = min(start_page_idx + batch_size, total_pages)

    if not is_model_healthy():
        return jsonify({'error': 'Model not ready. Please check server status.'}), 500

    results = []
    doc = None

    try:
        # ===== Fix: If processed images exist, use processed images; otherwise use original PDF =====
        use_processed_images = bool(processed_images)

        if use_processed_images:
            print(f"🔍 Processing batch {batch_index + 1} with PREPROCESSED images, pages {start_page_idx + 1}-{end_page_idx}")
            print(f"📋 Received processed_images keys: {list(processed_images.keys())}")
        else:
            doc = fitz.open(pdf_path)
            print(f"🔍 Processing batch {batch_index + 1}, pages {start_page_idx + 1}-{end_page_idx}")

        for i in range(start_page_idx, end_page_idx):
            page_num = i + 1
            pix = None  # Initialize pix variable

            if use_processed_images and str(page_num) in processed_images:
                # ===== Use processed image =====
                processed_path = processed_images[str(page_num)]
                print(f"📄 Loading PREPROCESSED page {page_num}/{total_pages} for OCR...")

                # Load processed image (processed_path is full path)
                try:
                    file_path = Path(processed_path)
                    if not file_path.exists():
                        raise FileNotFoundError(f"Processed image not found: {processed_path}")

                    img = Image.open(file_path).convert('RGB')
                    print(f"✅ Loaded preprocessed image for page {page_num}: {file_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load preprocessed image for page {page_num}: {e}")
                    # Fallback to original PDF
                    if doc is None:
                        doc = fitz.open(pdf_path)
                    page = doc[page_num - 1]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                # ===== Use original PDF =====
                if doc is None:
                    doc = fitz.open(pdf_path)
                print(f"📄 Loading page {page_num}/{total_pages} for OCR...")
                page = doc[page_num - 1]

                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Preprocess based on configuration (resize)
            img_processed = preprocess_image_by_config(img, config['image_size'])

            print(f"📄 Processing page {page_num} with: {content_type}/{subcategory}/{complexity}")
            text = generate_with_timeout_and_process(
                image=img_processed,
                prompt=prompt,
                max_tokens=config['max_tokens'],
                timeout=160
            )

            results.append({'page': page_num, 'text': text})

            # Cleanup resources
            img.close()
            if pix is not None:
                del pix
            del img
            gc.collect()

            print(f"✅ Page {page_num} completed, text length: {len(text)}")
        
        has_more = end_page_idx < total_pages
        next_batch = batch_index + 1 if has_more else None

        print(f"✅ Batch {batch_index + 1} completed, processed: {end_page_idx}/{total_pages}")

        return jsonify({
            'success': True,
            'results': results,
            'has_more': has_more,
            'next_batch_index': next_batch,
            'processed_pages': end_page_idx,
            'config': {
                'content_type': content_type,
                'subcategory': subcategory,
                'complexity': complexity,
                'image_size': config['image_size'],
                'max_tokens': config['max_tokens']
            }
        })
    except TimeoutError:
        return jsonify({'error': 'OCR processing timeout'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500
    finally:
        if doc:
            doc.close()
        gc.collect()

@app.route('/api/video/process-batch', methods=['POST'])
def process_video_batch():
    """Process video frame batch OCR"""
    data = request.get_json()
    batch_index = data.get('batch_index', 0)
    batch_size = data.get('batch_size', 2)
    processed_images = data.get('processed_images', {})  # Processed image path mapping
    content_type = data.get('content_type', 'Document')
    subcategory = data.get('subcategory', 'Academic')
    complexity = data.get('complexity', 'Medium')
    
    if not processed_images:
        return jsonify({'error': 'No processed images provided'}), 400

    # Get preprocessing configuration
    config = get_preprocessing_config(content_type, subcategory, complexity)
    if not config:
        return jsonify({'error': f'Invalid configuration: {content_type}/{subcategory}/{complexity}'}), 400

    # Use default prompt
    prompt = prompts.get('basic', '<image>\nExtract all text from the image.')

    # Get all frame numbers and sort
    frame_numbers = sorted([int(k) for k in processed_images.keys()])
    total_frames = len(frame_numbers)

    start_idx = batch_index * batch_size
    end_idx = min(start_idx + batch_size, total_frames)

    if not is_model_healthy():
        return jsonify({'error': 'Model not ready. Please check server status.'}), 500

    results = []

    try:
        print(f"🔍 Processing video frames batch {batch_index + 1}, frames {start_idx + 1}-{end_idx}")

        for i in range(start_idx, end_idx):
            if i >= len(frame_numbers):
                break

            frame_num = frame_numbers[i]
            processed_path = processed_images[str(frame_num)]

            print(f"📄 Loading PREPROCESSED frame {frame_num}/{total_frames} for OCR...")

            # Load processed image
            try:
                file_path = Path(processed_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Processed image not found: {processed_path}")

                img = Image.open(file_path).convert('RGB')
                print(f"✅ Loaded preprocessed image for frame {frame_num}: {file_path}")
            except Exception as e:
                print(f"⚠️ Failed to load preprocessed image for frame {frame_num}: {e}")
                results.append({'page': frame_num, 'text': '', 'error': str(e)})
                continue

            # Preprocess based on configuration (resize)
            img_processed = preprocess_image_by_config(img, config['image_size'])

            print(f"📄 Processing frame {frame_num} with: {content_type}/{subcategory}/{complexity}")
            text = generate_with_timeout_and_process(
                image=img_processed,
                prompt=prompt,
                max_tokens=config['max_tokens'],
                timeout=160
            )

            results.append({'page': frame_num, 'text': text})

            # Cleanup resources
            img.close()
            del img
            gc.collect()

            print(f"✅ Frame {frame_num} completed, text length: {len(text)}")

        has_more = end_idx < total_frames
        next_batch = batch_index + 1 if has_more else None

        print(f"✅ Video batch {batch_index + 1} completed, processed: {end_idx}/{total_frames}")
        
        return jsonify({
            'success': True,
            'results': results,
            'has_more': has_more,
            'next_batch_index': next_batch,
            'processed_pages': end_idx,
            'config': {
                'content_type': content_type,
                'subcategory': subcategory,
                'complexity': complexity,
                'image_size': config['image_size'],
                'max_tokens': config['max_tokens']
            }
        })
    except TimeoutError:
        return jsonify({'error': 'OCR processing timeout'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500
    finally:
        gc.collect()

@app.route('/api/pdf/cancel', methods=['POST'])
def cancel_pdf_task():
    data = request.get_json()
    task_id = data.get('task_id')
    if task_id in pdf_tasks:
        pdf_file_path = pdf_tasks[task_id].get('pdf_path')
        if pdf_file_path and os.path.exists(pdf_file_path):
            try:
                os.remove(pdf_file_path)
                print(f"🗑️ Removed PDF file for cancelled task: {pdf_file_path}")
            except Exception as e:
                print(f"❌ Error removing PDF file for cancelled task {pdf_file_path}: {e}")
        del pdf_tasks[task_id]
        gc.collect()
        print(f"❌ PDF task cancelled: {task_id}")
        return jsonify({'success': True})
    return jsonify({'error': 'Task not found or expired'}), 404

def cleanup():
    print("🧹 Cleaning up resources on application exit...")
    model_loaded_status.value = False

    # Cleanup all tasks
    for task_dict in [pdf_tasks, preprocess_tasks, video_tasks]:
        for task_id in list(task_dict.keys()):
            task = task_dict[task_id]
            task_dir = task.get('task_dir')
            pdf_path = task.get('pdf_path')

            if task_dir and os.path.exists(task_dir):
                try:
                    shutil.rmtree(task_dir)
                    print(f"🗑️ Removed task directory: {task_dir}")
                except Exception as e:
                    print(f"❌ Error removing task directory {task_dir}: {e}")

            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    print(f"🗑️ Removed PDF file: {pdf_path}")
                except Exception as e:
                    print(f"❌ Error removing PDF file {pdf_path}: {e}")

            del task_dict[task_id]

    gc.collect()
    print("✅ All resources cleaned up.")

atexit.register(cleanup)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    print(f"🚀 Starting OCR Butterfly with {MODEL_DISPLAY_NAME}...")
    if not preload_model_main_process():
        print("❌ CRITICAL: Model preload status setting failed. Cannot start application.")
        sys.exit(1)

    run_port = int(os.environ.get('PORT', 5001))

    print(f"✅ Application starting on http://0.0.0.0:{run_port}")
    print(f"\n🤖 Model: {MODEL_NAME}")
    print("📊 Features: OCR | PDF Batch | Video Extraction | Image Preprocessing\n")

    app.run(host='0.0.0.0', port=run_port, debug=False, threaded=True)
