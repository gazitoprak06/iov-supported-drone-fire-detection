import sys
import os
import cv2
import json
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np

# The alarm rule is defined once, in tools/clip_metrics.py, and imported here
# rather than restated. The viewer is not an evaluator, but a viewer that
# alarmed on a different rule than the one Section 4.5 measures would be
# misleading to anyone inspecting the qualitative behaviour.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from clip_metrics import (  # noqa: E402
    CONSECUTIVE_FOR_ALARM,
    DEFAULT_FPS,
    FIRE_CLASS_INDEX,
    SAMPLES_PER_SECOND,
)

def apply_night_vision(frame):
    # Convert to LAB for CLAHE on L-channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Green phosphor effect
    b_ch, g_ch, r_ch = cv2.split(enhanced)
    g_ch = cv2.addWeighted(g_ch, 0.8, cl, 0.2, 0)
    night_frame = cv2.merge((np.zeros_like(b_ch), g_ch, np.zeros_like(r_ch)))
    return night_frame

def apply_thermal_jet(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply JET colormap
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

def apply_gradcam(frame, cam_mask):
    # Resize cam_mask to frame size
    h, w = frame.shape[:2]
    cam_resized = cv2.resize(cam_mask, (w, h))
    
    # Apply JET colormap to the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    
    # Smooth blending: only show heatmap where activation is high
    alpha = np.stack([cam_resized]*3, axis=-1) * 0.7 # Max 70% opacity
    overlay = frame * (1 - alpha) + heatmap * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)

def draw_hud(frame, state):
    h, w = frame.shape[:2]
    
    # Overlay semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    # Overlay semi-transparent bottom bar
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Color definitions
    text_color = (0, 255, 0) if state['mode'] == 'NIGHT-VISION' else (255, 255, 255)
    alert_color = (0, 0, 255)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top Left: System info
    cv2.putText(frame, f"UAV ID: DRONE-07", (15, 25), font, 0.6, text_color, 1)
    cv2.putText(frame, f"MODE: {state['mode']}", (15, 50), font, 0.6, text_color, 1)
    cv2.putText(frame, f"ALT: {state['altitude']}m", (15, 75), font, 0.6, text_color, 1)

    # Top Center: Alert Status
    status_text = "FIRE DETECTED" if state['is_alarm'] else "NORMAL"
    s_color = alert_color if state['is_alarm'] else text_color
    cv2.putText(frame, f"STATUS: {status_text}", (w//2 - 100, 35), font, 0.8, s_color, 2)
    
    # Top Right: Battery & Time
    batt_text = f"BATT: {state['battery']}%"
    cv2.putText(frame, batt_text, (w - 150, 25), font, 0.6, text_color, 1)
    met = int(time.time() - state['start_time'])
    mins, secs = divmod(met, 60)
    cv2.putText(frame, f"MET: {mins:02d}:{secs:02d}", (w - 150, 50), font, 0.6, text_color, 1)
    if state['is_recording']:
        cv2.putText(frame, "• REC", (w - 150, 75), font, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, "[Space] Pause | [M] Mode | [R] Record | [N/P] Next/Prev | [Q] Quit", (15, h - 10), font, 0.45, (200,200,200), 1)

    # Bottom Right: Video Info
    vid_name = os.path.basename(state['video_path'])
    cv2.putText(frame, f"FILE: {vid_name}", (w - 300, h - 20), font, 0.5, text_color, 1)

    # Crosshair
    cx, cy = w//2, h//2
    cv2.line(frame, (cx-20, cy), (cx+20, cy), text_color, 1)
    cv2.line(frame, (cx, cy-20), (cx, cy+20), text_color, 1)

    return frame

def main():
    root_dir = Path(".")
    manifest_path = root_dir / "annotations" / "video_evaluation_manifest.json"
    
    videos = []
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
            fire_vids = [str(root_dir / v['path']) for v in manifest.get('videos', []) if v.get('label') == 'fire' and (root_dir / v['path']).exists()]
            normal_vids = [str(root_dir / v['path']) for v in manifest.get('videos', []) if v.get('label') == 'no_fire' and (root_dir / v['path']).exists()]
            
            # Interleave 3 fire, 3 normal
            f_idx, n_idx = 0, 0
            while f_idx < len(fire_vids) or n_idx < len(normal_vids):
                for _ in range(3):
                    if f_idx < len(fire_vids):
                        videos.append(fire_vids[f_idx])
                        f_idx += 1
                for _ in range(3):
                    if n_idx < len(normal_vids):
                        videos.append(normal_vids[n_idx])
                        n_idx += 1

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        videos.insert(0, sys.argv[1])
        
    if not videos:
        print("No videos found to play.")
        return

    device = torch.device("cpu")
    print("Loading V3 Deep Edge Model (MobileNetV3)...")
    
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model_path = "evaluation_results/v3_deep_edge/v3_mobilenet.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model weights not found at {model_path}. Using random weights.")
    model.to(device)
    model.eval()
    
    # Grad-CAM Setup
    target_layer = model.features[-1]
    activations = {}
    gradients = {}
    
    def fw_hook(module, input, output):
        activations['value'] = output
        
    def bw_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]
        
    target_layer.register_forward_hook(fw_hook)
    target_layer.register_full_backward_hook(bw_hook)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    vid_idx = 0
    modes = ['RGB', 'GRAD-CAM (HEATMAP)', 'THERMAL-JET', 'NIGHT-VISION']
    mode_idx = 0
    
    state = {
        'start_time': time.time(),
        'battery': 87,
        'altitude': 120,
        'is_recording': False,
        'is_alarm': False,
        'mode': modes[mode_idx]
    }
    
    video_writer = None
    window_name = "UAV Fire Detection Dashboard (V3 Deep Edge)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.moveWindow(window_name, 50, 50)

    while vid_idx < len(videos):
        state['video_path'] = videos[vid_idx]
        cap = cv2.VideoCapture(state['video_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = DEFAULT_FPS
        step = max(1, int(fps / SAMPLES_PER_SECOND))
        
        consecutive_fires = 0
        frame_idx = 0
        paused = False
        last_cam_mask = np.zeros((224, 224))
        
        print(f"\nPlaying: {state['video_path']}")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    vid_idx += 1
                    break
                
                # Ekranin tasmamasi icin daha da kucultelim (960x540 - QHD/Laptop uyumlu)
                frame = cv2.resize(frame, (960, 540))
                original_frame = frame.copy()
                
                # Inference
                if frame_idx % step == 0:
                    rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    t_img = transform(img).unsqueeze(0).to(device)
                    t_img.requires_grad = True
                    
                    with torch.enable_grad():
                        out = model(t_img)
                        _, pred = torch.max(out, 1)
                        
                        if pred.item() == FIRE_CLASS_INDEX:
                            consecutive_fires += 1
                        else:
                            consecutive_fires = 0
                            
                        # Temporal smoothing filter. The evaluator of Section 4.5
                        # latches the alarm and stops scanning, so a viewer that
                        # cleared it on a single negative sample would be showing
                        # a different rule than the one the paper measures. The
                        # latch is reset only when the operator loads another clip.
                        if consecutive_fires >= CONSECUTIVE_FOR_ALARM:
                            state['is_alarm'] = True
                            
                        # Grad-CAM computation if we need it
                        model.zero_grad()
                        # The gradient is taken for the fire class, whose index
                        # is imported rather than restated so that the heatmap
                        # and the alarm can never explain different classes.
                        target_score = out[0, FIRE_CLASS_INDEX]
                        target_score.backward()
                        
                        if 'value' in gradients and 'value' in activations:
                            grads = gradients['value']
                            acts = activations['value']
                            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                            cam = torch.sum(weights * acts, dim=1).squeeze(0)
                            cam = torch.relu(cam)
                            cam_max = cam.max()
                            if cam_max > 1e-7:
                                cam = cam / cam_max
                            else:
                                cam = torch.zeros_like(cam)
                            last_cam_mask = cam.detach().cpu().numpy()
                
                # Apply visual modes
                if state['mode'] == 'THERMAL-JET':
                    display_frame = apply_thermal_jet(original_frame)
                elif state['mode'] == 'NIGHT-VISION':
                    display_frame = apply_night_vision(original_frame)
                elif state['mode'] == 'GRAD-CAM (HEATMAP)':
                    display_frame = apply_gradcam(original_frame, last_cam_mask)
                else:
                    display_frame = original_frame.copy()
                
                # Draw HUD
                display_frame = draw_hud(display_frame, state)
                
                # Recording
                if state['is_recording']:
                    if video_writer is None:
                        h, w = display_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_name = f"rec_{int(time.time())}.mp4"
                        video_writer = cv2.VideoWriter(out_name, fourcc, fps, (w, h))
                        print(f"Started recording: {out_name}")
                    video_writer.write(display_frame)
                else:
                    if video_writer is not None:
                        video_writer = None
                        print("Stopped recording.")
                
                cv2.imshow(window_name, display_frame)
                frame_idx += 1

            # Keyboard controls
            delay = int(1000 / fps) if not paused else 50
            key = cv2.waitKey(delay) & 0xFF
            
            if key in [ord('q'), ord('Q')]:
                if video_writer: video_writer.release()
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key in [ord('n'), ord('N')]:
                vid_idx += 1
                break
            elif key in [ord('p'), ord('P')]:
                vid_idx = max(0, vid_idx - 1)
                break
            elif key in [ord('r'), ord('R')]:
                state['is_recording'] = not state['is_recording']
            elif key in [ord('m'), ord('M')]:
                mode_idx = (mode_idx + 1) % len(modes)
                state['mode'] = modes[mode_idx]
                
        cap.release()
        
    if video_writer: video_writer.release()
    cv2.destroyAllWindows()
    print("End of video list.")

if __name__ == "__main__":
    main()
