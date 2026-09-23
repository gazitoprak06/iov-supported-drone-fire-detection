import cv2
import numpy as np

def get_ellipse_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def run_heuristic_pipeline(frame, prev_frame=None, h=50.0):
    """
    Implements the exact Algorithm 1 from Section 3 of the manuscript:
    Multi-Color-Space Pixel Intersection (V1) + Temporal Flickering (V2) + NMS & Merge.
    """
    # 1. Multi-Color-Space Pixel Intersection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    # OpenCV Hue is 0-179, S is 0-255, V is 0-255
    m_hsv = (((H >= 0) & (H <= 15) & (S >= 60)) | 
             ((H >= 15) & (H <= 35) & (S >= 80)) | 
             ((H >= 160) & (H <= 180) & (S >= 60))) & (V >= 210)
    m_hsv = m_hsv.astype(np.uint8) * 255
    
    B, G, R = cv2.split(frame)
    m_rgb = (R > G) & (G > B) & (R > 210) & (G > 130)
    m_rgb = m_rgb.astype(np.uint8) * 255
    
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycc)
    m_ycc = (Cr > 150) & (Cr > Cb) & (Y > Cb)
    m_ycc = m_ycc.astype(np.uint8) * 255
    
    M = cv2.bitwise_and(m_hsv, cv2.bitwise_and(m_rgb, m_ycc))
    
    # White-hot bypass mask
    m_wh = ((R > 240) & (G > 230) & (B > 200)).astype(np.uint8) * 255
    kernel_15 = get_ellipse_kernel(15)
    m_wh = cv2.erode(m_wh, kernel_15)
    
    M = cv2.bitwise_or(M, m_wh)
    
    # Morphology
    M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, kernel_15)
    M = cv2.morphologyEx(M, cv2.MORPH_OPEN, get_ellipse_kernel(5))
    M = cv2.dilate(M, kernel_15, iterations=2)
    
    # 2. Temporal Flickering (V2 logic)
    m_flick = None
    if prev_frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_bin = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        m_flick = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, get_ellipse_kernel(7))
        
    # Minimum Area threshold based on altitude h
    A_min = 600.0 * (50.0 / max(1.0, float(h)))**2
    contours, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > A_min:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            s = area / hull_area if hull_area > 0 else 0
            
            x, y, wb, hb = cv2.boundingRect(c)
            
            if m_flick is not None:
                roi = m_flick[y:y+hb, x:x+wb]
                rho = np.mean(roi) / 255.0
                delta = -0.30 if rho < 0.01 else 0.30 * rho
            else:
                rho = 0.0
                delta = 0.0
                
            # Base heuristic score + temporal delta
            c_alg = min(1.0, 0.60 + 0.10 * s + delta)
            if c_alg >= 0.50:
                boxes.append({
                    'rect': (x, y, wb, hb),
                    'score': c_alg,
                    'area': area
                })
                
    # 3. Object Localization (Asymmetric NMS)
    surviving = []
    for i, bA in enumerate(boxes):
        keep = True
        xA, yA, wA, hA = bA['rect']
        areaA = wA * hA
        for j, bB in enumerate(boxes):
            if i == j: continue
            xB, yB, wB, hB = bB['rect']
            areaB = wB * hB
            if areaB > areaA:
                xi1, yi1 = max(xA, xB), max(yA, yB)
                xi2, yi2 = min(xA+wA, xB+wB), min(yA+hA, yB+hB)
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    if inter / float(areaA) > 0.30:
                        keep = False
                        break
        if keep:
            surviving.append(bA)
            
    # Proximity Merge
    d_merge = max(20.0, 100.0 - 0.5 * h)
    while True:
        merged = False
        new_boxes = []
        used = set()
        
        for i, bA in enumerate(surviving):
            if i in used: continue
            xA, yA, wA, hA = bA['rect']
            merged_box = dict(bA)
            
            for j, bB in enumerate(surviving):
                if i == j or j in used: continue
                xB, yB, wB, hB = bB['rect']
                
                # Check separation
                dx = max(0, max(xA - (xB + wB), xB - (xA + wA)))
                dy = max(0, max(yA - (yB + hB), yB - (yA + hA)))
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= d_merge:
                    # Merge boxes
                    x_new, y_new = min(xA, xB), min(yA, yB)
                    w_new = max(xA+wA, xB+wB) - x_new
                    h_new = max(yA+hA, yB+hB) - y_new
                    
                    merged_box['rect'] = (x_new, y_new, w_new, h_new)
                    merged_box['score'] = max(merged_box['score'], bB['score'])
                    merged_box['area'] += bB['area']
                    
                    used.add(j)
                    merged = True
                    xA, yA, wA, hA = x_new, y_new, w_new, h_new
            
            used.add(i)
            new_boxes.append(merged_box)
            
        surviving = new_boxes
        if not merged:
            break
            
    return surviving
