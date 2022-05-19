import time
l = [] # List of images

def find_frame(seq):
    for frame in l:
        if frame.getSequenceNum() == seq:
            return frame
def correct_bb(bb):
    bb.xmin = max(0, bb.xmin)
    bb.ymin = max(0, bb.ymin)
    bb.xmax = min(bb.xmax, 1)
    bb.ymax = min(bb.ymax, 1)
    return bb
while True:
    time.sleep(0.001)
    preview = node.io['frame'].tryGet()
    if preview is not None:
        # node.warn(f"New frame {preview.getSequenceNum()}, size {len(l)}")
        l.append(preview)
        # Max pool size is 10.
        if 3 < len(l):
            l.pop(0)

    face_dets = node.io['face_det_in'].tryGet()
    if face_dets is not None:
        # node.warn(f"New detection start")
        passthrough = node.io['face_pass'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New detection {seq}")
        if len(l) == 0:
            continue

        img = find_frame(seq) # Matching frame is the first in the list
        if img is None:
            continue

        for det in face_dets.detections:
            # bboxes.append(det) # For the rotation
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(112, 112)
            cfg.setKeepAspectRatio(False)
            node.warn(f"New detection {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)
