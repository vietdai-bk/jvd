import sys
import os
import cv2
import time
import multiprocessing as mp
from pathlib import Path
from typing import Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jvd.core.pipeline import JunctionDetectorPipeline
from jvd.utils.visualizer import Visualizer
from jvd.utils.logger import setup_logger
from jvd.utils.roi_editor import ROIEditor
from jvd.utils.evidence_recorder import EvidenceRecorder
from jvd.utils.plate_reader import PlateReader
from jvd.utils.violation_logger import ViolationLogger
from jvd.config import load_config, save_config

def ocr_worker(input_queue, shared_plate_cache, config):
    plate_reader = PlateReader(config)
    while True:
        task = input_queue.get()
        if task is None: break
        tid, frame, bbox, frame_count = task
        plate_text = plate_reader.read(tid, frame, bbox, frame_count)
        shared_plate_cache[tid] = plate_text if plate_text else "N/A"

def storage_worker(log_queue, config):
    ev_cfg = config.get('evidence', {})
    recorder = EvidenceRecorder(
        output_dir=ev_cfg.get('output_dir', 'evidence'),
        fps=config['video'].get('fps', 25.0),
        pre_seconds=ev_cfg.get('pre_seconds', 3.0),
        post_seconds=ev_cfg.get('post_seconds', 10.0)
    )
    violation_logger = ViolationLogger(config)
    while True:
        data = log_queue.get()
        if data is None: break
        if data['type'] == 'frame':
            recorder.push_frame(data['frame'])
        elif data['type'] == 'violation':
            v = data['violation_obj']
            img_path = recorder.trigger(v.event.track_id, data['img'], v.event.class_label.display_name)
            violation_logger.log(
                track_id=v.event.track_id,
                vehicle_type=v.event.class_label.display_name,
                plate_number=data['plate'],
                dwell_time_sec=v.dwell_time_seconds,
                violation_type=v.violation_type,
                snapshot_path=img_path,
                video_path=str(Path(img_path).with_suffix('.mp4'))
            )

def ask_roi_setup(config: dict, config_path: Path, video_path: str, logger) -> dict:
    print("\n" + "="*40)
    print("      HỆ THỐNG THIẾT LẬP ROI")
    print("="*40)
    print("  [y] Vẽ lại vùng ROI")
    print("  [n] Dùng ROI cũ (Mặc định)")
    print("  [r] Reset ROI toàn màn hình")
    try:
        choice = input("Lựa chọn của bạn [y/n/r]: ").strip().lower()
    except EOFError:
        choice = 'n'
    if choice == 'r':
        config['roi']['points'] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        save_config(config, config_path)
    elif choice == 'y':
        editor = ROIEditor(config_path)
        new_roi = editor.run(video_path, config['roi']['points'])
        if new_roi:
            config['roi']['points'] = new_roi
            save_config(config, config_path)
    return config

def main():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    config = load_config(str(config_path))
    logger = setup_logger("JVD_Pi5_Final")
    video_path = config['video']['path']

    config = ask_roi_setup(config, config_path, video_path, logger)

    mp.set_start_method('spawn', force=True)
    ocr_queue = mp.Queue(maxsize=5)
    log_queue = mp.Queue(maxsize=30)
    manager = mp.Manager()
    shared_plate_cache = manager.dict()
    ocr_sent_ids = set()

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frame)
    config['video']['fps'] = fps_video

    p_ocr = mp.Process(target=ocr_worker, args=(ocr_queue, shared_plate_cache, config))
    p_store = mp.Process(target=storage_worker, args=(log_queue, config))
    p_ocr.start()
    p_store.start()

    pipeline = JunctionDetectorPipeline(config)
    visualizer = Visualizer(config)
    
    frame_count = 0
    prev_time = time.time()
    avg_fps = 0.0
    alpha = 0.1 
    
    window_name = "JVD Pi5 - Turbo Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            current_time_sec = frame_count / fps_video
            
            events, violations, states, _ = pipeline.process_frame(frame, current_time_sec)

            # Tạo dictionary ánh xạ nhanh dwell_time từ danh sách violations hiện tại
            violation_dwell_map = {v.event.track_id: v.dwell_time_seconds for v in violations}

            display_frame = frame.copy()
            roi_px = pipeline.get_roi_points_pixel(width, height)
            display_frame = visualizer.draw_roi(display_frame, roi_px)

            for ev in events:
                tid = ev.track_id
                if tid is None: continue
                
                state = states.get(tid)
                # Ưu tiên 1: Lấy từ map violations vừa trả về
                # Ưu tiên 2: Lấy từ thuộc tính dwell_time của state
                # Ưu tiên 3: Lấy từ thuộc tính last_dwell_time (tên phổ biến khác)
                dwell_time = violation_dwell_map.get(tid, 
                             getattr(state, 'dwell_time', 
                             getattr(state, 'last_dwell_time', 0.0)))

                if state:
                    if state.violation_triggered and tid not in ocr_sent_ids:
                        if not ocr_queue.full():
                            ocr_queue.put((tid, frame.copy(), ev.bbox, frame_count))
                            ocr_sent_ids.add(tid)

                plate = shared_plate_cache.get(tid, "SCANNING..." if tid in ocr_sent_ids else "")
                is_violating = tid in ocr_sent_ids
                
                display_frame = visualizer.draw_vehicle(
                    display_frame, ev, state, is_violating, dwell_time, plate
                )

            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                instant_fps = 1.0 / time_diff
                avg_fps = (alpha * instant_fps) + ((1.0 - alpha) * avg_fps) if avg_fps > 0 else instant_fps
            prev_time = curr_time
            
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (20, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

            if not log_queue.full():
                log_queue.put({'type': 'frame', 'frame': display_frame.copy()})
            
            for v in violations:
                tid = v.event.track_id
                state = states.get(tid)
                if state and state.violation_triggered and not getattr(state, 'logged_to_csv', False):
                    state.logged_to_csv = True
                    if not log_queue.full():
                        log_queue.put({
                            'type': 'violation',
                            'violation_obj': v,
                            'img': display_frame.copy(),
                            'plate': shared_plate_cache.get(tid, "N/A")
                        })

            show_frame = cv2.resize(display_frame, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, show_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        ocr_queue.put(None)
        log_queue.put(None)
        p_ocr.join()
        p_store.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()