import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue
import time

class OptimizedOpticalFlow:
    def __init__(self, grid_step=20, preset="balanced"):
        """
        Initialize optical flow tracker with performance presets
        
        Presets:
        - 'performance': Maximum FPS, lower quality
        - 'balanced': Good balance (default)
        - 'quality': Best quality, lower FPS
        """
        self.grid_step = grid_step
        self.preset = preset
        self.fullscreen = False
        self.show_ui = True
        
        # Performance presets
        self.presets = {
            'performance': {
                'resolution': (2560, 1600),
                'pyr_scale': 0.5,
                'levels': 2,
                'winsize': 10,
                'iterations': 2,
                'poly_n': 5,
                'poly_sigma': 1.1,
                'skip_frames': 0,
                'arrow_scale': 2.5
            },
            'balanced': {
                'resolution': (640, 800),
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'skip_frames': 0,
                'arrow_scale': 3
            },
            'quality': {
                'resolution': (640, 800),
                'pyr_scale': 0.5,
                'levels': 4,
                'winsize': 20,
                'iterations': 5,
                'poly_n': 7,
                'poly_sigma': 1.5,
                'skip_frames': 0,
                'arrow_scale': 3.5
            }
        }
        
        self.config = self.presets[preset]
        
        # Flow calculation parameters
        self.flow_params = dict(
            pyr_scale=self.config['pyr_scale'],
            levels=self.config['levels'],
            winsize=self.config['winsize'],
            iterations=self.config['iterations'],
            poly_n=self.config['poly_n'],
            poly_sigma=self.config['poly_sigma'],
            flags=0
        )
        
        # Threading
        self.flow_queue = Queue(maxsize=2)
        self.frame_queue = Queue(maxsize=2)
        self.flow_lock = Lock()
        self.running = False
        
        # State
        self.old_gray = None
        self.current_flow = None
        self.frame_count = 0
        self.fps = 0
        self.processing_time = 0
        
    def calculate_flow_thread(self):
        """Background thread for flow calculation"""
        local_old_gray = None
        
        while self.running:
            if not self.frame_queue.empty():
                frame_gray = self.frame_queue.get()
                
                if local_old_gray is not None:
                    start_time = time.time()
                    
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        local_old_gray, frame_gray, None, **self.flow_params
                    )
                    
                    self.processing_time = time.time() - start_time
                    
                    # Store result
                    with self.flow_lock:
                        self.current_flow = flow
                
                local_old_gray = frame_gray.copy()
            else:
                time.sleep(0.001)
    
    def draw_flow_arrows(self, frame, flow):
        """Draw clean, minimalist arrows"""
        if flow is None:
            return frame
        
        h, w = frame.shape[:2]
        arrow_layer = frame.copy()
        
        # Create grid points
        y_coords = np.arange(self.grid_step, h, self.grid_step)
        x_coords = np.arange(self.grid_step, w, self.grid_step)
        
        for y in y_coords:
            for x in x_coords:
                fx, fy = flow[y, x]
                
                # Scale vectors
                fx_scaled = fx * self.config['arrow_scale']
                fy_scaled = fy * self.config['arrow_scale']
                
                magnitude = np.sqrt(fx**2 + fy**2)
                
                # Only draw if there's significant motion
                if magnitude < 0.8:
                    continue
                
                # End point
                x_end = int(x + fx_scaled)
                y_end = int(y + fy_scaled)
                
                # Clean color scheme: cyan for slow, magenta for fast
                intensity = min(1.0, magnitude / 10.0)
                
                # Interpolate between cyan (0, 255, 255) and magenta (255, 0, 255)
                r = int(intensity * 255)
                g = int((1 - intensity) * 255)
                b = 255
                color = (b, g, r)
                
                # Draw clean, thin arrows
                cv2.arrowedLine(arrow_layer, (x, y), (x_end, y_end), 
                               color, 1, tipLength=0.3, line_type=cv2.LINE_AA)
        
        return arrow_layer
    
    def draw_modern_ui(self, frame):
        """Draw clean, minimalist UI overlay"""
        if not self.show_ui:
            return frame
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent dark panel at top
        panel_height = 50
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Font settings - clean and small
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (255, 255, 255)
        
        # Compact info display
        info_text = f"FPS {self.fps} · {self.preset.upper()} · {self.grid_step}px · {self.processing_time*1000:.0f}ms"
        
        cv2.putText(frame, info_text, (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Controls hint (smaller, bottom right)
        controls = "F:fullscreen  U:UI  1-3:preset  +/-:density  Q:quit"
        text_size = cv2.getTextSize(controls, font, 0.35, 1)[0]
        cv2.putText(frame, controls, (w - text_size[0] - 10, h - 10), 
                   font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def change_preset(self, preset):
        """Change performance preset on the fly"""
        if preset in self.presets:
            self.preset = preset
            self.config = self.presets[preset]
            self.flow_params.update({
                'pyr_scale': self.config['pyr_scale'],
                'levels': self.config['levels'],
                'winsize': self.config['winsize'],
                'iterations': self.config['iterations'],
                'poly_n': self.config['poly_n'],
                'poly_sigma': self.config['poly_sigma']
            })
            print(f"→ Preset: {preset}")
            return True
        return False
    
    def run(self):
        """Main loop with optimizations"""
        cap = cv2.VideoCapture(0)
        
        # Set resolution based on preset
        res_w, res_h = self.config['resolution']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Hardware acceleration
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        time.sleep(1)
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Cannot access camera")
            cap.release()
            return
        
        # Start flow calculation thread
        self.running = True
        flow_thread = Thread(target=self.calculate_flow_thread, daemon=True)
        flow_thread.start()
        
        # Initialize
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.old_gray = frame_gray
        
        window_name = 'Optical Flow'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\n" + "─" * 50)
        print("  OPTICAL FLOW - M1 OPTIMIZED")
        print("─" * 50)
        print(f"  Preset: {self.preset}")
        print(f"  Resolution: {res_w}×{res_h}")
        print("\n  Controls:")
        print("    1-3  : Change preset")
        print("    +/-  : Arrow density")
        print("    F    : Fullscreen")
        print("    U    : Toggle UI")
        print("    Q    : Quit")
        print("─" * 50 + "\n")
        
        fps_time = time.time()
        fps_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Send to processing thread
            if self.frame_queue.qsize() < 2:
                self.frame_queue.put(frame_gray)
            
            # Get latest flow result
            with self.flow_lock:
                flow = self.current_flow
            
            # Skip frame processing if configured
            if self.config['skip_frames'] > 0:
                if self.frame_count % (self.config['skip_frames'] + 1) != 0:
                    self.frame_count += 1
                    continue
            
            # Draw arrows
            if flow is not None:
                vis_frame = self.draw_flow_arrows(frame, flow)
            else:
                vis_frame = frame
            
            # Add UI overlay
            vis_frame = self.draw_modern_ui(vis_frame)
            
            # FPS calculation
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            cv2.imshow(window_name, vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Fullscreen
            if key == ord('f'):
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                         cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                         cv2.WINDOW_NORMAL)
            
            # Toggle UI
            elif key == ord('u'):
                self.show_ui = not self.show_ui
                print(f"→ UI: {'ON' if self.show_ui else 'OFF'}")
            
            # Presets
            elif key == ord('1'):
                self.change_preset('performance')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['resolution'][0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['resolution'][1])
                res_w, res_h = self.config['resolution']
                
            elif key == ord('2'):
                self.change_preset('balanced')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['resolution'][0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['resolution'][1])
                res_w, res_h = self.config['resolution']
                
            elif key == ord('3'):
                self.change_preset('quality')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['resolution'][0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['resolution'][1])
                res_w, res_h = self.config['resolution']
            
            # Arrow density
            elif key == ord('+') or key == ord('='):
                if self.grid_step > 8:
                    self.grid_step -= 4
                    print(f"→ Grid: {self.grid_step}px")
            
            elif key == ord('-') or key == ord('_'):
                if self.grid_step < 60:
                    self.grid_step += 4
                    print(f"→ Grid: {self.grid_step}px")
            
            # Quit
            elif key == ord('q'):
                break
            
            self.frame_count += 1
        
        # Cleanup
        self.running = False
        flow_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Stopped (avg {self.fps} FPS)\n")


if __name__ == "__main__":
    tracker = OptimizedOpticalFlow(grid_step=20, preset="balanced")
    tracker.run()