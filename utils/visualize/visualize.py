import time
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import mujoco
import numpy as np
# import cv2
import os
import glfw 
from scipy.spatial.transform import Rotation as R

Array = npt.NDArray[np.float64]

class MjVisualizer():
    def __init__(self, xml_path: str, close_on_enter: bool = True):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.Nq = self.mj_model.nq
        self.step = 0
        self.paused = {"active": False}  # use dict so closure can mutate
        self.step_request = {"delta": 0}
        self.clean_request = {"active": False}
        self.exit_request = {"active": False}
        self.next_file_batch0_request = {"active": False}
        self.close_on_enter = close_on_enter

        def key_callback(keycode: int):
            # Space bar
            if keycode == 32:  # space
                self.paused["active"] = not self.paused["active"]
            # Left arrow
            elif keycode == 263:
                self.step_request["delta"] = -1
                self.paused["active"] = True
            # Right arrow
            elif keycode == 262:
                self.step_request["delta"] = 1
                self.paused["active"] = True
            # Escape
            elif keycode == 256:
                self.exit_request["active"] = True
            # 'C' for clean
            elif keycode == 67:  
                self.clean_request["active"] = True
            # 'N': jump to next file, batch 0 (consumed by caller)
            elif keycode == glfw.KEY_N:
                self.next_file_batch0_request["active"] = True
                self.exit_request["active"] = True

        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=key_callback)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def update_data(self, q:np.ndarray, v: np.ndarray = None):
        self.mj_data.qpos[:] = q
        if v is not None and self.mj_model.nv > 0:
            self.mj_data.qvel[:] = v
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.viewer.sync()

    def render_trajectory_to_video(
        self,
        t: np.ndarray,
        x_traj: np.ndarray,
        save_path: str,
        fps: int = 100,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        """
        Render a trajectory using mujoco.Renderer and save it as a video file.

        Args:
            t: (T,) time array
            x_traj: (T, state_dim) trajectory array [qpos, qvel]
            save_path: Path to save the video (e.g., 'results/output.mp4')
            fps: Frames per second for video
            width: Render width in pixels
            height: Render height in pixels
        """
        try:
            import imageio
        except ImportError:
            print("ERROR: imageio not installed. Install with: pip install imageio imageio-ffmpeg")
            return

        # Create output directory if needed
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        # Determine actual fps from data
        if len(t) > 1:
            dt = t[1] - t[0]
            actual_fps = min(fps, int(1.0 / dt))
        else:
            actual_fps = fps

        # Initialize renderer
        renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
        
        nq = self.mj_model.nq
        nv = self.mj_model.nv
        frames = []
        
        print(f"Rendering trajectory to video: {save_path}")
        print(f"  Resolution: {width}x{height}, FPS: {actual_fps}, Frames: {len(x_traj)}")

        for i, timestep in enumerate(t):
            # Reset and set state
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            self.mj_data.qpos[:] = x_traj[i, :nq]
            if x_traj.shape[1] > nq and nv > 0:
                self.mj_data.qvel[:] = x_traj[i, nq:nq + nv]
            
            mujoco.mj_forward(self.mj_model, self.mj_data)

            # Render frame
            renderer.update_scene(self.mj_data, camera="track")
            frame = renderer.render()
            frames.append(frame)

            if (i + 1) % max(1, len(x_traj) // 10) == 0:
                print(f"  Rendered {i + 1}/{len(x_traj)} frames...")

        # Save video using imageio
        imageio.mimwrite(save_path, frames, fps=actual_fps)
        renderer.close()
        print(f"Video saved to {save_path}")

    def visualize_trajectory(
        self,
        t: np.ndarray,
        x_traj: np.ndarray,
        repeat: bool = True,
        guidance_vec: np.ndarray = None,
        goal_pos: np.ndarray = None, 
        goal_quat: np.ndarray = None,
        overlay_paths: list = None,
        markers: list = None,
        body_keypoint_indices: list = None,
    ) -> None:
        """
        Visualizes the trajectory in a Mujoco viewer with pause and step-by-step control.
        - Space: toggle pause
        - Right arrow: step forward
        - Left arrow: step backward
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self.key_callback)

        # Get goal marker mocap id if goals provided
        goal_mocap_id = -1
        if goal_pos is not None:
             try:
                goal_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
                if goal_body_id >= 0:
                    goal_mocap_id = self.mj_model.body_mocapid[goal_body_id]
                    # Find associated geom
                    if self.mj_model.body_geomnum[goal_body_id] > 0:
                        goal_geom_id = self.mj_model.body_geomadr[goal_body_id]
             except Exception as e:
                 print(f"Warning: Could not find 'goal_marker' body for visualization: {e}")

        # Get guidance arrow mocap id
        arrow_mocap_id = -1
        if guidance_vec is not None:
            if guidance_vec.shape[-1] == 2:
                guidance_vec = np.pad(guidance_vec, [(0,0), (0,1)], mode='constant')  # add z=0 for 2D guidance
            try:
                arrow_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "guidance_arrow")
                if arrow_body_id >= 0:
                    arrow_mocap_id = self.mj_model.body_mocapid[arrow_body_id]
            except Exception as e:
                print(f"Warning: Could not find 'guidance_arrow' body for visualization: {e}")

        T = len(x_traj)
        PAUSE_LOOP = 0.5
        dt_array = np.diff(t, append=0.)
        dt_array[-1] = PAUSE_LOOP  # pause at the end

        while (self.step < T):
            if self.exit_request["active"]:
                self.exit_request["active"] = False
                break

            q, v = np.split(x_traj[self.step], [self.Nq])
            self.mj_data.qpos = q
            if x_traj.shape[1] > self.Nq:
                self.mj_data.qvel = v
            
            if goal_mocap_id >= 0:
                if goal_pos is not None:
                     gp = goal_pos[:, self.step] if goal_pos.ndim > 2 else goal_pos
                     self.mj_data.mocap_pos[goal_mocap_id] = gp
                if goal_quat is not None:
                     gq = goal_quat[:, self.step] if goal_quat.ndim > 2 else goal_quat
                     self.mj_data.mocap_quat[goal_mocap_id] = gq

            if arrow_mocap_id >= 0 and guidance_vec is not None:
                # Assuming object position is at indices 36:39 (G1 robot has 36 DoF including base)
                # q layout: [robot (36), object (7)]
                obj_pos = q[36:39]
                
                direction = guidance_vec[self.step] if guidance_vec.ndim > 1 else guidance_vec
                norm = np.linalg.norm(direction)
                
                if norm > 1e-6:
                    z_axis = np.array([0, 0, 1])
                    target_dir = direction / norm
                    
                    cross = np.cross(z_axis, target_dir)
                    dot = np.dot(z_axis, target_dir)
                    
                    if np.linalg.norm(cross) < 1e-6:
                        if dot > 0:
                            quat = [1, 0, 0, 0] # Identity
                        else:
                            quat = [0, 1, 0, 0] # 180 deg X
                    else:
                        angle = np.arccos(np.clip(dot, -1.0, 1.0))
                        axis = cross / np.linalg.norm(cross)
                        quat = R.from_rotvec(axis * angle).as_quat()
                        quat = quat[[3, 0, 1, 2]] # xyzw -> wxyz
                        
                    self.mj_data.mocap_pos[arrow_mocap_id] = obj_pos
                    self.mj_data.mocap_quat[arrow_mocap_id] = quat

            # Overlay paths
            if overlay_paths:
                for path in overlay_paths:
                    if len(path) < 2: continue
                    # Subsample for performance
                    step_size = 5
                    for j in range(0, len(path)-1, step_size):
                        if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom: break
                        
                        p1 = path[j]
                        p2 = path[min(j+step_size, len(path)-1)]
                        
                        mujoco.mjv_initGeom(
                            self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            size=[0.005, 0, 0],
                            pos=(p1+p2)/2,
                            mat=np.eye(3).flatten(),
                            rgba=[0, 1, 0, 0.15]
                        )
                        mujoco.mjv_connector(
                            self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                            0.005,
                            p1,
                            p2
                        )
                        self.viewer.user_scn.ngeom += 1

            # Markers (dynamic points at current frame)
            if markers:
                for marker_traj in markers:
                    # marker_traj: (T, 3) or just (3,) if static? 
                    # Assuming (T, 3) for now based on request "at each frame"
                    if len(marker_traj) > self.step:
                        pos = marker_traj[self.step]
                        if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                            mujoco.mjv_initGeom(
                                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.02, 0, 0],
                                pos=pos,
                                mat=np.eye(3).flatten(),
                                rgba=[1, 0, 0, 0.5] # Red
                            )
                            self.viewer.user_scn.ngeom += 1

            mujoco.mj_forward(self.mj_model, self.mj_data)

            # Body keypoint spheres (similar to play.py reference body visualization)
            if body_keypoint_indices:
                for body_idx in body_keypoint_indices:
                    if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                        break
                    pos = self.mj_data.xpos[body_idx].copy()
                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.04, 0, 0], dtype=np.float64),
                        pos=pos,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1], dtype=np.float32),
                    )
                    self.viewer.user_scn.ngeom += 1

            # print(self.bodies_in_contact("left_wrist_pitch_link", "largebox"))
            self.viewer.sync()

            time.sleep(dt_array[self.step])

            if self.paused["active"]:
                # Step manually only when requested
                if self.step_request["delta"] != 0:
                    self.step = (self.step + self.step_request["delta"]) % T
                    self.step_request["delta"] = 0
                else:
                    time.sleep(0.05)
                    continue
            else:
                # Play mode
                if repeat:
                    if self.step == T-1:
                        time.sleep(0.5)
                    self.step = (self.step + 1) % T
                else:
                    self.step = (self.step + 1)
            
        self.step = 0

        if self.close_on_enter:
            self.viewer.close()
                    

    # def render_and_save_trajectory(
    #     self,
    #     t: np.ndarray,
    #     x_traj: np.ndarray,
    #     save_path: str,
    #     fps: int = 30,
    #     width: int = 640,
    #     height: int = 480,
    # ):
    #     """
    #     Render a trajectory in MuJoCo using mujoco.Renderer and save it as a video.

    #     Args:
    #         mj_model: MuJoCo model.
    #         mj_data: MuJoCo data (will be modified during rendering).
    #         t: (T,) time array.
    #         x_traj: (T, n_state) array of states [qpos, qvel].
    #         save_video_path: Path to save the video (e.g., 'videos/output.mp4').
    #         fps: Frames per second for video.
    #         width: Render width.
    #         height: Render height.
    #         camera: Optional camera name (string) from the MJCF model.
    #     """
    #     # Initialize renderer
    #     renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
    #     dt = t[1] - t[0]
    #     fps = min(fps, int(1/dt))

    #     # Check save path
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     if os.path.isdir(save_path):
    #         filename = "trajectory_vis.mp4"
    #         save_path = os.path.join(save_path, filename)

    #     # Prepare video writer
    #     writer = cv2.VideoWriter(
    #         save_path,
    #         cv2.VideoWriter_fourcc(*"avc1"),
    #         fps,
    #         (width, height),
    #     )

    #     nq = self.mj_model.nq
    #     nv = self.mj_model.nv
    #     n_frames = 0
        
    #     for i, timestep in enumerate(np.squeeze(t)):
    #         # Set MuJoCo state
    #         mujoco.mj_resetData(self.mj_model, self.mj_data)
    #         self.mj_data.qpos[:] = x_traj[i, :nq]
    #         self.mj_data.qvel[:] = x_traj[i, nq:nq + nv]
    #         mujoco.mj_forward(self.mj_model, self.mj_data)

    #         # Render frame
    #         if n_frames < (timestep * fps):
    #             n_frames += 1
    #             renderer.update_scene(self.mj_data, camera="track")
    #             frame = renderer.render()
    #             # Convert RGB to BGR for OpenCV and write
    #             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             writer.write(frame_bgr)

    #     writer.release()
    #     renderer.close()
    #     print(f"Saved video to {save_path}")


import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R


class DiffusionOverlayVisualizer:
    def __init__(self, xml_path: str):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

    def _set_state(self, qpos, qvel=None):
        self.mj_data.qpos[:] = qpos
        if qvel is not None and self.mj_model.nv > 0:
            self.mj_data.qvel[:] = qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _add_ghost_robot(self, scene, qpos, color, alpha=0.25):
        """
        Add a simplified ghost robot using mjvGeom.
        We draw boxes at body positions (fast & robust).
        """
        mujoco.mj_forward(self.mj_model, self.mj_data)

        for body_id in range(1, self.mj_model.nbody):  # skip world
            geom_id = self.mj_model.body_geomadr[body_id]
            if geom_id < 0:
                continue

            pos = self.mj_data.xpos[body_id]
            mat = self.mj_data.xmat[body_id].reshape(3, 3)

            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.04, 0.04, 0.04]),  # uniform ghost size
                pos=pos,
                mat=mat.flatten(),
                rgba=np.array([*color, alpha]),
            )
            scene.ngeom += 1

    def visualize_overlay(
        self,
        x_trajs: np.ndarray,
        repeated_xml_path: str = "./unitree_g1/mj_model_repeated.xml",
        timestep_delay: float = 0.05,
        loop: bool = True,
        guidance_vec: np.ndarray = None,
        world_goal_pos: np.ndarray = None, 
        world_goal_quat: np.ndarray = None,
        spacing: float = 1,
        ref_start_idx: int = -1,
    ):
        """
        Visualizes multiple trajectories simultaneously using a repeated-robot XML scene.
        """
        if not os.path.exists(repeated_xml_path):
            print(f"Error: Repeated XML not found at {repeated_xml_path}")
            return

        print(f"Loading multis-scene XML from {repeated_xml_path}")
        mj_model_rep = mujoco.MjModel.from_xml_path(repeated_xml_path)
        mj_data_rep = mujoco.MjData(mj_model_rep)
        
        B, T_len, D = x_trajs.shape
        nq_single = self.mj_model.nq
        
        if mj_model_rep.nq < B * nq_single:
            print(f"Warning: Repeated XML nq ({mj_model_rep.nq}) < Batch ({B}) * Single nq ({nq_single}). Truncating batch view.")
            num_to_show = mj_model_rep.nq // nq_single
        else:
            num_to_show = B
            
        print(f"Visualizing {num_to_show} trajectories out of {B}...")
        
        # Verify shapes before loop to prevent segfaults
        print(f"Debug: nq_single={nq_single}, nq_rep={mj_model_rep.nq}, x_trajs.shape={x_trajs.shape}")
        
        if num_to_show > 0:
             expected_fill = num_to_show * nq_single
             if expected_fill > mj_model_rep.nq:
                 print(f"Critical Error: Attempting to fill {expected_fill} qpos indices but repeated model only has {mj_model_rep.nq}.")
                 return

        # Initialize Mocap IDs
        goal_mocap_ids = []
        arrow_mocap_ids = []
        for i in range(num_to_show):
            # Goal
            name_g = f"goal_marker_{i}"
            bid_g = mujoco.mj_name2id(mj_model_rep, mujoco.mjtObj.mjOBJ_BODY, name_g)
            if bid_g >= 0:
                goal_mocap_ids.append(mj_model_rep.body_mocapid[bid_g])
            else:
                goal_mocap_ids.append(-1)
            
            # Arrow
            name_a = f"guidance_arrow_{i}"
            bid_a = mujoco.mj_name2id(mj_model_rep, mujoco.mjtObj.mjOBJ_BODY, name_a)
            if bid_a >= 0:
                arrow_mocap_ids.append(mj_model_rep.body_mocapid[bid_a])
            else:
                arrow_mocap_ids.append(-1)

        # ── Recolour reference robots so they stand out ───────────────────────
        # Robots with index >= ref_start_idx are ground-truth; colour them
        # sky-blue so they are visually distinct from the generated trajectories.
        # We must also clear the material assignment (geom_matid = -1) because
        # MuJoCo prioritises material rgba over geom_rgba.
        if ref_start_idx >= 0:
            _ref_rgba = np.array([0.25, 0.75, 0.95, 0.85], dtype=np.float32)
            _recoloured = 0
            for _gid in range(mj_model_rep.ngeom):
                _bid   = mj_model_rep.geom_bodyid[_gid]
                _bname = mujoco.mj_id2name(
                    mj_model_rep, mujoco.mjtObj.mjOBJ_BODY, _bid
                ) or ""
                # Body names end with _<robot_index>, e.g. "pelvis_3"
                _parts = _bname.rsplit('_', 1)
                if (len(_parts) == 2 and _parts[1].isdigit()
                        and int(_parts[1]) >= ref_start_idx):
                    mj_model_rep.geom_rgba[_gid]  = _ref_rgba
                    mj_model_rep.geom_matid[_gid] = -1   # disable material so rgba is used
                    _recoloured += 1
            print(f"Recoloured {_recoloured} geoms for reference robots.")
            print(f"Colour legend:  robots 0–{ref_start_idx-1} = generated (default)  "
                  f"| robots {ref_start_idx}–{num_to_show-1} = ground-truth (sky-blue)")

        # ── Playback state (mutable dicts so the key callback closure can write) ──
        _paused   = {"active": False}
        _step_req = {"delta": 0}
        _restart  = {"active": False}

        def _key_callback(keycode: int):
            if keycode == 32:       # Space  → toggle pause
                _paused["active"] = not _paused["active"]
            elif keycode == 262:    # Right  → step forward  (auto-pauses)
                _step_req["delta"] = 1
                _paused["active"] = True
            elif keycode == 263:    # Left   → step backward (auto-pauses)
                _step_req["delta"] = -1
                _paused["active"] = True
            elif keycode == 82:     # 'R'    → restart
                _restart["active"] = True

        with mujoco.viewer.launch_passive(mj_model_rep, mj_data_rep,
                                          key_callback=_key_callback) as viewer:

            # ── Camera setup ─────────────────────────────────────────────
            # All robots are overlaid at the origin (spacing=0), so aim at
            # the approximate whole-body COM and pull back far enough to
            # capture the full motion range.
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            viewer.cam.distance  = 5.0
            viewer.cam.azimuth   = 135.0   # front-right 3/4 view
            viewer.cam.elevation = -30.0   # slight top-down
            # ─────────────────────────────────────────────────────────────

            step = 0
            print("Controls:  Space=pause  \u2190/\u2192=step  R=restart")
            while viewer.is_running():

                # ── Handle restart request ────────────────────────────────
                if _restart["active"]:
                    step = 0
                    _restart["active"] = False

                n_cols = int(np.ceil(np.sqrt(num_to_show)))
                
                # 1. Update Robot/Object States using OFFSETS
                current_q = x_trajs[:num_to_show, step, :nq_single].copy()
                
                for i in range(num_to_show):
                    row = i // n_cols
                    col = i % n_cols
                    x_offset = row * spacing
                    y_offset = col * spacing
                    
                    # Offset Robot Root (free joint at 0:3)
                    if nq_single >= 3:
                        current_q[i, 0] += x_offset
                        current_q[i, 1] += y_offset
                    
                    # Offset Object Root (free joint at 36:39)
                    if nq_single >= 39:
                        current_q[i, 36] += x_offset
                        current_q[i, 37] += y_offset

                # Flatten -> (num_to_show * nq_single)
                flat_q = current_q.reshape(-1)
                len_fill = len(flat_q)
                mj_data_rep.qpos[:len_fill] = flat_q
                
                # 2. Update Mocap Bodies (Goals & Arrows)
                for i in range(num_to_show):
                    row = i // n_cols
                    col = i % n_cols
                    x_offset = row * spacing
                    y_offset = col * spacing
                    
                    # Skip goal markers and arrows for reference (ground-truth) robots
                    _is_ref = (ref_start_idx >= 0 and i >= ref_start_idx)

                    # --- Goal Marker ---
                    mid_g = goal_mocap_ids[i]
                    if mid_g >= 0 and not _is_ref:
                        if world_goal_pos is not None:
                            # Support (B, 3) or (3,)
                            gp = world_goal_pos[i] if (world_goal_pos.ndim > 1 and i < len(world_goal_pos)) else world_goal_pos.flatten()
                            # Apply offset for visualization
                            final_gp = gp.copy()
                            final_gp[0] += x_offset
                            final_gp[1] += y_offset
                            mj_data_rep.mocap_pos[mid_g] = final_gp
                            
                        if world_goal_quat is not None:
                            gq = world_goal_quat[i] if (world_goal_quat.ndim > 1 and i < len(world_goal_quat)) else world_goal_quat.flatten()
                            mj_data_rep.mocap_quat[mid_g] = gq
                            
                    # --- Arrow ---
                    mid_a = arrow_mocap_ids[i]
                    if mid_a >= 0 and guidance_vec is not None and not _is_ref:
                         # Get object pos from modified qpos
                         if nq_single >= 39:
                             # 36:39 is typically object root pos
                             obj_pos = current_q[i, 36:39]
                             
                             # Get direction
                             # guidance_vec is (B, T, 3) usually
                             direction = guidance_vec[i, step] if guidance_vec.ndim == 3 else guidance_vec[step]
                             norm = np.linalg.norm(direction)
                             
                             if norm > 1e-6:
                                z_axis = np.array([0, 0, 1])
                                target_dir = direction / norm
                                cross = np.cross(z_axis, target_dir)
                                dot = np.dot(z_axis, target_dir)
                                
                                if np.linalg.norm(cross) < 1e-6:
                                    if dot > 0:
                                        quat = [1, 0, 0, 0]
                                    else:
                                        quat = [0, 1, 0, 0]
                                else:
                                    angle = np.arccos(np.clip(dot, -1.0, 1.0))
                                    axis = cross / np.linalg.norm(cross)
                                    quat = R.from_rotvec(axis * angle).as_quat()
                                    quat = quat[[3, 0, 1, 2]] # xyzw -> wxyz
                                    
                                mj_data_rep.mocap_pos[mid_a] = obj_pos
                                mj_data_rep.mocap_quat[mid_a] = quat

                mujoco.mj_forward(mj_model_rep, mj_data_rep)
                viewer.sync()
                
                time.sleep(timestep_delay)

                # ── Frame advance ─────────────────────────────────────────
                if _paused["active"]:
                    if _step_req["delta"] != 0:
                        step = (step + _step_req["delta"]) % T_len
                        _step_req["delta"] = 0
                    else:
                        time.sleep(0.05)
                        continue
                else:
                    step += 1
                    if step >= T_len:
                        if loop:
                            step = 0
                            time.sleep(0.5)
                        else:
                            step = T_len - 1
                            _paused["active"] = True
