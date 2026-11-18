# System Specification: Latency-Compensated Asynchronous 3D Perception & Grasp Node (`async-perception-ros2`)

**Status:** Locked for Implementation (v1.0)  
**Timeline:** 4 Weeks  
**Target Roles:** Robotics Systems Researcher, Robot Learning / VLA ML Engineer, Embodied AI Systems Engineer  
**Development Environments:**
- **Local Edge (System 1):** Apple M1 MacBook Air (macOS ARM64, no CUDA) running ROS 2 via RoboStack (`conda-forge` / `robostack-staging`), portable to Ubuntu 22.04/24.04 (`x86_64`). IDE: Google Antigravity.
- **Cloud Inference (System 2):** Google Colab GPU (T4/L4) running Grounded-SAM 2 (`GroundingDINO` + `SAM 2.1`) exposed via an HTTP/REST tunnel (`cloudflared`) with $0.5\text{--}2.0\text{ s}$ WAN Round-Trip Time (RTT).
- **CI/CD & Verification:** GitHub Actions (CPU runners for deterministic ROS 2 integration tests & latency benchmarks on every push; Docker Buildx with GHA caching for CUDA server image verification).

---

## 1. Executive Summary & Core Task Definition

### 1.1 Locked Task Definition
Build a non-blocking, split-compute ROS 2 perception-to-grasping pipeline that dispatches throttled RGB frames ($1\text{--}2\text{ Hz}$) to a remote Grounded-SAM 2 server while locally maintaining a $30\text{ Hz}$ synchronized RGB-D ring buffer and `tf2` historical transform buffer to back-project delayed 2D segmentation masks into time-accurate, outlier-filtered 3D object point clouds and 6-DoF antipodal grasp poses in the `world` frame—benchmarked head-to-head against a Naive Asynchronous baseline across swept network latencies.

### 1.2 The Systems Wedge (Differentiation Against Prior Art)
Existing open-source ROS 2 + Grounded-SAM repositories treat foundation model inference as a synchronous, static-world operation:
1. **`rai-perception` (RobotecAI):** Wraps GroundingDINO and Grounded-SAM 2 as blocking request-response ROS 2 services (`/detection`, `/segmentation`) designed for slow conversational agent tool calls (`GetObjectPositionsTool`, `GetObjectGrippingPointsTool`) using a `single_threaded` connector. During the $1\text{--}2\text{ s}$ inference call, the pipeline is blind to camera or robot motion.
2. **`grounded_sam_vgn_ros2_pipeline`:** Defaults to `process_once=true` (unsubscribes after the first frame) and notes $30\text{--}40\text{ s}$ CPU execution times, assuming a completely static Gazebo scene between image capture and grasp synthesis.

**Our Architectural Wedge — Solving the System-2 / System-1 Temporal Mismatch:**  
In modern VLA and foundation-model robotics, heavy vision-language models run at $0.5\text{--}2\text{ Hz}$ (off-board or on secondary compute) while robot state estimation and control run at $30\text{--}500\text{ Hz}$. When a remote server returns a 2D segmentation mask after $\Delta t \in [0.5, 2.0]\text{ s}$, that mask corresponds to sensor capture timestamp $t_0$. Applying a mask from $t_0$ to the current depth frame and camera pose at $t_{\text{now}} = t_0 + \Delta t$ while the camera is moving projects background walls into the target object cloud and induces $50\text{--}150\text{ mm}$ 3D pose drift.

This repository solves that exact failure mode by pairing **asynchronous execution** with **temporal retro-projection**:
$$\mathbf{P}_{\text{world}}(t_0) = \mathbf{T}_{\text{world} \leftarrow \text{cam}}(t_0) \, \pi^{-1}\!\left(\mathcal{E}\!\left(\mathbf{M}_{2\text{D}}(t_0)\right), \, \mathbf{D}(t_0), \, \mathbf{K}\right)$$
where $\mathbf{M}_{2\text{D}}(t_0)$ is the delayed mask returned at $t_{\text{now}}$, $\mathcal{E}(\cdot)$ is morphological boundary erosion, $\mathbf{D}(t_0)$ is the historical depth frame retrieved from a local lock-free/thread-safe ring buffer, and $\mathbf{T}_{\text{world} \leftarrow \text{cam}}(t_0)$ is queried from the `tf2` buffer at exact capture time $t_0$.

---

## 2. Scope Decisions & Engineering Rationale

| Decision Area | Decision | Engineering Rationale |
| :--- | :--- | :--- |
| **Temporal Compensation (Core)** | Historical `tf2` lookup (`t_0`) + Timestamped RGB-D Ring Buffer | Mathematically eliminates ego-motion drift for moving cameras (wrist-mounted or mobile manipulator views) over static tabletop objects without introducing tracker drift. |
| **30 Hz Local Object Tracker** | **Moved to Stretch Goal** (Week 4 optional) | `tf2` time-travel + ring buffer completely handles the moving-camera demo and benchmark scenario. Deferring optical-flow/Kalman object tracking protects the 4-week schedule from scope creep. |
| **Evaluation Harness** | **Promoted to Core Scope** | Senior interviewers respect quantitative proof. A side-by-side ROS 2 evaluation harness comparing `compensated` vs. `naive_latest_frame` across swept latencies ($\text{RTT} \in [100, 2000]\text{ ms}$) turns a demo into a systems paper artifact. |
| **Grasp Synthesis Method** | Local 3D Point Cloud Filtering (Erosion + SOR) + Analytic PCA Antipodal Grasp Pose | **Cut AnyGrasp:** AnyGrasp requires machine-locked proprietary licenses, closed-source binaries, and `MinkowskiEngine` source compilation ($20+\text{ min}$ build times that frequently break on Colab CUDA upgrades). Analytic PCA + antipodal synthesis on clean, eroded 3D clouds is deterministic, dependency-free, runs in $<3\text{ ms}$ on M1 CPU, and isolates the perception/latency metric cleanly. |
| **WAN Payload Protocol** | Send JPEG RGB ($\sim 40\text{ KB}$) + Prompt; Return COCO RLE Mask ($\sim 5\text{ KB}$) | **Never send raw Depth over WAN:** Raw $640\times480$ RGB-D is $\sim 1.5\text{ MB}$ per frame, which inflates Colab tunnel latency to $3\text{--}5\text{ s}$. Depth remains $100\%$ local on the M1 Mac inside the ring buffer. |
| **CI/CD Architecture** | Split CPU Edge CI (every push) from CUDA Server Image Build (cached/tagged) | Compiling `GroundingDINO` custom C++ CUDA ops (`DeformableAttention`) + PyTorch inside standard GitHub Actions runners hits the $14\text{ GB}$ disk limit and lacks a live GPU (`nvcc` requires cross-arch flags). Testing the ROS 2 node against a deterministic Mock Latency Server on CPU CI guarantees fast ($<4\text{ min}$), reliable PR checks. |

---

## 3. Anti-Toy Engineering Requirements (Pre-empting Senior Engineer Critique)

1. **Zero Executor Starvation:** The ROS 2 node must use `rclpy.executors.MultiThreadedExecutor` with separated `MutuallyExclusiveCallbackGroup` instances so the $30\text{ Hz}$ RGB-D synchronizer and telemetry publishers never block while waiting on a $0.5\text{--}2.0\text{ s}$ HTTP response.
2. **Boundary Flying-Pixel Suppression:** Raw 2D segmentation masks include $1\text{--}2\text{ px}$ boundary bleed onto background surfaces, creating severe depth veil ("flying pixels") that corrupt 3D centroids by $10\text{--}30\text{ cm}$. The pipeline mandates **$3\times3$ morphological mask erosion** in 2D prior to depth indexing, followed by **3D Statistical Outlier Removal (SOR)** or median depth cluster filtering.
3. **Strict Sensor Timestamp Discipline:** Zero usage of `rclpy.clock.Clock().now()` for geometric transforms. All `tf2` lookups and ring buffer queries must key strictly off `sensor_msgs/Image.header.stamp` ($t_0$).
4. **Graceful Degradation & Circuit Breaker:** Cloud tunnels drop and rate-limit. The node implements an explicit state machine (`ACTIVE` $\rightarrow$ `DEGRADED_STALE` $\rightarrow$ `CIRCUIT_OPEN`) governed by a configurable mask Time-To-Live (TTL, default $3.0\text{ s}$) and consecutive failure counter. When degraded or open, confidence decays exponentially and `/diagnostics` alerts downstream controllers rather than silently publishing stale poses.
5. **Explicit QoS Profiles:** Sensor subscriptions (`RGB`, `Depth`, `CameraInfo`) use `qos_profile_sensor_data` (`BestEffort`, `Volatile`, `KeepLast(5)`) to prevent backpressure stalls, while output grasp poses and evaluation topics use `Reliable`, `Volatile`, `KeepLast(10)`.

---

## 4. Dataset Selection & `rosbag2` Conversion Strategy (Day 1–2 Task)

### 4.1 Primary Choice: BOP Benchmark `ycbv` (YCB-Video) Archive
Instead of downloading the full $100+\text{ GB}$ original YCB-Video dataset or converting legacy ROS 1 `.bag` files, we use the standardized **BOP Benchmark `ycbv` archives** hosted on the official BOP HuggingFace repository (`bop_datasets/ycbv`).
- **Scenario Fit:** A handheld/robot-mounted RGB-D camera orbits a static cluttered tabletop containing standard YCB manipulation objects (`002_master_chef_can`, `003_cracker_box`, `006_mustard_bottle`, `021_bleach_cleanser`) at $640\times480$.
- **Why it is ideal for our benchmark:** Because the camera moves continuously while the objects remain static in the `world` frame, ground-truth 3D object centroids in `world` are constant, while camera-frame coordinates change rapidly. Under a $1.0\text{ s}$ inference delay, the **Naive Async** baseline exhibits massive 3D world-frame drift proportional to camera velocity, whereas our **Latency-Compensated** pipeline stays locked on the ground-truth 3D pose.

### 4.2 Ground-Truth Pose Verification & Train/Test Split Fallback Rule
BOP dataset archives package files per sequence folder (`000048/`, `000050/`, etc.) containing:
- `rgb/*.png` ($640\times480$ RGB)
- `depth/*.png` ($16\text{-bit}$ millimeter depth)
- `scene_camera.json` (per-frame camera intrinsics $\mathbf{K}$ and extrinsics $\mathbf{R}_{c\leftarrow w}, \mathbf{t}_{c\leftarrow w}$)
- `scene_gt.json` (per-frame 6-DoF ground-truth object poses $\mathbf{R}_{c\leftarrow o}, \mathbf{t}_{c\leftarrow o}$ and object IDs)

**Day 1 Verification & Fallback Protocol:**
1. First inspect `ycbv_test_bop19.zip` ($\sim 600\text{ MB}$, sequences `000048`–`000059`). Note that on the HuggingFace BOP mirror (`https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/`), `ycbv_test_bop19.zip` includes `scene_gt.json` for YCB-V; however, certain older BOP challenge test mirrors strip `scene_gt.json` from test archives to prevent challenge cheating.
2. **Immediate Fallback:** If the downloaded test archive lacks `scene_gt.json`, the download script (`datasets/download_bop_ycbv_sample.sh`) will either:
   - Fetch the standalone ground-truth annotations archive (`ycbv_test_gt.zip` / `scene_gt.json` from the BOP HuggingFace repo), **OR**
   - Fall back to downloading a single validation/training sequence from **`ycbv_train_real.zip`** (via HTTP Range / selective extraction or HuggingFace per-folder download of sequence `000000`), which is guaranteed to include `scene_gt.json`, `scene_gt_info.json`, and `mask_visib/`.
3. **CI Micro-Sequence:** Extract a 60-frame ($2.0\text{ s}$ at $30\text{ Hz}$) slice with pre-computed Grounded-SAM 2 RLE responses into `datasets/sample_seq_ci/` ($\sim 12\text{ MB}$) so GitHub Actions can run full end-to-end latency compensation tests without external downloads.

### 4.3 `rosbag2` (`.mcap`) Conversion (`datasets/bopycb_to_rosbag2.py`)
Converts a BOP sequence folder into a deterministic ROS 2 `.mcap` bag file containing:
- `/camera/color/image_raw` (`sensor_msgs/msg/Image`, `rgb8`)
- `/camera/depth/image_rect_raw` (`sensor_msgs/msg/Image`, `16UC1`, depth in mm)
- `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`)
- `/tf` (`tf2_msgs/msg/TFMessage`: dynamic transform `world` $\rightarrow$ `camera_color_optical_frame` inverted from `scene_camera.json`)
- `/eval/ground_truth_poses` (`async_perception_msgs/msg/SegmentedObject3D`: exact 3D ground-truth object pose and centroid transformed into `world` frame from `scene_gt.json`)

---

## 5. System Architecture & Dataflow

```text
+---------------------------------------------------------------------------------------------------+
| LOCAL EDGE NODE (Apple M1 Mac / RoboStack ROS 2 OR Ubuntu x86_64 Docker)                          |
|                                                                                                   |
|  [rosbag2 play (30 Hz)]                                                                           |
|     ├── /camera/color/image_raw ──────┐                                                           |
|     ├── /camera/depth/image_rect_raw ─┼──► [ApproxTimeSynchronizer (CallbackGroup A: 30 Hz)]      |
|     ├── /camera/camera_info ──────────┘           │                                               |
|     └── /tf (world -> cam_optical) ───────┐       ├──► 1. Push (t_k, RGB, Depth, K) to RingBuffer |
|                                           ▼       └──► 2. Update PerceptionTelemetry (10 Hz)      |
|                                    [tf2_ros.Buffer]                                               |
|                                     (10s cache)   [Async Inference Timer (CallbackGroup B: 1 Hz)] |
|                                           │               │                                       |
|                                           │               ├──► Grab latest (t_0, RGB)             |
|                                           │               ├──► JPEG encode (~40 KB)               |
|                                           │               ▼                                       |
|                                           │        (Non-blocking HTTP POST + Circuit Breaker)     |
|                                           │               │                                       |
+-------------------------------------------|---------------|---------------------------------------+
                                            |               | WAN Tunnel (cloudflared / mock)
                                            |               | RTT = 500 - 2000 ms
+-------------------------------------------|---------------|---------------------------------------+
| REMOTE CLOUD SERVER (Google Colab GPU / CUDA Docker)      ▼                                       |
|                                           │        [FastAPI /v1/segment]                          |
|                                           │          └──► GroundingDINO + SAM 2.1                 |
|                                           │          └──► Encode binary mask to COCO RLE (~5 KB)  |
|                                           │          └──► Return JSON with echoed frame_stamp_ns  |
+-------------------------------------------|---------------|---------------------------------------+
                                            |               |
+-------------------------------------------|---------------|---------------------------------------+
| LOCAL EDGE NODE (Response Handler at t_now = t_0 + RTT)   ▼                                       |
|                                           │        [3D Retro-Projection & Grasp Synthesis]        |
|  If mode == "compensated":                │          ├──► Decode RLE mask M(t_0)                  |
|    Depth(t_0), K ◄── RingBuffer.get(t_0)  │          ├──► 2D 3x3 Morphological Erosion            |
|    T_w_c(t_0)    ◄────────────────────────┘          ├──► Back-project to 3D cam frame at t_0     |
|  If mode == "naive_latest_frame":                    ├──► Statistical Outlier Removal (SOR)       |
|    Depth(t_now)  ◄── RingBuffer.latest()             ├──► Transform cloud to 'world' via T_w_c    |
|    T_w_c(t_now)  ◄── tf2.lookup(latest)              └──► Compute 3D Centroid + PCA 6-DoF Grasp   |
|                                                                 │                                 |
|                                                                 ├──► /perception/grasp_pose       |
|                                                                 ├──► /perception/target_cloud     |
|                                                                 └──► /perception/grasp_markers    |
|                                                                           │                       |
|  [Eval Evaluator Node] ◄── Compares /perception/grasp_pose vs. /eval/ground_truth_poses           |
+---------------------------------------------------------------------------------------------------+
```

---

## 6. Repository Directory Layout

```text
async-perception-ros2/
├── SPEC.md                           # This specification document
├── README.md                         # Portfolio landing page: architecture, GIFs, benchmark plots
├── .github/
│   └── workflows/
│       ├── ros2_ci.yml               # Runs on push/PR: lint, colcon build, unit tests, 60-frame latency eval
│       └── docker_cuda_build.yml     # Runs on release tag / workflow_dispatch: builds CUDA server image
├── docker/
│   ├── Dockerfile.ros2_edge          # Lightweight ROS 2 Humble/Jazzy CPU image for CI & Ubuntu portability
│   ├── Dockerfile.inference_server   # CUDA 12.1 image for Grounded-SAM 2 cloud server (headless build flags)
│   └── robostack_env.yml             # M1 Mac native osx-arm64 conda environment lockfile
├── datasets/
│   ├── download_bop_ycbv_sample.sh   # Downloads BOP ycbv sequence (with scene_gt.json check & train fallback)
│   ├── bopycb_to_rosbag2.py          # Converts PNGs + scene_camera.json + scene_gt.json -> .mcap rosbag2
│   └── sample_seq_ci/                # Committed 60-frame mini-sequence + cached RLE masks for offline CI
├── inference_server/                 # Pure Python/FastAPI (Zero ROS dependencies - runs on Colab or local CPU)
│   ├── requirements-server.txt
│   ├── schemas.py                    # Pydantic wire schemas (request headers + COCO RLE mask response)
│   ├── app.py                        # FastAPI server exposing POST /v1/segment and GET /v1/health
│   ├── gsam2_wrapper.py              # GroundingDINO (HF/local) + SAM 2.1 predictor wrapper
│   ├── mock_server.py                # Deterministic CPU mock server with configurable artificial RTT & jitter
│   └── colab_server_launcher.ipynb   # 1-click Colab notebook: clones repo, loads weights, starts cloudflared
├── src/
│   ├── async_perception_msgs/        # Custom ROS 2 interfaces package (ament_cmake)
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   └── msg/
│   │       ├── SegmentedObject3D.msg
│   │       ├── GraspCandidate6D.msg
│   │       └── PerceptionTelemetry.msg
│   └── async_gsam2_perception/       # Core ROS 2 perception & evaluation package (ament_python)
│       ├── package.xml
│       ├── setup.py
│       ├── config/
│       │   ├── perception_params.yaml # Mode (compensated vs naive), ring_buffer_size, erosion_px, TTL
│       │   └── rviz_demo.rviz        # Dual-display RViz config showing Naive (red) vs Compensated (green)
│       ├── launch/
│       │   ├── demo_pipeline.launch.py
│       │   └── eval_sweep.launch.py
│       ├── async_gsam2_perception/
│       │   ├── __init__.py
│       │   ├── perception_node.py    # MultiThreadedExecutor, ApproxTimeSync, CallbackGroups, State Machine
│       │   ├── ring_buffer.py        # Thread-safe timestamp-indexed RGB-D + CameraInfo circular buffer
│       │   ├── async_client.py       # Non-blocking HTTP client with circuit-breaker & RTT telemetry
│       │   ├── geometry_3d.py        # RLE decode, 2D erosion, depth projection, SOR, PCA antipodal grasp
│       │   └── eval_evaluator_node.py# Subscribes to grasp & GT poses, computes 3D error/IoU, exports CSV
│       └── test/
│           ├── test_ring_buffer.py
│           ├── test_geometry_3d.py
│           └── test_latency_compensation_integration.py
└── scripts/
    ├── run_rtt_sweep.sh              # Automates rosbag2 replay across RTT=[100..2000ms] for both modes
    └── plot_latency_benchmark.py     # Generates publication-grade README plots from eval CSVs
```

---

## 7. Wire & ROS 2 Message Interface Definitions

### 7.1 WAN HTTP Wire Protocol (`inference_server/schemas.py`)
Designed to minimize bandwidth over `cloudflared` tunnels while remaining stateless and self-identifying via `frame_stamp_ns`.

- **Endpoint `POST /v1/segment`** (`multipart/form-data` or JSON with base64 JPEG):
  - `frame_stamp_ns` (`int64`): Exact `header.stamp.sec * 10^9 + header.stamp.nanosec` of the source RGB frame ($t_0$).
  - `frame_id` (`string`): Optical frame ID (e.g., `"camera_color_optical_frame"`).
  - `text_prompt` (`string`): Grounding prompt formatted for GroundingDINO (e.g., `"mustard bottle."`).
  - `box_threshold` (`float`, default `0.35`): Detection box threshold.
  - `text_threshold` (`float`, default `0.25`): Text association threshold.
  - `image_jpeg` (`bytes`): JPEG-compressed RGB image (quality `85`, $\sim 35\text{--}50\text{ KB}$).

- **Response (`200 OK` `application/json`):**
  ```json
  {
    "frame_stamp_ns": 1710000123456789000,
    "frame_id": "camera_color_optical_frame",
    "server_inference_ms": 318.4,
    "img_height": 480,
    "img_width": 640,
    "detections": [
      {
        "class_name": "mustard bottle",
        "score": 0.91,
        "bbox_xyxy": [210.4, 140.2, 295.1, 310.8],
        "rle_mask": {
          "size": [480, 640],
          "counts": "eNq1VNuO2jAQ/ZVR..."
        }
      }
    ]
  }
  ```

### 7.2 Custom ROS 2 Message Definitions (`src/async_perception_msgs/msg/`)

#### `SegmentedObject3D.msg`
Used for both perceived 3D objects and ground-truth evaluation broadcasts (`/eval/ground_truth_poses`).
```text
std_msgs/Header header                        # frame_id = "world", stamp = t_published (or t_gt)
builtin_interfaces/Time source_sensor_stamp   # Exact t_0 of the RGB-D frame used for projection
string class_label                            # e.g., "006_mustard_bottle" or "mustard bottle"
float32 detection_score                       # Model confidence [0.0 - 1.0]
geometry_msgs/Point centroid_world            # 3D centroid of filtered point cloud in world frame (m)
geometry_msgs/Pose bounding_box_pose_world    # Oriented 3D bounding box center & orientation in world frame
geometry_msgs/Vector3 bounding_box_extent     # 3D box dimensions (x, y, z in meters)
uint32 inlier_point_count                     # Number of 3D points surviving erosion + outlier removal
```

#### `GraspCandidate6D.msg`
Primary output consumed by downstream motion planners or the benchmark evaluator (`/perception/grasp_pose`).
```text
std_msgs/Header header                        # frame_id = "world", stamp = t_published
builtin_interfaces/Time source_sensor_stamp   # Exact t_0 of the RGB-D frame used for 3D projection
string target_label                           # Target object prompt label
geometry_msgs/Pose grasp_pose_world           # 6-DoF antipodal gripper pose in world frame
geometry_msgs/Point object_centroid_world     # 3D centroid of segmented target in world frame
float32 grasp_width_m                         # Required parallel-jaw opening width along minor PCA axis (m)
float32 confidence                            # Time-decayed confidence score [0.0 - 1.0]
float32 end_to_end_latency_ms                 # (t_published - t_0) in milliseconds
bool is_latency_compensated                   # True = RingBuffer(t_0) + tf2(t_0); False = Naive(t_now)
```

#### `PerceptionTelemetry.msg`
Published continuously at $10\text{ Hz}$ on `/perception/telemetry` to prove deterministic local execution and fault handling.
```text
std_msgs/Header header
uint8 STATE_ACTIVE=0
uint8 STATE_DEGRADED_STALE=1
uint8 STATE_CIRCUIT_OPEN=2
uint8 system_state
float32 last_wan_rtt_ms                       # Latest HTTP round-trip time (ms)
float32 current_mask_age_ms                   # Age of the active segmentation mask (t_now - t_0)
float32 local_callback_exec_ms                # Execution time of the local 3D projection + grasp math (ms)
uint32 ring_buffer_depth                      # Current number of synchronized RGB-D frames buffered
uint32 consecutive_wan_failures               # Circuit-breaker failure counter
```

---

## 8. Quantitative Evaluation Protocol & Success Criteria

### 8.1 Baseline vs. Proposed Method
Both modes run inside the exact same node (`perception_node.py`) toggled via the ROS 2 parameter `compensation_mode`:
1. **Baseline (`naive_latest_frame`):** When the delayed RLE mask $\mathbf{M}_{2\text{D}}(t_0)$ arrives at $t_{\text{now}}$, it is back-projected using the *latest* depth image $\mathbf{D}(t_{\text{now}})$ and transformed into `world` using the *latest* transform $\mathbf{T}_{\text{world}\leftarrow\text{cam}}(t_{\text{now}})$.
2. **Proposed (`compensated`):** When $\mathbf{M}_{2\text{D}}(t_0)$ arrives at $t_{\text{now}}$, the node retrieves $\mathbf{D}(t_0)$ from `RingBuffer` (tolerance $\le 16.7\text{ ms}$) and looks up $\mathbf{T}_{\text{world}\leftarrow\text{cam}}(t_0)$ from `tf2_ros.Buffer`, projecting the eroded mask into `world` at exact capture time $t_0$.

### 8.2 Target Metrics (Numbers an Interviewer Respects)
Evaluated across swept simulated RTTs $\Delta t \in \{100, 250, 500, 1000, 1500, 2000\}\text{ ms}$ plus live Colab WAN runs on BOP YCB-Video sequences:
1. **3D Centroid & Grasp Position Error ($e_{\text{pos}}$ in mm):**
   $$e_{\text{pos}} = \left\| \hat{\mathbf{c}}_{\text{world}} - \mathbf{c}_{\text{world}}^{\text{GT}} \right\|_2$$
   - **Success Criterion:** At $\text{RTT} = 1000\text{ ms}$, `compensated` mode achieves **$\ge 60\%$ reduction in median 3D translation error** compared to `naive_latest_frame` (maintaining $e_{\text{pos}} < 20\text{ mm}$ across all latencies up to $2000\text{ ms}$, whereas `naive_latest_frame` degrades linearly with camera velocity to $> 60\text{--}120\text{ mm}$).
2. **3D Bounding Box / Point Cloud Overlap (3D IoU / Inlier Ratio):**
   - **Success Criterion:** **$\ge 25\%$ improvement in 3D bounding-box IoU** (or fraction of projected points falling inside the ground-truth 3D object volume) at $\text{RTT} \ge 1000\text{ ms}$.
3. **Local Real-Time Systems SLA:**
   - **Success Criterion:** Local sensor synchronizer and telemetry loop execute with **$p99$ callback latency $< 15\text{ ms}$** on M1 CPU, and **zero unhandled exceptions or thread locks** during a $60\text{ s}$ simulated network blackout (`CIRCUIT_OPEN` recovery test).

---

## 9. Four-Week Milestone Schedule & Environment Mapping

### Week 1: Data Pipeline, Interfaces & Deterministic CPU CI
- **Days 1–2 `[Local Antigravity - M1 RoboStack]`:**
  - Create `docker/robostack_env.yml` and initialize native `osx-arm64` ROS 2 Humble/Jazzy workspace.
  - Write `datasets/download_bop_ycbv_sample.sh` (verifying `scene_gt.json` presence in `ycbv_test_bop19.zip` and automatically pulling `ycbv` GT annotations or falling back to a `ycbv_train_real` sequence if absent).
  - Write `datasets/bopycb_to_rosbag2.py` to generate `.mcap` bags with `/tf`, RGB, Depth, CameraInfo, and `/eval/ground_truth_poses`. Verify playback in RViz2 on M1 Mac.
- **Days 3–4 `[Local Antigravity]`:**
  - Implement `src/async_perception_msgs` (`SegmentedObject3D.msg`, `GraspCandidate6D.msg`, `PerceptionTelemetry.msg`) and verify `colcon build`.
  - Implement `inference_server/schemas.py` and `inference_server/mock_server.py` (FastAPI server returning cached Grounded-SAM 2 RLE masks with configurable `--simulated-rtt-ms`).
- **Days 5–7 `[GitHub Actions + Local Antigravity]`:**
  - Author `docker/Dockerfile.ros2_edge` and `.github/workflows/ros2_ci.yml` to build the workspace and run `colcon test` on every push.
  - **Week 1 Exit Gate:** `.mcap` bag streams cleanly in RViz2 on M1 Mac; GitHub Actions CI badge is green.

### Week 2: Asynchronous Core Node, Ring Buffer & 3D Grasp Geometry
- **Days 8–10 `[Local Antigravity]`:**
  - Implement `ring_buffer.py` (thread-safe `collections.deque` keyed by nanosecond timestamps, holding $150$ frames / $5.0\text{ s}$ of RGB-D + CameraInfo, with `get_closest(stamp_ns, max_dt_ns)` lookup).
  - Implement `perception_node.py` with `MultiThreadedExecutor`, `ApproximateTimeSynchronizer` (`SensorDataQoS`), `tf2_ros.Buffer` ($10\text{ s}$ cache), and separated `MutuallyExclusiveCallbackGroup`s for $30\text{ Hz}$ ingestion vs. $1\text{ Hz}$ async HTTP dispatch.
- **Days 11–13 `[Local Antigravity]`:**
  - Implement `geometry_3d.py`: COCO RLE decoding (`pycocotools` / pure NumPy), $3\times3$ morphological erosion, vectorized depth back-projection, Statistical Outlier Removal (SOR), `tf2` point cloud transformation to `world`, and 3D PCA antipodal grasp pose + gripper marker generation.
- **Day 14 `[Local Antigravity]`:**
  - Implement the `compensation_mode` toggle (`compensated` vs. `naive_latest_frame`) and verify both visually in RViz2 against `mock_server.py` at $1000\text{ ms}$ RTT.
  - **Week 2 Exit Gate:** RViz2 clearly shows the green (`compensated`) 3D cloud and grasp marker staying locked on the YCB object while the red (`naive`) marker drifts into background space as the camera moves.

### Week 3: Live Colab Grounded-SAM 2 Server, Fault Tolerance & CUDA Docker
- **Days 15–17 `[Google Colab]`:**
  - Implement `inference_server/gsam2_wrapper.py` and `app.py` wrapping `IDEA-Research/Grounded-SAM-2` (`grounding-dino-tiny` + `sam2.1_hiera_small`).
  - Build `colab_server_launcher.ipynb` to mount/clone the repo, start Uvicorn/FastAPI, and expose a public HTTPS endpoint via `cloudflared` Quick Tunnels.
- **Days 18–19 `[Local Antigravity + Google Colab]`:**
  - Connect the local M1 ROS 2 node to the live Colab GPU endpoint over WAN.
  - Implement `async_client.py` and the **Circuit Breaker / TTL State Machine** (`ACTIVE` $\rightarrow$ `DEGRADED_STALE` $\rightarrow$ `CIRCUIT_OPEN`). Verify live open-vocabulary prompt switching (`ros2 param set /perception_node text_prompt "cracker box."`) and graceful recovery when killing/restarting the Colab server.
- **Days 20–21 `[GitHub Actions + Local Antigravity]`:**
  - Write `docker/Dockerfile.inference_server` (setting `CUDA_HOME`, `FORCE_CUDA=1`, and `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"` so GroundingDINO C++ ops compile headlessly without an active GPU).
  - Configure `.github/workflows/docker_cuda_build.yml` with Docker Buildx GitHub Actions cache (`type=gha`).
  - **Week 3 Exit Gate:** Live end-to-end M1 $\leftrightarrow$ Colab pipeline running at $\sim 0.6\text{--}1.2\text{ s}$ real WAN RTT; CUDA Docker workflow builds cleanly.

### Week 4: Evaluation Harness, Latency Sweep Benchmarks & Portfolio Polish
- **Days 22–24 `[Local Antigravity + Google Colab]`:**
  - Implement `eval_evaluator_node.py` and `scripts/run_rtt_sweep.sh`.
  - Execute the full quantitative sweep across $\text{RTT} \in \{100, 250, 500, 1000, 1500, 2000\}\text{ ms}$ (mock deterministic + live Colab WAN) across 3 YCB objects (`006_mustard_bottle`, `003_cracker_box`, `002_master_chef_can`), logging 3D error ($e_{\text{pos}}$), 3D IoU, and callback execution latency to CSV.
- **Days 25–26 `[Local Antigravity + GitHub Actions]`:**
  - Write `scripts/plot_latency_benchmark.py` to generate the hero benchmark plots for the README.
  - Add `test_latency_compensation_integration.py` to `ros2_ci.yml` so every push automatically verifies that `compensated` error is $< 50\%$ of `naive` error at $1000\text{ ms}$ RTT on the 60-frame CI sample sequence.
- **Days 27–28 `[Local Antigravity]`:**
  - Record a $45\text{ s}$ split-screen portfolio video/GIF (Camera RGB + delayed 2D mask, Colab WAN telemetry, and RViz2 showing red Naive vs. green Compensated 3D grasp poses simultaneously).
  - Finalize `README.md` with architecture diagrams, quantitative benchmark tables, and 1-command reproduction instructions for both M1 RoboStack and Ubuntu Docker.
  - *(Optional Stretch Goal if ahead on Day 24)*: Add a lightweight local 2D Lucas-Kanade optical flow or 3D Kalman velocity filter between remote Grounded-SAM 2 updates to handle moving objects in addition to moving cameras.
