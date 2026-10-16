# Physical drone transition handoff — 2026-10-16

**Purpose:** park the current VILLOC research block cleanly and resume the
physical work-drone investigation with reproducible scripts/logs.

This handoff deliberately separates:

1. what is already established in the localization research;
2. what was tested informally on the physical drone but not saved in the repo;
3. what must now be repeated and versioned;
4. why the hardware phase is needed for the unfinished GSD / geometry work.

---

## 1. Why move to the drone now

The localization research has reached a useful stopping point:

- DINOv2 retrieval is mature enough to shortlist candidate localities;
- SIFT+LightGlue is a materially stronger absolute frontend than the earlier
  ORB-only stack;
- the residual nadir failure mechanism has been analyzed;
- 45-degree stress has been measured;
- adding more matchers immediately is lower value than obtaining the real
  camera/autopilot geometry needed for physical scale and deployment work.

The important unfinished research question is:

> Can physically justified UAV-frame-to-orthophoto scale/GSD normalization and
> real camera geometry make the cross-view frontend stronger and more
> deployable?

That question depends on the real drone's camera calibration, pose, altitude,
timestamps, and compute/network interfaces.

---

## 2. Known physical work-drone state from prior bench testing

The following facts were established in the earlier chat but were **not**
properly saved as repository evidence. They must be revalidated and logged.

Known platform components:

- NVIDIA Jetson Orin NX onboard compute
- Pixhawk 6X flight controller
- Ethernet connection between Jetson and Pixhawk
- separate cameras; not yet established as a calibrated stereo pair
- GPS
- 4G
- Bluetooth
- Wi-Fi
- intended MAVLink interface

Known network discovery:

- Pixhawk cable corresponds to Jetson interface enP8p1s0
- another interface enx0c5b8f279a64 was observed separately
- target Jetson address used for the Pixhawk link: 192.168.144.15/24
- Pixhawk address: 192.168.144.14
- ping succeeded with 0% loss and approximately 0.727 ms average latency

Known MAVLink result:

- UDP target: 192.168.144.14:14550
- heartbeat received
- system/component: 1 / 1
- autopilot type: ArduPilot
- vehicle type: quadcopter
- AUTOPILOT_VERSION returned flight software 4.7.1

Known limitation at pause:

- heartbeat/version communication worked;
- normal telemetry such as ATTITUDE, GPS, battery and IMU was not yet observed
  in the listener;
- the next planned test was requesting ATTITUDE at 2 Hz with
  MAV_CMD_SET_MESSAGE_INTERVAL.

Important:

These results came from interactive bench work. They are context, **not yet
repo-frozen evidence**.

---

## 3. Safety boundary for the resumed hardware phase

Until the platform and control paths are fully understood:

- bench test only;
- read-only telemetry;
- no arming;
- no motor commands;
- no actuator commands;
- no mode changes unless explicitly required and reviewed;
- do not modify Pixhawk parameters casually;
- preserve current network/config state before changes.

The first hardware scripts should enumerate and log. They should not control.

---

## 4. Repository discipline from this point onward

The earlier hardware test was useful but not reproducible because the code and
terminal evidence were not saved.

That should not happen again.

Recommended structure:

~~~text
scripts/hardware/work_drone/
    probe_platform.sh
    probe_network.sh
    probe_cameras.py
    probe_mavlink.py
    request_telemetry.py
    record_camera_telemetry.py

configs/hardware/work_drone/
    work_drone.yaml
    telemetry_messages.yaml

outputs/hardware/work_drone/<run_id>/
    system/
    network/
    cameras/
    mavlink/
    calibration/
    captures/
    logs/
    summary/

docs/hardware/work_drone/
    README.md
    calibration.md
    telemetry.md
    gsd_geometry.md
~~~

Every command should be reproducible from a committed script where practical.

Every run should record:

- timestamp;
- Git commit;
- hostname;
- OS / L4T;
- connected devices;
- interface/IP state;
- camera IDs;
- Pixhawk firmware;
- telemetry availability;
- command line;
- stdout/stderr logs.

---

## 5. Recommended resumed workflow

### D0 — platform inventory and evidence snapshot

Goal:

Create a versioned baseline of the onboard computer before changing anything.

Collect:

- device model;
- OS;
- kernel;
- L4T / JetPack;
- CPU/GPU/memory;
- storage;
- Python;
- CUDA;
- USB devices;
- PCI/network devices;
- current routes;
- current services/processes relevant to cameras/MAVLink.

Useful read-only probes include:

~~~bash
cat /proc/device-tree/model
uname -a
cat /etc/os-release
cat /etc/nv_tegra_release
df -h
free -h
ip -br link
ip -br addr
ip route
lsusb
~~~

Save the outputs to a run directory.

### D1 — restore and document Jetson <-> Pixhawk Ethernet

Verify:

~~~bash
ip -br addr show enP8p1s0
ip route
ping -c 4 192.168.144.14
~~~

Do not assume the interface/IP is still configured exactly as before.

Record:

- link state;
- assigned IP;
- route;
- latency;
- packet loss.

### D2 — restore MAVLink and save a reusable telemetry probe

Create a committed pymavlink probe instead of one-off terminal snippets.

Required checks:

1. heartbeat;
2. AUTOPILOT_VERSION;
3. SYS_STATUS;
4. ATTITUDE;
5. LOCAL_POSITION_NED if available;
6. GLOBAL_POSITION_INT / GPS_RAW_INT for evaluation/reference;
7. HIGHRES_IMU / RAW_IMU if available;
8. battery;
9. rangefinder / distance sensor if present.

If streams are not emitted automatically, request them explicitly with
MAV_CMD_SET_MESSAGE_INTERVAL.

The first requested stream should reproduce the previously planned ATTITUDE at
2 Hz test.

### D3 — enumerate every camera and driver path

This stage is critical for GSD.

Capture:

- camera vendor/model;
- serial/device ID;
- RGB / thermal / other stream identity;
- resolution and available FPS modes;
- V4L2 / GStreamer / vendor SDK path;
- whether intrinsics are exposed;
- whether calibration EEPROM/file exists;
- distortion model;
- rolling/global shutter;
- hardware/software synchronization capability.

Possible read-only probes, depending on installed tools:

~~~bash
lsusb
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl --all -d /dev/videoX
~~~

Do not assume the two cameras form a calibrated stereo pair merely because two
cameras exist.

### D4 — establish coordinate frames and extrinsics

Define at minimum:

- Pixhawk/body frame;
- camera optical frame;
- gimbal frame if present;
- Jetson timestamp domain;
- Pixhawk timestamp domain.

Determine:

- camera mounting orientation;
- camera-to-body rotation;
- camera-to-body translation if needed;
- whether gimbal angles are available over MAVLink;
- whether camera and Pixhawk timestamps can be synchronized.

### D5 — capture a synchronized bench/flight sample

Create one recorder that writes:

~~~text
frame_id
host_timestamp
camera_timestamp
image path / stream timestamp
camera intrinsics ID
vehicle attitude
angular rate
altitude / AGL source
rangefinder if available
GPS only as reference
Pixhawk boot time
~~~

GNSS must remain evaluation/reference data for GNSS-denied localization
experiments.

### D6 — compute physically justified UAV-frame ground scale

For a rectified pinhole camera near nadir over approximately planar ground:

~~~text
GSD_x approximately H / fx_pixels
GSD_y approximately H / fy_pixels
~~~

where H is camera height above the local ground plane.

For a more exact model:

1. undistort image points;
2. build camera rays from K^-1;
3. rotate rays with camera/body attitude and extrinsics;
4. intersect rays with a ground plane / DEM / range surface;
5. measure the ground footprint represented by each image region.

For oblique imagery, use ray-ground intersection rather than one scalar GSD.

Required real data:

- fx, fy, cx, cy;
- distortion coefficients;
- image resolution;
- camera pose/extrinsics;
- AGL or ground-range estimate;
- timestamps;
- attitude.

Physical sensor width is optional if calibrated focal length in pixels is
available.

### D7 — reopen the frozen GSD-normalization experiment

Only after D0-D6 are documented should the localization research branch reopen
the true scale experiment.

Scientific question:

> If the UAV frame is resampled or represented at a physically justified
> ground scale relative to the orthophoto, does candidate discrimination improve
> for ORB, SIFT+LightGlue, or both?

Run scale normalization separately from rotation/yaw normalization first.

Do not mix scale, rotation, confidence, and new matcher changes in one
experiment.

---

## 6. Relation to the 45-degree result

OB1.0 showed that oblique imagery degrades final locality selection even though
DINO Top20 remains strong.

The physical-drone geometry work can address a part of that problem by enabling:

~~~mermaid
flowchart TD
    C[Camera intrinsics] --> G[Ray / ground geometry]
    E[Camera-body extrinsics] --> G
    A[Vehicle attitude] --> G
    H[AGL / range / ground model] --> G
    G --> F[Per-frame ground footprint / local GSD]
    F --> S[Physically normalized UAV-to-map representation]
    S --> M[SIFT+LightGlue / ORB comparison]
~~~

This is a stronger approach than applying one constant 45-degree footprint
offset.

---

## 7. Thermal camera future use

If a thermal stream exists on the work drone, do not immediately integrate it
into the primary stack.

First record its:

- model;
- resolution;
- calibration;
- timestamp behavior;
- FOV;
- frame alignment with RGB.

Later possibilities:

- thermal structural challenger;
- RGB + thermal confidence fusion;
- cross-spectral retrieval/matching;
- failure recovery under low light.

The first algorithmic test should remain small.

---

## 8. Immediate next chat / coding-agent task

Start a **new hardware-inventory stage**, not a new localization-method stage.

The new agent should:

1. read CURRENT_BEST_VILLOC_LOCALIZATION_STACK.md;
2. read FA1_OB1_CLOSEOUT.md;
3. read this file;
4. inspect the repository for any existing hardware/drone scripts;
5. create D0 platform/network probes;
6. run them on the Jetson;
7. save every output;
8. proceed to D1 MAVLink only after D0 is documented.

Do not redesign the localization stack before the physical sensor geometry is
known.

---

## 9. Parking statement

The VILLOC localization research is **parked, not abandoned**.

Reason for parking:

- current stack is materially stronger than the earlier baseline;
- current failure mechanisms are documented;
- the next high-value unresolved scale/geometry experiment depends on physical
  sensor/calibration data;
- physical drone integration is now the prerequisite.

Re-entry condition:

> Resume localization research after camera + Pixhawk geometry, timestamps,
> intrinsics, AGL source, and telemetry are documented well enough to define a
> causal GSD-normalization protocol.



---

# Update — additional Jetson tooling, camera/network architecture and relative-work priority

This update supersedes the earlier assumption that the Jetson MAVLink tooling
still needed to be created from scratch.

## A. Existing Jetson MAVLink tooling

The Jetson already has a separate `mavtools` workspace containing:

~~~text
check_battery_status.py
check_pixhawk.py
led_control.py
preflight_checks.py
preflight_messages_20261014_164428.json
preflight_parameters_20261014_164428.json
preflight_report_20261014_164428.txt
venvs/
~~~

pymavlink and supporting packages were installed in a dedicated virtual
environment under that workspace.

Implications:

- inspect and preserve these existing scripts before writing replacements;
- do not commit the `venvs/` directory;
- freeze the environment with `python -m pip freeze`;
- review `led_control.py` before execution because it may send commands rather
  than remaining read-only;
- review saved Pixhawk parameter/message dumps for secrets, credentials,
  device-specific identifiers, or unnecessary bulk configuration before Git.

## B. Recommended Git/worktree migration from Jetson

The clean approach is a dedicated hardware worktree and branch.

First discover the actual Jetson Git topology rather than assuming paths:

~~~bash
git -C /path/to/drone-localisation rev-parse --show-toplevel
git -C /path/to/drone-localisation status --short --branch
git -C /path/to/drone-localisation worktree list
git -C /path/to/drone-localisation remote -v
~~~

From the main repository shown by `git worktree list`:

~~~bash
git fetch origin

git worktree add \
  -b hardware/work-drone-instrumentation-v1 \
  /home/rodu/projects/drone-localisation-hardware \
  origin/research/crossview-scale-rotation-domain-v1
~~~

Then:

~~~bash
cd /home/rodu/projects/drone-localisation-hardware
git status --short --branch

mkdir -p scripts/hardware/work_drone/mavlink
mkdir -p configs/hardware/work_drone
mkdir -p docs/hardware/work_drone
~~~

Copy only reviewed source scripts from the existing `mavtools` directory:

~~~bash
cp /actual/path/to/mavtools/check_battery_status.py \
  scripts/hardware/work_drone/mavlink/

cp /actual/path/to/mavtools/check_pixhawk.py \
  scripts/hardware/work_drone/mavlink/

cp /actual/path/to/mavtools/led_control.py \
  scripts/hardware/work_drone/mavlink/

cp /actual/path/to/mavtools/preflight_checks.py \
  scripts/hardware/work_drone/mavlink/
~~~

Do **not** copy the virtual environment. Instead:

~~~bash
source /actual/path/to/mavtools/venvs/<venv>/bin/activate

python --version
python -m pip freeze \
  > configs/hardware/work_drone/requirements-mavtools-freeze.txt
~~~

Before committing the preflight JSON/TXT artifacts, inspect them:

~~~bash
grep -RniE 'password|passwd|secret|token|key|ssid|apn|user|credential' \
  /actual/path/to/mavtools/preflight_* || true
~~~

A full parameter dump may be noisy or device-specific. Preserve the original
locally, then commit either a sanitized evidence copy or a summarized report
plus the original SHA256.

Finally:

~~~bash
git add scripts/hardware/work_drone configs/hardware/work_drone docs/hardware/work_drone
git status --short
git diff --cached
~~~

Review before committing.

## C. MAVLink timing/rate characterization

Do not stop at "the message arrived."

For every stream needed by localization, record:

- requested rate;
- observed rate;
- inter-arrival p50/p95/p99;
- jitter;
- sequence gaps/drops where observable;
- Jetson monotonic receive timestamp;
- Pixhawk boot/message timestamp;
- clock offset/drift;
- burstiness;
- CPU cost while logging.

Priority telemetry includes:

- ATTITUDE / attitude quaternion as appropriate;
- angular rates;
- HIGHRES_IMU or suitable raw/scaled IMU;
- LOCAL_POSITION_NED if available;
- barometric altitude;
- rangefinder/distance sensor if present;
- battery voltage/current/status;
- GPS only as reference/evaluation in GNSS-denied experiments.

Do not request maximum rates blindly. Determine reliable rates and whether
camera/localization load changes them.

## D. Camera pipeline — localization must receive unburned pixels

The current camera path has been observed burning timestamp text into the image.
The localization branch should receive pixels **before** any overlay.

Target structure:

~~~mermaid
flowchart TD
    C[Camera / JPEG-MJPEG source] --> J[jpegdec / camera decode]
    J --> T[raw-frame tee]

    T --> Q1[queue]
    Q1 --> L[Localization appsink<br/>raw frame, no overlay]

    T --> Q2[queue]
    Q2 --> O[Optional visible timestamp/display overlay]
    O --> E[H.264 hardware encoder]
    E --> ET[encoded-stream tee]

    ET --> F[MP4 recording]
    ET --> N[Network streaming]
~~~

Conceptual GStreamer form:

~~~text
camera
  -> jpegdec
  -> tee name=rawtee

rawtee. -> queue -> appsink
rawtee. -> queue -> optional timeoverlay -> hardware H.264 encoder
         -> h264parse -> tee name=enctee

enctee. -> queue -> mp4mux -> filesink
enctee. -> queue -> network streaming branch
~~~

Exact elements depend on the real camera/Jetson driver and must be discovered.

Rules:

- put a `queue` on each tee branch;
- keep overlay and encoder off the localization branch;
- record timestamp metadata in a sidecar/log even if a visible overlay exists;
- do not accidentally resize/crop the localization branch;
- log camera source timestamp, Jetson receipt timestamp and frame sequence ID;
- measure decode/queue latency and dropped frames;
- preserve some raw unoverlaid recordings for offline parity/calibration.

## E. 4G, Wi-Fi, radio and secure remote access

Inventory separately:

- 4G modem model/interface/SIM/APN behavior;
- Wi-Fi chipset/mode/range;
- radio transmitter/telemetry link and its control/data role;
- Ethernet paths;
- whether one or multiple operator laptops need access.

Do **not** expose MAVLink, SSH or camera-stream services directly to the public
mobile network.

Preferred structure:

~~~mermaid
flowchart LR
    D[Drone Jetson] --> V[Private VPN overlay]
    V --> L1[Authorized laptop 1]
    V --> L2[Authorized laptop 2]
    D --> FC[Pixhawk private onboard link]
    D --> CAM[Camera/video service]
~~~

Use an authenticated private VPN/overlay such as a WireGuard/Tailscale-class
design, with ACLs, SSH keys and firewall rules appropriate to the final
deployment.

Measure under idle and streaming load:

- RTT p50/p95/p99;
- jitter;
- packet loss;
- throughput;
- reconnect time after link outage;
- route changes;
- CPU cost;
- end-to-end video latency;
- dropped/corrupted frames;
- MAVLink rate degradation while video is active.

Also test contention: high-rate IMU logging + camera decode + H.264 recording +
network streaming + localization.

## F. Synchronized recorder contract

The first reusable recorder should capture at least:

~~~text
frame_id
camera_source_timestamp
jetson_monotonic_receive_timestamp
image_path / buffer sequence
camera_intrinsics_id
pixhawk_boot_or_message_timestamp
vehicle_attitude
angular_rate
imu_accel
imu_gyro
barometric_altitude
AGL/rangefinder if available
battery voltage/current/status
GPS/GNSS only as evaluation/reference
~~~

This recorder is the bridge between hardware characterization and later
relative/GSD experiments.

## G. Next algorithmic focus — relative estimation first

Once D0-D5 timing/sensor characterization is trustworthy, the next active
localization work on the drone should be the **relative-estimation path**.

Start with controlled stationary/hover data.

Questions:

1. What inertial/IMU drift appears while physically stationary or holding hover?
2. What apparent motion does the camera relative frontend report over the same
   interval?
3. What are the timing offsets/rate differences?
4. Can visual relative motion constrain inertial drift without suppressing real
   short-term dynamics?

Initial conceptual architecture:

~~~mermaid
flowchart LR
    IMU[High-rate Pixhawk IMU / attitude] --> E[Relative state estimator]
    CAM[Raw unoverlaid camera frames] --> V[Visual relative motion]
    V --> E
    ALT[Range / altitude if reliable] --> E
    E --> R[Relative pose + uncertainty]
    GPS[GNSS/RTK if available] --> GT[Evaluation only]
    GT -. post-run comparison .-> R
~~~

Important boundary:

- first run the estimator in observer/offline mode;
- Pixhawk's flight-control EKF/stabilization remains authoritative during data
  collection;
- do not immediately feed experimental visual estimates into flight control;
- prove timing alignment, bounded hover drift and recovery first.

Suggested progression:

~~~text
bench stationary
  -> controlled hover data
  -> offline/replay estimator
  -> slow translation
  -> yaw/turn segments
  -> larger relative trajectory
~~~

Metrics:

- apparent position/velocity drift during stationary/hover;
- yaw drift;
- visual frame-to-frame failure rate;
- visual/IMU directional agreement;
- innovation/residual distributions;
- acquisition-to-estimator latency;
- dropped/late measurement behavior;
- recovery after visual degradation;
- CPU/GPU load and thermals.

The final fusion method is intentionally **not frozen yet**. Real measurements
may justify EKF/ESKF, visual-inertial optimization, or another controlled
design.

## H. Updated parking decision

Absolute-localization research is now parked with a reproducible current-stack
software path. The next active work is:

~~~text
preserve existing Jetson tools
    -> hardware/network/camera characterization
    -> synchronized sensor recorder
    -> stationary/hover relative-estimation dataset
    -> relative visual/inertial estimator study
    -> later physical GSD / camera-ray experiment
    -> later reopen absolute localization
~~~

Re-entry to absolute-localization research should occur only when real camera
intrinsics/extrinsics, timing, attitude and AGL/range information are
documented well enough to support the physical GSD/ground-ray experiment.
