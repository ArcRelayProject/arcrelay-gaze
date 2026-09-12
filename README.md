# arcrelay-gaze

`arcrelay-gaze` is ArcRelay's local-first gaze estimation module. It captures RGB frames through
[`camera-rs`](https://crates.io/crates/camera-rs), executes the six-stage Intel Open Model
Zoo pipeline with MNN, calibrates observations into Arc Input's physical desk coordinates, and
stabilizes a multi-display target. Frames remain in memory and the crate performs no face
recognition.

The implementation follows four layers that can be tested independently:

1. `GazeEngine`: face detection → 35 landmarks → head pose → eye-state gate → gaze vector.
2. `Calibrator` and `WorkspaceMapper`: per-camera, per-layout calibration into physical desk
   micrometres, followed by display and logical-coordinate resolution.
3. `GazeTracker`: explicit camera lifecycle, latest-frame delivery, bounded memory and structured
   diagnostics for a desktop host.
4. `TargetStabilizer`: dwell, display hysteresis and loss grace used by Arc Input's safe
   gaze-preselection policy. Physical mouse or keyboard activity remains the confirmation signal.

## Model bundle

Production builds never embed model bytes in an executable. Download and extract the architecture-
independent `arcrelay-gaze-models-v1.0.0.zip` asset from the
[`gaze-models-v1.0.0`](https://github.com/ArcRelayProject/arcrelay/releases/tag/gaze-models-v1.0.0)
GitHub release, then set `ARCRELAY_GAZE_MODEL_DIR` to that directory. The loader checks every size
and SHA-256 value before creating an MNN model.

Camera capture is also target-specific: AVFoundation on Apple platforms, Media Foundation on
Windows, V4L2 on Linux, and Camera2 on Android. The dependency disables `camera-rs` defaults so
production builds do not include fallback UVC or unrelated platform backends.

## Run the standalone demo

```bash
ARCRELAY_GAZE_MODEL_DIR=/path/to/models \
  CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --features demo --example egui_demo
```

Use `--self-test` to execute all six bundled MNN graphs without opening a camera:

```bash
ARCRELAY_GAZE_MODEL_DIR=/path/to/models \
  CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --features demo --example egui_demo -- --self-test
```

Use `--camera-test` for a bounded 20-inference smoke test. It selects the first camera unless a
`camera-rs` device ID follows the flag:

```bash
ARCRELAY_GAZE_MODEL_DIR=/path/to/models \
  CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --features demo --example egui_demo -- --camera-test
```

The demo deliberately uses the public library API. It is therefore also an integration test for
the `camera-rs` adapter rather than a second capture implementation.

## Privacy and safety defaults

- RGB frames and preview pixels are never serialized or written to disk by this crate.
- Preview delivery is opt-in and intended only for calibration/debug UI.
- A calibration profile is invalidated when its physical display-layout digest changes.
- Closed eyes, a lost face, stale calibration, and points outside the workspace produce no target.
- Stable gaze selects a candidate only. The ArcRelay desktop integration owns any subsequent input
  routing and must require a physical confirmation event.

## Models and licensing

See [models/README.md](models/README.md), [model-bundle-manifest.json](model-bundle-manifest.json),
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The Rust source is licensed under
AGPL-3.0-only. Downloadable model files retain their upstream terms.
