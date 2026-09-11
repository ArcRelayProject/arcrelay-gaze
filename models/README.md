# Downloadable MNN models

These files are distributed in the `arcrelay-gaze-models-v1.0.0.zip` asset attached to the
[`gaze-models-v1.0.0`](https://github.com/ArcRelayProject/arcrelay/releases/tag/gaze-models-v1.0.0)
GitHub release. They are not stored in Git or embedded into ArcRelay architecture slices. The
packed gaze graph preserves the three-input gaze network as one flat NCHW-compatible MNN input:
left eye, right eye, then the three head-pose angles.

| File | SHA-256 |
| --- | --- |
| `face-detection-retail-0004.mnn` | `7f73526600c83ecf30b480e283ed612f0f8b9716a0257b0822d41cf0de8d7550` |
| `face-reidentification-retail-0095.mnn` | `455a55d7ebced693ee44230a7c832f7321674bf2f47604380f7f14a602dfd92e` |
| `facial-landmarks-35-adas-0002.mnn` | `c956b4a7f41eccbffe3f016aedbf0ae04c1a735db13665a0359f7cd9ed38537d` |
| `gaze-estimation-adas-0002-packed.mnn` | `4d9d3fe0a752a6afda45de76a73907d1f8dad3183230aec14ce01ab752d76d53` |
| `head-pose-estimation-adas-0001.mnn` | `2b0c4786462845997cd11622854a137ba31e1024fcd9bfa4d0ea107761e8d1b7` |
| `open-closed-eye-0001.mnn` | `f196cf4f906a3554515d0b82983e80339e6cc44b59b818458f3ca0f190a76f18` |

`face-reidentification-retail-0095.mnn` was converted from the official FP32 OpenVINO IR with
`openvino2onnx` 1.1.0 and MNNConvert 3.5.0. It produces a 256-dimensional embedding used only for
local, single-owner presence matching; ArcRelay does not transmit or expose raw embeddings.
