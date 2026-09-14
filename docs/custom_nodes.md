# Custom (non-generated) nodes

Every node in this pack that is **hand-written**, i.e. everything except the
~470 auto-generated `cv2_*` wrappers (one per entry in `registry.py` /
`registry_contrib.py`, skipped when the installed OpenCV lacks the function)
built at import time by `lowlevel.py`. 317 nodes over 18 modules.

## Classification

| Tag | Meaning |
| --- | --- |
| **CG** | *Wraps a cv2 class* - the underlying cv2 API is a class or a `create*` factory rather than a plain function. The node builds the object, uses it and drops it within one run, so nothing stateful travels along a link. |
| **PS** | *Pipeline simplification* - the cv2 entry points are plain functions, but a useful step needs several of them plus shape/dtype plumbing, named dropdowns, batching or a `found` flag. |
| **NC** | *Non-OpenCV* - the algorithm or data handling is implemented here; cv2 provides at most primitives. |
| **TB** | *Type bridge / IO* - moves data between ComfyUI socket types, files, literals and NPARRAY. No image processing. |
| **VIZ** | *Visualization* - draws or previews. Always a separate node: the data nodes never draw, so you choose what gets rendered and what stays data. |

A node may carry two tags when both halves are load-bearing.

---

## `highlevel.py` - curated IMAGE/MASK nodes (25)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Transform (Rotate/Scale/Shift) | Rotate about centre + scale + shift + flip in one warpAffine | PS |
| CV Contrast (CLAHE/Equalize) | Adaptive or global histogram equalization, batch-aware | CG (CLAHE is `createCLAHE`) |
| CV Color Map | Grayscale -> false colour; standard viewer for score maps | VIZ |
| CV Detect Lines (Hough) | Canny + probabilistic Hough in one step | PS |
| CV Find Absent Color (Lab) | First CSS named color not in the image, furthest from the existing ones in L*a*b* (L2/L1); returns scalar, name, hex, distance, `found` | NC |
| CV Match Template Multi-Scale | Template match over a scale pyramid, best score + box | NC |
| CV Connected Components (Split Mask) | One MASK per blob | PS |
| CV Masks to BBoxes | Tight box per mask of a batch | NC |
| CV BBoxes to Masks | Inverse: filled rect masks | NC |
| CV Snap BBoxes (Crop Policy) | Authors the crop sizing policy string (padding/square/multiple/max) | NC |
| CV Crop by BBoxes | Crop per box in the source's native container (no uint8 round trip); batch or list output | NC |
| CV Crop by Masks | Same, driven by a MASK batch | NC |
| CV Paste by BBox | Paste crops back at their boxes | NC |
| CV Scale BBoxes | Rescale boxes between image spaces | NC |
| CV BBoxes To Array | BOUNDING_BOX -> plain arrays (xyxy, scores) | TB |
| CV Array To BBoxes | Plain arrays -> BOUNDING_BOX (per-frame nesting) | TB |
| CV NMS Boxes | Non-maximum suppression over any BOUNDING_BOX | PS |
| CV Box IoU Matrix | All-pairs IoU of two box sets | NC |
| CV Overlay Masks | Colour each mask of a batch over the image | VIZ |
| CV Apply Chromatic Aberration | Simulates lateral CA by per-channel radial scaling | NC |
| Create CA Calibration | Synthesizes a CA calibration target | NC |
| CV Chromatic Aberration Correction | Polynomial per-channel radial correction | NC |
| CV Photometric Align (Gain/Bias) | Trimmed robust fit of `src*gain+bias ~ ref`, seeded by raw difference | NC |
| CV Local Linear Fit (per pixel) | Per-pixel windowed regression (slope/intercept) - backs clean-plate matting | NC |
| CV Labels to Masks (full size) | Label map (HFS/superpixels/watershed) -> one MASK per region | NC |

## `features.py` - features, geometry, calibration, stitching (75)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Detect Features | SIFT/ORB/BRISK/AKAZE/KAZE keypoints + descriptors | CG |
| CV Filter Keypoints | Keep a robust subset (response/size/count), descriptors in lockstep | NC |
| CV Match Features | BFMatcher / FLANN, ratio test | CG |
| CV Find Homography (RANSAC) | H + inlier mask + `found`, safe identity fallback | PS |
| CV Find Fundamental Matrix | F + inliers + `found` | PS |
| CV Draw Epipolar Lines | Epiline overlay | VIZ |
| CV Draw Keypoints | Keypoint overlay | VIZ |
| CV Draw Matches | Side-by-side match overlay | VIZ |
| CV Array Size | Reads w/h of an ndarray for `dsize` inputs | TB |
| CV Detect Circle Grid | Symmetric / asymmetric circle calibration pattern | PS |
| CV Calibrate Camera (Circle Grid) | Intrinsics + distortion from circle-grid views | PS |
| CV Image Corners | The 4 image corners as a point array | NC |
| CV Draw Polygon | Polyline/polygon overlay | VIZ |
| CV Draw Text | putText label | VIZ |
| CV Flow To Color | Angle field -> HSV colour wheel | VIZ |
| CV Optical Flow (Farneback) | Dense flow, free-function | PS |
| CV Draw Flow Grid | Dense flow field as an arrow grid | VIZ |
| CV Stereo Disparity (SGBM) | Disparity from a rectified pair (`StereoSGBM_create`), exposes `disp12MaxDiff` | CG |
| CV Stereo Rectify (Uncalibrated) | Hartley rectifying homographies from matches | PS |
| CV Calibrate Camera (Chessboard) | Intrinsics + distortion from chessboard views | PS |
| CV Fisheye Calibrate (Chessboard) | Equidistant model; handles the (1,N,k) shape + flag traps | PS |
| CV Fisheye Undistort | Fisheye rectification; detects the all-NaN new-K case | PS |
| CV Omnidir Calibrate (Chessboard) | CMei catadioptric calibration (`cv2.omnidir` flags/module) | PS |
| CV Omnidir Undistort | Re-projection; closed-form CMei inversion instead of the wrong `undistortPoints` | PS + NC |
| CV Omnidir Stereo Reconstruct | Omnidir stereo rectify + match + cloud, judged by the valid mask | PS |
| CV Stereo Calibrate (Chessboard) | Relative R,T of two cameras | PS |
| CV Stereo Calibrate (Circle Grid) | Same from circle grids | PS |
| CV Decompose Projection Matrix (RQ) | P -> K, R, t | PS |
| CV Decompose Homography | 4 planar-motion candidates + normalized-point disambiguation | PS |
| CV Solve PnP (All Poses) | Multi-solution PnP (IPPE/SQPNP), all poses + errors | PS |
| CV Solve PnP (Pose) | Single pose, all-points or RANSAC + inliers/reproj error | PS |
| CV Pose To Matrix | rvec+tvec -> 4x4 | NC |
| CV Matrix To Pose | 4x4 / 3x4 -> rvec+tvec, i.e. a 3D matcher's pose back into something renderable; nearest rotation by SVD, reports the scale it divided out | NC |
| CV ArUco Detect Markers | `ArucoDetector` marker detection | CG |
| CV ArUco Draw Markers | Marker overlay | VIZ |
| CV ArUco Board Pose (Average) | Planar board pose | CG |
| CV ArUco Board Image | Printable GridBoard / ChArUco render (aspect derived to dodge the assert) | CG |
| CV ChArUco Detect | ChArUco corners + `found` | CG |
| CV Calibrate Camera (ChArUco) | Intrinsics from a ChArUco batch | CG |
| CV Recover Pose (Essential Matrix) | E -> R,t with cheirality check | PS |
| CV Triangulate Points (Two-View) | Sparse 3D from two views | PS |
| CV Register Point Clouds (3D) | `estimateAffine3D` rigid/affine fit between clouds | PS |
| CV Track Features (KLT) | Pyramidal Lucas-Kanade tracking with status filtering | PS |
| CV Visual Odometry (Sequence) | The whole KLT -> recoverPose -> compose chain folded over an IMAGE batch, carrying tracks between frames and re-detecting only when too few survive. Shares its code with the three per-pair nodes; exists because a graph loop cannot collect a trajectory (a foreach fold keeps one accumulator, the pose needs two). Failed steps HOLD the pose and flag `found_mask` | PS |
| CV Trajectory Error (ATE / RPE) | Scores one trajectory against another after a rigid (SE3) or similarity (Sim3) Umeyama alignment - without which the comparison measures each run's arbitrary frame, and for monocular its arbitrary scale | NC |
| CV Find Transform (ECC) | Direct intensity registration; mask must be a coherent region | PS |
| CV Phase Correlate (Translation) | Sub-pixel global shift | PS |
| CV Track Window | One meanShift / CamShift step | PS |
| CV Compose Pose (Trajectory) | Chains relative poses into a trajectory | NC |
| CV Stitch Canvas | Output canvas + offset for a two-image stitch | NC |
| CV Stitch | `cv2.Stitcher` over a batch | CG |
| CV Stitch (List) | Same over a ComfyUI list | CG |
| CV Stitch (Advanced) | Stitcher with exposed estimator/blender knobs | CG |
| CV Stitch (Advanced List) | Same over a list | CG |
| CV Constant Like | uint8 constant array shaped like another | NC |
| (warn) CV ZeroStride Like | Zero-stride broadcast helper (advanced/diagnostic) | NC |
| CV Feather Blend | Coverage-weighted blend of two warped images | NC |
| CV Stack Images | hconcat/vconcat of N images/masks (autogrow, type-following) | NC |
| CV Stereo Calibration From CSV | Per-image stereo calibration read from CSV | TB |
| CV Scale Homography | Rescale H between resolutions | NC |
| CV Eye | NxN identity matrix | NC |
| CV Matrix Multiply | `left @ right` | NC |
| CV Stereo Disparity (BM) | `StereoBM_create` disparity | CG |
| CV Quasi-Dense Stereo | Seed-and-grow stereo; scatters SIGNED disparity from `getDenseMatches`, NaN-safe | CG |
| CV Kalman Filter Step | One predict/correct of `cv2.KalmanFilter` for a 2-D track | CG |
| CV Detect Line Segments (LSD) | `createLineSegmentDetector` | CG |
| CV Detect Blobs | `SimpleBlobDetector` (validates disabled filters - handled) | CG |
| CV Detect Corners | FAST / GFTT / Harris descriptor-free corners | CG |
| CV Compute Descriptors | Describes externally-found keypoints | CG |
| CV Generalized Hough (Shape Template) | Arbitrary-shape template search (Ballard/Guil) | CG |
| CV Optical Flow (DIS) | Dense Inverse Search flow (copies the flow arg) | CG |
| CV Refine Flow (Variational) | `VariationalRefinement` over an existing field | CG |
| CV Thin Plate Spline Warp | Non-rigid warp from correspondences; two fits (points vs image) | CG |
| CV Affine Shape Warp | Rigid/affine shape transformer; one fit for both directions | CG |
| CV Paste Through Warp | Warp an image through correspondences and composite it onto a background; 5 models, premultiplied | PS |

## `contours.py` - contours, shapes, components (22)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Find Contours | Binary -> CV_CONTOURS | PS |
| CV Draw Contours | Contour overlay | VIZ |
| CV Select Contour | Pick one by rank (area/length/position) | NC |
| CV Filter Contours | Keep by measured property range | NC |
| CV Filter Contours By Shape | Keep by `matchShapes` similarity to a reference, with a selectable scale / rotation / mirror invariance policy | PS |
| CV Merge Contours | Union contours within a gap | NC |
| CV Find Quadrilateral | Largest convex quad (document/poster finding) | PS |
| CV Quad To Rectangle | Target rectangle for a quad rectification | NC |
| CV Quad Warp | Apply the 4-corner homography | PS |
| CV Enclosing Circle | Min enclosing circle of a contour | PS |
| CV Shape Moments | moments / HuMoments table per contour, plus canonical 0-360 orientation, signed chirality and pose confidence | PS |
| CV Contour To Points | CV_CONTOURS -> NPARRAY | TB |
| CV Points To Contour | NPARRAY -> CV_CONTOURS | TB |
| CV Skeletonize | ximgproc thinning with a morphological fallback | PS + NC |
| CV Select Component At Point | Keep the blob(s) covering a point | NC |
| CV Keep Largest Component | Keep the biggest blob | NC |
| CV Components Touching Border | Keep/drop blobs reaching a frame edge (sky masks, clipped detections) | NC |
| CV Fill Holes | Fill interior holes of a mask | NC |
| CV Region Properties | Regions -> (N,D) feature matrix + label map; flag-selected columns (position/size/shape/orientation/colour/Hu moments/equivalent ellipse/min-area rect + rectangularity/centroid clearance/topology: hole count, skeleton loose ends, convexity defects, corner count/distance transform: median thickness + variation, inscribed circle radius, roundness, skeleton length, radial distance variation) | NC |
| CV Region Property Flags | Authors that column selection on the canvas (the widget a promoted subgraph input does not carry) + names/width | TB |
| CV Detect MSER Regions | `cv2.MSER` stable regions | CG |
| CV Shape Distance | Shape-context / Hausdorff distance from the `create*` factory | CG |

## `moments.py` - image moments over the INTENSITY (2)

`cv2.moments` takes either a point set or a raster, and the raster case defaults
to `binaryImage=False` - the mass of a pixel is its VALUE, which is Hu's original
density function and the case `contours.py` cannot reach (it integrates a uniform
density inside an outline). Two regions with identical silhouettes but different
internal shading are the same shape to `Shape Moments` and different shapes here.

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Image Moments | Per-frame moments/Hu of the image intensity + canonical pose, mirroring `Shape Moments` output-for-output | PS |
| CV Match Image Moments | Reference + batch -> distances, keep-mask, best_index and the same invariance policy as `Filter Contours By Shape` | PS |

## `points.py` - point sets and 3D clouds (30)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Annotate Points | Manual point input on a canvas widget | NC |
| CV Annotate Correspondences | Manual point pairs across two images | NC |
| CV Filter Points By Distance | Keep points in a radius band | NC |
| CV Filter Points By Mask | Keep points flagged by a per-point mask | NC |
| CV Points To Polar | Cartesian -> (r, theta) about an origin | NC |
| CV Polar To Points | Inverse | NC |
| CV Reduce Points By Label | Per-cluster centroid/extreme point | NC |
| CV Reduce Values By Label | Per-cluster reduction of a value array | NC |
| CV Pick Value (sorted) | Pick the k-th value out of a small array | NC |
| CV Draw Points | Point markers | VIZ |
| CV Draw Connections | Edge list overlay | VIZ |
| CV Draw Labels | Per-point text (core Draw BBoxes cannot label non-COCO) | VIZ |
| CV Draw Rays | Origin-to-point rays | VIZ |
| CV Draw Segments | Endpoint-pair segments | VIZ |
| CV Draw Circles | (x, y, r) circles | VIZ |
| CV Draw Ellipses | Nx5 ellipses | VIZ |
| CV Draw Flow Vectors | Arrows between two point sets | VIZ |
| CV Project To Plane | Drop one axis of a 3D cloud (orthographic view) | NC |
| CV Fit Points To Box | Scale/centre a 2D point set into a box; `extent_from` shares ONE mapping across several sets (fitted separately, each fills the box and the overlay lies about their relative size), `y_axis` flips the world axis that pixel rows would otherwise draw upside down (a mirrored top-down map turns a left turn into a right one), and the mapping itself comes back as scale/offset | NC |
| CV Draw Path (ordered) | An ordered point sequence drawn so its DIRECTION and ENDS read: colour or thickness ramp start-to-end, distinct marker per end, heading arrows, and `gap_mask` points marked as held. Redundant on purpose - it survives greyscale | VIZ |
| CV Draw Plot Frame | Grid, world-unit ticks and a scale bar under a 2D plot, reproducing Fit Points To Box's mapping exactly so a tick and a data point agree | VIZ |
| CV Scale Points | Multiply coordinates | NC |
| CV Delaunay / Voronoi | `Subdiv2D` triangulation / Voronoi cells | CG |
| CV Sample Array At Points | Read array values at points, with `valid_in` chaining (empty-safe) | NC |
| CV Transform Points 3D | Apply 3x4 / 4x4 / 3x3 to an Nx3 cloud | NC |
| CV Merge Point Clouds | Concatenate clouds + colours. The second cloud is OPTIONAL (unconnected -> cloud A alone, `count_b` 0), which is what lets a subgraph expose an optional cloud at its boundary | NC |
| CV Filter Points 3D By Mask | Keep cloud rows by mask, colours in lockstep | NC |
| CV Point Cloud Nearest Distance | Per-point NN distance via FLANN index (coverage metric) | CG |
| CV Point Cloud Normals | Nx3 -> Nx6 normals; mandatory in front of the ppf nodes | PS |
| CV Downsample Point Cloud | Thins a cloud onto a grid sized as a fraction of the bbox diagonal (`samplePCByQuantization`); goes in front of PPF, whose training is quadratic in the point count. Re-normalizes the averaged normals, which cv2 does not | PS |

## `convert.py` - bridges, literals, previews, cloud IO (68)

| Node | Does / used for | Class |
| --- | --- | --- |
| Image -> CV Array | IMAGE -> ndarray (BGR uint8, frame 0) | TB |
| Image Batch -> CV Batch | Whole IMAGE/MASK/LATENT batch -> 4-D ndarray | TB |
| CV Batch -> Image Batch | Inverse | TB |
| CV Array -> Image | ndarray -> IMAGE | TB |
| Mask -> CV Array | MASK -> single-channel ndarray | TB |
| CV Array -> Mask | ndarray -> MASK (min-max normalizes floats) | TB |
| CV Array -> Latent | Float ndarray -> LATENT | TB |
| Latent -> CV Array | LATENT -> ndarray | TB |
| * -> CV Array | Polymorphic bridge from any of the four | TB |
| CV Scalar | Constant vector for scalar-typed cv2 args | TB |
| CV Term Criteria | (type, max_count, epsilon) literal | TB |
| CV Warp Flags | Authors the flag bits for the generated `cv2_warpAffine` / `warpPerspective` / `warpPolar` / `remap` wrappers | TB |
| CV DFT Flags | Flag bits for `cv2_dft` / `cv2_idft` (e.g. DFT_COMPLEX_OUTPUT, DFT_SCALE \| DFT_REAL_OUTPUT) | TB |
| CV DCT Flags | Flag bits for `cv2_dct` / `cv2_idct` | TB |
| CV Optical Flow Flags | Flag bits for `cv2_calcOpticalFlowPyrLK` / `cv2_calcOpticalFlowFarneback` (each ignores the other's bit) | TB |
| CV Chessboard Flags | Flag bits for `cv2_findChessboardCorners` (classic vs SB variants) | TB |
| CV GEMM Flags | GEMM_1_T/2_T/3_T transposes for `cv2_gemm` | TB |
| CV SVD Flags | Flag bits for `cv2_SVDecomp` | TB |
| CV FloodFill Flags | Connectivity + mask-fill + FIXED_RANGE/MASK_ONLY for `cv2_floodFill` | TB |
| CV Threshold Flags | Method bits (BINARY/OTSU/TRIANGLE/...) for `cv2_threshold` / `cv2_thresholdWithMask` | TB |
| CV Grid Points | Planar 3D object-point grid for calibration | NC |
| CV Points | Point array from a Python literal | TB |
| CV Split Point | Point literal -> two numbers (the CV_TUPLE counterpart is CV Split Tuple) | TB |
| Parse Matrix | Delimited text -> 2-D matrix | TB |
| CV Camera Matrix | 3x3 pinhole K from fx/fy/cx/cy | NC |
| CV Cast Array | dtype conversion (with/without scaling) | TB |
| CV Reshape Array | Reshape without touching data (required before cv2 arithmetic on `(3,)` rows) | TB |
| CV Permute Axes | `np.moveaxis` (was the column-slicing workaround; use CV Slice Array now) | TB |
| CV Slice Array | Range out of ONE axis (crop a padded result back, take a column, subsample, reverse) | NC |
| CV To Blob | HWC -> NCHW blob | TB |
| CV Unstack Batch | Batched NPARRAY -> ComfyUI list | TB |
| CV Stack Batch | List -> batched NPARRAY | TB |
| CV Concat Arrays | Join along an existing axis, or a NEW last axis (replaces the absent `cv2.merge` wrapper) | NC |
| CV Index Batch | Pick one element of a batch | TB |
| CV Mosaic (RGB -> Bayer CFA) | Synthesize a Bayer CFA for demosaicing demos | NC |
| CV Roll (FFT Shift) | Cyclic shift of an array | NC |
| CV Array Statistic | Whole-array reduction per channel (median, percentile, ...) | NC |
| CV Random Seed | cv2.setRNGSeed + wildcard (`*`) passthrough (repeatable kmeans/randu/RANSAC) | PS |
| CV Reduce Array By Label | One row per label (mean/sum/count) | NC |
| CV Take By Index | Lookup-table indexing | NC |
| Preview CV Array | Preview raw ndarray (image / normalize / heatmap) | VIZ |
| Inspect CV Data | Shape/dtype/statistics summary (pair with Preview as Text) | VIZ |
| CV Array To Text | Array as a text table | VIZ |
| CV Array To Numbers | Array -> ComfyUI FLOAT list | TB |
| CV Numbers To Array | FLOAT list -> array | TB |
| CV Save Camera Params (JSON) | Write K + dist to JSON | TB |
| CV Load Camera Params (JSON) | Read them back | TB |
| CV Camera Pose To 3D View | OpenCV camera -> core LOAD3D_CAMERA (matrices smuggled through camera_info) | TB |
| CV Preview 3D (Calibrated Camera) | Own three.js viewer: full K + Brown-Conrady distortion, depth-map occlusion | VIZ + NC |
| CV 3D Object Transform | One number string places a model in BOTH lanes that carry a placement: the preview widget's `object_transform` and the 4x4 for CV Transform Points 3D (they live in different coordinate systems) | NC |
| CV Parse Object Transform | Reads an `object_transform` string back as a 4x4 - how the viewer gizmo's by-hand placement becomes data | NC |
| CV Mesh From 3D Model | Model file geometry -> vertices (Nx3) + triangle indices (Mx3), ready for RAPIID / ICP / PPF / normals | NC |
| CV Mesh Split Long Edges | Longest-edge bisection until no triangle edge exceeds `max_edge` (denser control points for cv2.rapid at ~half the triangles of uniform subdivision) | NC |
| CV Convert Axis Convention (3D) | Cloud between OpenCV (Y down, Z into the scene) and glTF/Three.js (Y up): negates Y AND Z, i.e. a 180-deg turn about X. Negating one alone is a REFLECTION nothing downstream catches. Nx6 clouds keep their normals. The cloud-level counterpart of the axis_convention on CV Mesh From 3D Model | NC |
| CV Mesh Vertex Normals | Mesh (pts3d + tris) -> the Nx6 cloud PPF/ICP want, normals taken from the triangle WINDING (area-weighted) so they carry a sign. Use instead of CV Point Cloud Normals whenever the cloud came from a mesh: a plane fit cannot orient a closed model and inverts the PPF pose (~180 deg) | NC |
| CV Rasterize Mesh | Software z-buffer render of a graph mesh through an OpenCV camera -> silhouette MASK + metric camera-space depth (shares the renderer with CV Preview 3D) | NC |
| CV Project Points (Sequence) | Projects one object-point set through a whole STACK of poses (Bx3x1 rvec/tvec, e.g. CV Rapid Track output); optional depth-test against a scene depth map | PS |
| CV Unproject Points | Pixel positions + known depth -> 3D camera-space points (the few-points inverse of projectPoints; CV Depth to 3D Points does the whole image) | NC |
| CV Text To Array | Typed text -> (N,) STRING array for caption/label inputs | TB |
| CV Annotate Points On Model | Orbit a loaded model and click its surface to drop 3D anchors (SHIFT+click removes); points come out in the model's own frame | NC + VIZ |
| CV Array Shape | Spatial dims + channels as INTs | TB |
| CV DType | dtype name as STRING | TB |
| CV Color From Hex | COLOR hex -> tuple string | TB |
| CV Depth to 3D Points | Back-project a depth map to a coloured cloud | NC |
| CV Write PLY (Point Cloud) | Binary PLY export | TB |
| CV Filter Point Cloud | Distance/percentile outlier removal | NC |
| CV Flip Axis (3D) | Negate one axis (mirrored stereo reconstructions) | NC |
| CV Build Information | OpenCV build banner with path redaction; reports Eigen/non-free/OpenCL/CUDA availability | TB |

## `tuples.py` - the composite-value socket type (2)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Tuple | Authors ONE composite cv2 value (Size / Point / 3-D point / Rect) and emits it on the `CV_TUPLE` socket, plus the same numbers as an NPARRAY vector and as a `'(a, b)'` literal. The atomic alternative to wiring split components one by one - and the only way to set a composite input on a subgraph INSTANCE, since a promoted input is a bare socket with no widget | TB |
| CV Split Tuple | The reverse: one composite in (as a SOCKET - a widget there would be CV Tuple again), its components out as INT or FLOAT (generalises CV Split Point) | TB |

## `dnn.py` - DNN models and generic inference pipeline (25)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV YuNet Face Detect | `cv2.FaceDetectorYN` faces + 5 landmarks | CG |
| CV MediaPipe Palm Detect | Palm SSD + anchor decoding | CG + NC |
| CV MediaPipe Hand Pose | 21 hand keypoints per palm | CG + NC |
| CV WeChat QR Detect | `wechat_qrcode_WeChatQRCode` | CG |
| CV Colorize Model | Learned colorization (Zhang et al.) with the cluster-centre hull | CG + NC |
| CV DNN Blob From Image | `blobFromImages` with typed options | PS |
| CV DNN Forward | Load ONNX + forward (single output) | CG |
| CV DNN Images From Blob | NCHW blob -> IMAGE batch | TB |
| CV DNN Mask Output | Segmentation blob -> MASK batch | NC |
| CV EAST Text Detect | EAST scene text + geometry decoding | CG + NC |
| CV Feature Extract (Model) | DISK / SuperPoint-style ONNX keypoints+descriptors; LightGlue preprocessing (0..1, RGB or gray per the model, long side 1024) | CG + NC |
| CV Match Features (Model) | Learned matcher ONNX over two keypoint sets; normalizes keypoints the way LightGlue expects | CG + NC |
| CV DNN Letterbox | Aspect-preserving resize + pad, with the undo scale | NC |
| CV DNN Forward All | Load `.onnx`/`.tflite`, forward, return ALL outputs (engine choice) | CG |
| CV DNN Pick Output | Select an output by index / name / shape | TB |
| CV Load Labels | Class-labels text file -> list | TB |
| CV YOLO Detect Decode | Decode 4 YOLO head layouts -> boxes/scores/classes | NC |
| CV YOLO Seg Masks | Prototype x coefficients -> one MASK per detection | NC |
| CV Class Scores Decode | Classification vector -> top-k labels + scores | NC |
| CV DNN Tokenizer | `cv2.dnn.Tokenizer_load` encode/decode (wants the config.json path) | CG |
| CV LLM Generate | Decoder-only greedy generation, KV-cache or full-sequence, folder-scoped model resolution | CG + NC |
| CV VLM Generate | Vision-language prompt answering over a multi-part export | CG + NC |
| CV Seq2Seq Generate | Encoder-decoder generation | CG + NC |
| CV SFace Embeddings | `FaceRecognizerSF` alignCrop + 128-D identity vectors | CG |
| CV Embedding Match | All-pairs cosine/L2 matching of two embedding sets | NC |

## `contrib.py` - contrib class APIs (17)

All of this module exists because the entry point is a class or `create*` factory.

| Node | Does / used for | Class |
| --- | --- | --- |
| CV White Balance (xphoto) | Simple/GrayWorld/Learning white balance | CG |
| CV Superpixels | SLIC / SEEDS / LSC regions + label map | CG |
| CV Saliency | Spectral-residual / fine-grained / objectness | CG |
| CV Background Subtract (Batch) | MOG/MOG2/KNN/GMG/CNT learned over a clip | CG |
| CV Image Hash Compare | pHash/aHash/blockMean... same-picture decision | CG |
| CV Stereo Disparity (WLS filtered) | SGBM + the bound WLS filter + confidence | CG |
| CV Disparity Filter (WLS) | WLS over ANY matcher's map; derives ROI and radius the generic factory gets wrong | CG + NC |
| CV Disparity Interpolate (Edge-Aware) | RIC / EPIC locally-affine hole filling with an escalating retry post-condition | CG + NC |
| CV ICP Register (Point Clouds) | `ppf_match_3d.ICP` refinement; catches cv2's 1e10 divergence sentinel, which `registerModelToScene` reports alongside `retval == 0` | CG |
| CV PPF Pose Estimation | PPF global pose (ranked by votes; applies normals internally) | CG |
| CV Rapid Track (Sequence) | Tracks a known 3D mesh through a batch with `cv2.rapid`: control points along the projected silhouette, gradient search, PnP pose refine; warm-started per frame | CG |
| CV HFS Segmentation (hierarchical feature selection) | Hierarchical feature selection segmentation | CG |
| CV Optical Flow (TV-L1) | TV-L1 dense flow | CG |
| CV Optical Flow (RLOF, Dense) | Dense RLOF optical flow | CG |
| CV Optical Flow (DeepFlow / PCAFlow) | Two classic dense flows via free functions on abstract classes | CG |
| CV Detect Line Segments (FLD) | Fast Line Detector; `canny_aperture_size=0` accepts an edge map | CG |
| CV Edge Drawing | Edge Drawing segments + lines + ellipses (normalizes the packed 6-column row) | CG + NC |

## `ml.py` - training and classic features (9)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Random Points | Synthetic labelled point clouds | NC |
| CV Coordinate Grid | Dense (x, y) meshgrid for decision-boundary rendering | NC |
| CV Stack Feature Classes | Build a labelled training matrix from per-class features | NC |
| CV HOG Features | `cv2.HOGDescriptor` per frame | CG |
| CV Deep Features | Frozen ONNX backbone as a fixed extractor (ENGINE_CLASSIC pinned; raises where the build has no such engine, e.g. OpenCV 5.1) | CG |
| CV Train Classifier | `cv2.ml` train + evaluate + query in one execute | CG |
| CV Save Classifier | Trained model -> gzipped `.yml.gz` in models/cvml | CG + TB |
| CV Predict Classifier | Load + predict in one execute (cache-safe) | CG |
| CV Cascade Detect | Viola-Jones `CascadeClassifier` (4.x XMLs; OpenCV 5 deleted `data/`) | CG |

## `segmentation.py` (7)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV GrabCut | Rect/mask-seeded foreground extraction | PS |
| CV Color Range | inRange band around a colour, chosen colour space | PS |
| CV Color Range From Sample | Learns the band from a sampled region | NC |
| CV Histogram | calcHist with named channels/ranges | PS |
| CV Back Project | Histogram back-projection (tracking) | PS |
| CV Temporal Reduce (Background Plate) | Reduce a clip to one plate (median/mean/min/max over time) | NC |
| CV Background Model (Update) | Running-average background subtraction step | NC |

## `remap.py` - composable map generators (11)

Maps compose in NPARRAY space and collapse into ONE `cv2.remap` pass.

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Identity Map | Chain start: the do-nothing map | NC |
| CV Radial Lens Map | ImageMagick-style polynomial barrel/pincushion | NC |
| CV Cylinder Map | Cylindrical wrap / unwrap | NC |
| CV Pinch/Stretch Map | Per-axis pinch or stretch | NC |
| CV Wave Map | Sinusoidal row/column displacement | NC |
| CV Displacement Map | Displace by another image's brightness | NC |
| CV Flow Map | Add a dense flow field to the map | NC |
| CV Homography Map | Fold a 3x3 projective transform into the map | PS |
| CV Polynomial Radial Map | Arbitrary radial polynomial in r | NC |
| CV Relative Identity Map | (0,0)-based map for relative-mode remaps | NC |
| CV Vector Displacement Map | Add a per-pixel (dx, dy) field | NC |

## `hdr.py` (5)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV HDR Exposure Fusion (Mertens) | Bracket -> displayable image, no response curve | CG |
| CV HDR Merge to Radiance | Debevec/Robertson response + radiance map | CG |
| CV HDR Tonemap | Drago/Mantiuk/Reinhard tonemapping | CG |
| CV Align MTB | Median Threshold Bitmap alignment of a batch | CG |
| CV Align MTB To Reference | Same, against a chosen reference frame | CG + NC |

## `quality.py` (4)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Quality Compare Batch (cv2.quality) | MSE/PSNR/SSIM/GMSD over a batch vs one reference + best index + quality maps | CG |
| CV Quality BRISQUE (no-reference) | No-reference quality score | CG |
| CV Quality BRISQUE Features | The 36 NSS features behind it | CG |
| CV Quality Compare (cv2.quality) | Single-image score against a reference | CG |

## `ccm.py` - colour correction (5)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Detect Color Checker | Macbeth/Vinyl/DigitalSG chart detection | CG |
| CV Create CCM Model | Fit a ColorCorrectionModel | CG |
| CV Apply Color Correction | Apply a fitted model | CG |
| CV Get CCM Matrix | Read the fitted matrix out | CG |
| CV Get CCM Loss | Read the fit loss out | CG |

## `codes.py` - QR / barcode (4)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV QR Encode | Text -> QR image (`QRCodeEncoder`) | CG |
| CV QR Detect | Detect + decode every QR (`QRCodeDetector`) | CG |
| CV QR Decode At Points | Decode inside quads found elsewhere | CG |
| CV Barcode Detect | 1-D product barcodes (`BarcodeDetector`) | CG |

## `plots.py` - data visualization (4)

| Node | Does / used for | Class |
| --- | --- | --- |
| CV Plot 2D (cv2.plot) | `Plot2d_create` raster plot (needs CV_64F; y-axis inverted by default) | CG + VIZ |
| CV Chart Series | Interactive bar/line/area/step (full ECharts option built in Python) | NC + VIZ |
| CV Chart Scatter | Interactive x/y scatter | NC + VIZ |
| CV Chart Matrix | Interactive labelled heatmap | NC + VIZ |

## `colorize.py` (1)

| Node | Does / used for | Class |
| --- | --- | --- |
| Colorize Superpixels (FLANN) | Transfers colour from a reference image by matching superpixel statistics through a FLANN index | NC |

---

## Supporting modules with no nodes

`boxpolicy.py` (crop geometry + policy string), `polytype.py` (polymorphic
socket resolve/restore + lossless crop/paste), `imageutils.py`,
`render3d.py` (software GLB rasterizer + depth PNG codec behind
`CV Preview 3D` / `CV Rasterize Mesh`),
`buildinfo.py` (path-redacted `getBuildInformation` banner behind
`CV Build Information`), `css_colors.py` (CSS named-colour table behind
`Find Absent Color`), `scalars.py` (the one Scalar parser shared by the
drawing nodes, `CV Scalar` and the generated wrappers), `shapepose.py`
(canonical orientation/chirality shared by Shape Moments and Image Moments),
`enums.py` (named constant dropdowns, resolved against the installed cv2),
`param_docs.py` / `param_tables.py` / `return_docs.py` (documentation and
large constant tables for the generated wrappers), `workerutils.py`,
`lowlevel.py` and the two generated registries. The `workers/` package holds
the interruptible subprocess workers (generic slow cv2 calls, DNN inference
including the LLM/VLM generation loops, both stitchers); `generator/` builds
the registries from the cv2 type stubs; `web/` carries the frontend JS
widgets.

## Rough tally by class

| Class | Approx. count |
| --- | --- |
| Class-gated workaround (CG) | ~91 |
| Non-OpenCV (NC) | ~118 |
| Pipeline simplification (PS) | ~51 |
| Type bridge / IO (TB) | ~50 |
| Visualization (VIZ) | ~31 |

(Counts overlap where a node carries two tags.)
