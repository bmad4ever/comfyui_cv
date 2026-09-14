# Subgraph blueprints

The 65 JSON files in `subgraphs/` are **blueprints**: ready-made compositions of
this pack's nodes for steps that come up often enough to be worth reusing -
each one a small graph you can drop in and open up, rather than a black box.

Each file is a one-node workflow: open it in ComfyUI and you get a single
subgraph instance plus its definition. Copy that node into your own graph, or
load the blueprint and build around it.

**Three things to know before you use one.**

1. **A copy is a fork.** Dropping a blueprint into a workflow embeds a *copy* of
   its definition. Later edits to the file do not reach workflows that already
   carry it, and edits inside a workflow do not reach the file. Several shipped
   workflows therefore carry a slightly renamed copy of a blueprint (`Sobel 2D`
   in workflow 03, `Wiener Filter` in workflow 19, `CV Match Template` in
   workflow 37, and so on); the entries below point at those too.
2. **Subgraph inputs are bare sockets - no tooltips, no widgets.** Whatever
   values are set on the nodes *inside* apply unless you promote and wire them.
   This page is the tooltip layer the canvas cannot show: what each input means
   and what it expects. Where a constant is baked in rather than promoted, the
   entry says so.
3. **A blueprint can nest another blueprint.** `CV Wiener Filter (deblur, color)`
   contains `CV Wiener Filter (deblur)`; the motion variants contain
   `CV Edge Taper (tanh window)`; `CV Document Scan (SAM seg)` contains
   `Image Segmentation (SAM3)`. Opening the outer file gives you all of them.

Cosmetic quirk: a dozen blueprints still carry the `.json` suffix inside their
definition `name` (`CV Sharpen.json`), so that is what the canvas shows. It has
no effect on what the blueprint does.

Model-backed blueprints keep a download note on the blueprint's own canvas (URL,
file size, target folder). That note sits *outside* the definition, so it never
travels into a workflow that instantiates the subgraph. The same URLs are
repeated here.

---

## Index

| Blueprint | Does | Reference workflow |
| --- | --- | --- |
| [cv2.add (scalar)](#cv2add-scalar) | Add a literal scalar to an array | *(helper)* |
| [cv2.divide (by scalar)](#cv2divide-by-scalar) | Divide by a literal scalar | *(helper)* |
| [cv2.multiply (by scalar)](#cv2multiply-by-scalar) | Multiply by a literal scalar | *(helper)* |
| [cv2.morphology (simple)](#cv2morphology-simple) | Morphology with an ellipse kernel from two ints | `20_color_range_playground` |
| [CV Clip Array \[lo, hi\]](#cv-clip-array-lo-hi) | Clamp an array between two literals | *(helper)* |
| [CV Adjust Range \[0, N\] to \[0, 1\]](#cv-adjust-range-0-n-to-0-1) | Divide by the array's own max | `15_hdr_playground` |
| [CV Merge Channels (3)](#cv-merge-channels-3) | Three planes -> one 3-channel array | *(helper)* |
| [CV Merge Channels (4)](#cv-merge-channels-4) | Four planes -> one 4-channel array | *(helper)* |
| [CV Split Channels (3)](#cv-split-channels-3) | 3-channel array -> three planes | *(helper)* |
| [CV Split Channels (4)](#cv-split-channels-4) | 4-channel array -> four planes | *(helper)* |
| [CV Split Image into BGR](#cv-split-image-into-bgr) | IMAGE -> B, G, R planes | *(helper)* |
| [CV Sobel 2D](#cv-sobel-2d) | dx and dy in one node, shared settings | `03_debug_visualization_playground` |
| [CV Sharpen](#cv-sharpen) | Unsharp mask (amount + radius) | *(none)* |
| [CV Optimal DFT](#cv-optimal-dft) | Pad to `getOptimalDFTSize`, then DFT | `19_fourier_playground` |
| [CV Add Gaussian noise](#cv-add-gaussian-noise) | Additive Gaussian noise, both signs | `09_denoise_playground` |
| [CV Add salt & pepper](#cv-add-salt--pepper) | ~3% white and ~3% black impulses | `09_denoise_playground` |
| [CV Power Spectrum (PSD)](#cv-power-spectrum-psd) | Squared DFT magnitude and log(1+PSD), both shifted | `19_fourier_playground` |
| [CV Detect Spectral Peaks](#cv-detect-spectral-peaks) | Sigma-threshold spectrum peaks -> notch list | `19_fourier_playground` |
| [CV Detect Spectral Peaks (K-Means)](#cv-detect-spectral-peaks-k-means) | Same, knob-free (2-means on the noise floor) | `19_fourier_playground` |
| [CV Notch Reject Filter (periodic noise)](#cv-notch-reject-filter-periodic-noise) | Punch symmetric holes in the spectrum | `19_fourier_playground` |
| [CV Edge Taper (tanh window)](#cv-edge-taper-tanh-window) | Sigmoid border window, kills DFT wrap ringing | `19_fourier_playground` |
| [CV Wiener Filter (deblur)](#cv-wiener-filter-deblur) | Disc-PSF Wiener deconvolution, one channel | `19_fourier_playground` |
| [CV Wiener Filter (deblur, color)](#cv-wiener-filter-deblur-color) | The same, per BGR channel | `19_fourier_playground` |
| [CV Wiener Filter (motion deblur)](#cv-wiener-filter-motion-deblur) | Line-PSF (LEN/THETA) Wiener, one channel | `19_fourier_playground` |
| [CV Wiener Filter (motion deblur, color)](#cv-wiener-filter-motion-deblur-color) | The same, per BGR channel | `19_fourier_playground` |
| [CV Anisotropic Segmentation (GST)](#cv-anisotropic-segmentation-gst) | Gradient structure tensor -> coherency + orientation | `27_anisotropic_segmentation` |
| [CV K-Means Colors](#cv-k-means-colors) | Colour quantization + coverage-sorted palette | `24_kmeans_clusters` |
| [CV K-Means Mask Clusters](#cv-k-means-mask-clusters) | Group a mask's blobs by measured properties | `24_kmeans_clusters` |
| [CV K-Means Points](#cv-k-means-points) | Cluster a point set, empty-safe | `24_kmeans_clusters` |
| [CV Snap Mask To Labels](#cv-snap-mask-to-labels) | Round a rough mask onto superpixel edges | `25_saliency_superpixels` |
| [CV Superpixel Paint](#cv-superpixel-paint) | Superpixel posterize + inked boundaries | `16_npr_filters_playground` |
| [CV Cartoonify](#cv-cartoonify) | Bilateral flatten x adaptive-threshold ink | `16_npr_filters_playground` |
| [CV Clean Plate Matte](#cv-clean-plate-matte) | Alpha from one take + one empty plate | `exercise_clean_plate_matting` |
| [CV Clean Plate Matte (RGBA)](#cv-clean-plate-matte-rgba) | The same, IMAGE in / RGBA out | `exercise_clean_plate_matting` |
| [CV Clean Plate Matte (Align + Cleanup, RGBA)](#cv-clean-plate-matte-align--cleanup-rgba) | Plus plate registration and matte cleanup | `exercise_clean_plate_matting` |
| [CV Plate Align (Geometric + Photometric)](#cv-plate-align-geometric--photometric) | Register and exposure-match a plate to a take | `exercise_clean_plate_matting` |
| [CV Matte Cleanup (Noise Floor + Speckle)](#cv-matte-cleanup-noise-floor--speckle) | Dead-zone, despeckle, guided-filter an alpha | `exercise_clean_plate_matting` |
| [CV Triangulation Matte](#cv-triangulation-matte) | Classic two-background triangulation matte | *(none)* |
| [CV Exposure Match (Masked Per-Channel)](#cv-exposure-match-masked-per-channel) | Match mean and sigma per channel inside a mask | `exercise_match_two_views_over_a_mask` |
| [CV Multi-Band Blend (Laplacian pyramid)](#cv-multi-band-blend-laplacian-pyramid) | 3-level Laplacian blend along a soft seam | `exercise_seam_pipeline` |
| [CV Paste Through Homography](#cv-paste-through-homography) | Warp content into a scene, premultiplied | `47_paste_into_scene` |
| [CV Match Template](#cv-match-template) | matchTemplate + best-match box | `37_match_template` |
| [CV Find Planar Object (Reference in Scene)](#cv-find-planar-object-reference-in-scene) | Detect a planar object, H + quad + footprint | `36_feature_matching_playground` |
| [CV Hue Track Step](#cv-hue-track-step) | One frame of meanShift back-projection tracking | `40_kalman_tracking` |
| [CV Rapid Pose Refine](#cv-rapid-pose-refine) | RAPID edge-search pose refinement | `41_rapid_tracking_playground` |
| [CV Pick Point On Mesh](#cv-pick-point-on-mesh) | Clicked pixel -> 3D point in model space | `63_ar_model_annotation` |
| [CV Annotate Model](#cv-annotate-model) | Project anchors, hide occluded, draw captions | `63_ar_model_annotation` |
| [CV Draw 3D Measurement](#cv-draw-3d-measurement) | Dimension line + distance between two 3D points | `63_ar_model_annotation` |
| [CV Stabilize Frame (Align to Reference)](#cv-stabilize-frame-align-to-reference) | ECC or feature alignment with a `found` gate | `exercise_image_registration_ecc` |
| [CV Trajectory Plot (Top-Down)](#cv-trajectory-plot-top-down) | XZ plot of estimate vs truth, shared extent | `exercise_visual_odometry` |
| [CV Transform Points (Scale / Rotate / Translate)](#cv-transform-points-scale--rotate--translate) | Build and apply a 4x4 from Euler angles | *(none)* |
| [CV Polygon IoU (convex)](#cv-polygon-iou-convex) | IoU of two convex polygons | `31_convex_geometry_playground` |
| [CV Background Plate (Video)](#cv-background-plate-video) | Temporal median of a clip = the empty scene | `28_background_subtract_bgsegm` |
| [CV Foreground Mask (Video)](#cv-foreground-mask-video) | Per-frame foreground masks, adaptive or static | `exercise_background_subtraction` |
| [CV Flow Interpolate (Video)](#cv-flow-interpolate-video) | Flow-warped middle frames, 2N-1 out | `exercise_optical_flow_interpolation_video` |
| [CV Motion Segmentation (Flow Residual)](#cv-motion-segmentation-flow-residual) | Flow minus camera model -> moving object | `exercise_optical_flow_motion_segmentation` |
| [CV Dense Flow (DNN Model)](#cv-dense-flow-dnn-model) | RAFT ONNX optical flow with a NaN retry | `76_raft_optical_flow_dnn` |
| [CV Frame Interpolate (DNN Model)](#cv-frame-interpolate-dnn-model) | RIFE ONNX middle frame at time `t` | `77_rife_frame_interpolation` |
| [CV Frame Interpolate (Video, DNN Model)](#cv-frame-interpolate-video-dnn-model) | RIFE over a whole clip, integer multiplier | `77_rife_frame_interpolation` |
| [CV Image To Image (DNN Model)](#cv-image-to-image-dnn-model) | Any image->image ONNX, one frame | `73_dnn_style_transfer` |
| [CV Image To Image (Batch, DNN Model)](#cv-image-to-image-batch-dnn-model) | The same, one batched forward pass | `74_batch_dnn_style_transfer` |
| [CV YOLO Detection](#cv-yolo-detection) | YOLOv4 ONNX -> boxes, scores, COCO labels | `69_yolov4_detection` |
| [CV YOLO-Seg Inference](#cv-yolo-seg-inference) | YOLO-seg ONNX -> boxes + instance masks | `70_yolo_seg` |
| [CV Document Scan (SAM seg)](#cv-document-scan-sam-seg) | SAM3 segment -> quad -> true-aspect rectify | `exercise_document_perspective` |
| [CV Color Correct](#cv-color-correct) | Detect a colour checker, fit and apply a CCM | `11_color_correction_basic` |

---

## Array and channel helpers

These are the "no magic literal on a wire" blueprints: they wrap a cv2 call
whose second operand should be a typed constant, so the constant lives on a
widget instead of on an extra node you re-create every time. None is
instantiated by a shipped workflow - they are there for graphs you build.

### cv2.add (scalar)

`CV_ScalarArray` + `cv2.add`: adds a literal such as `(10, 0, 0)` to an
array, with the optional `mask` and `dtype` of the real call.

* **In** `src1` (NPARRAY/IMAGE/MASK/LATENT); `values` STRING; `dtype`; `mask`
* **Out** `nparray` (same polytype as the input)
* The scalar is a STRING tuple, so `(5,)` broadcasts to every channel while
  `(5, 0, 0)` touches only the first. A 4-tuple is what cv2 actually wants; the
  shorter forms are padded.
* Saturating on uint8 - a uint8 250 + 10 is 255, never a wrap to 4.

### cv2.divide (by scalar)

`cv2.divide` with the divisor as a literal. Same shape as the adder.

* **In** `src1`; `values` STRING; `dtype` - **Out** `nparray`
* Division by zero yields 0 in OpenCV, not an error and not an inf.
* OpenCV 5 is stricter about mixed dtypes in `arithm_op` than 4.x was. Cast to
  float32 first if the two operands are not the same depth.

### cv2.multiply (by scalar)

`cv2.multiply` with a literal factor.

* **In** `src1`; `values` STRING; `dtype` - **Out** `result`
* Use it for per-channel gain (`(1.0, 0.9, 1.2)` is a crude white balance).
* uint8 saturates. For a gain that must not clip, cast to float32 first.

### cv2.morphology (simple)

The three-node ritual - build an ellipse kernel, pick an op, run
`morphologyEx` - collapsed into one node with two integer sizes.

* **In** `src`; `op` (grow/shrink/open/close); `ksize_x` INT; `ksize_y` INT
* **Out** `result` (same polytype as the input)
* The kernel is always `MORPH_ELLIPSE`. For a rectangular or cross kernel, build
  it yourself with `getStructuringElement`.
* Even `ksize` values are legal but shift the result half a pixel; prefer odd.
* **Reference** `20_color_range_playground.json` (twice: opening then closing an
  `inRange` mask). Also used in `16`, `23` and `exercise_background_subtraction`.

### CV Clip Array [lo, hi]

`cv2.max(src, lo)` then `cv2.min(., hi)` - a `numpy.clip` with both bounds as
STRING literals.

* **In** `src`; `lo` STRING; `hi` STRING - **Out** `nparray`
* The point is float pipelines: a uint8 pipeline clips for free, a float32 one
  does not, and an out-of-range float silently ruins the next `convertTo`.
* Bounds use the same tuple syntax as the arithmetic helpers.

### CV Adjust Range [0, N] to [0, 1]

Divides an array by its own maximum, so any positive range lands in `[0, 1]`.

* **In** `nparray` - **Out** `nparray`
* Uses `CV_ArrayStat` for the max, with a threshold guard so an all-zero
  array does not divide by zero.
* This is **not** a min-max stretch: the low end is not moved, and negative
  values survive as negatives.
* **Reference** `15_hdr_playground.json` carries its own copy under the same
  name (tone-mapped HDR output is `[0, N]`).

### CV Merge Channels (3)

One `CV_ConcatArrays` on the channel axis.

* **In** `channel 0..2` - **Out** `nparray`
* All three planes must share height, width and dtype. A `(H, W)` plane and a
  `(H, W, 1)` plane are both accepted.

### CV Merge Channels (4)

The same with four planes - the one to use for BGRA.

* **In** `channel 0..3` - **Out** `nparray`

### CV Split Channels (3)

Three `cv2.extractChannel` calls with indices 0, 1, 2.

* **In** `src` - **Out** `nparray`, `nparray_1`, `nparray_2`
* Outputs are 2-D `(H, W)` planes, not `(H, W, 1)`.
* The input must really have 3 channels; `extractChannel` throws otherwise.

### CV Split Channels (4)

The same with four outputs.

* **In** `src` - **Out** `nparray` .. `nparray_3`

### CV Split Image into BGR

`CV Split Channels (3)` with the channel order spelled out, plus an
`Image -> CV Array` in front so an IMAGE can be wired straight in.

* **In** `src` (NPARRAY/IMAGE/MASK) - **Out** blue, green, red
* The naming assumes OpenCV's BGR order. A ComfyUI IMAGE is RGB, and the
  conversion node is what flips it - bypass it and the labels lie.

### CV Sobel 2D

Both derivatives in one node, sharing `ksize`, `scale`, `delta`, `ddepth` and
`borderType` so the pair cannot drift apart.

* **In** `src`; `dx` INT; `ksize`; `scale`; `delta`; `borderType`; `ddepth`
* **Out** `nparray` (d/dy), `nparray_1` (d/dx)
* **`ddepth` must be a signed type** (`CV_32F`, `CV_16S`). At `CV_8U` every
  negative gradient clips to 0 and one edge of every object disappears.
* The promoted `dx` input is a leftover from the original node; the two Sobel
  calls inside set their own orders.
* **Reference** `03_debug_visualization_playground.json`, as an inline copy
  named `Sobel 2D`.

### CV Sharpen

Unsharp mask: `addWeighted(src, 1 + amount, blur(src), -amount, 0)`.

* **In** `image` IMAGE; `amount` FLOAT; `radius` FLOAT - **Out** `IMAGE`
* `radius` is the Gaussian sigma, with `ksize = (0, 0)` so OpenCV derives the
  kernel - a big radius is therefore expensive, not just soft.
* `amount` above ~1.5 halos high-contrast edges. There is no threshold input, so
  noise is sharpened along with detail.
* No shipped workflow uses it.

### CV Optimal DFT

Pads an image with `BORDER_CONSTANT` up to `cv2.getOptimalDFTSize` on both axes,
then runs the DFT.

* **In** `image` - **Out** `spectrum`
* The padding makes the transform several times faster on awkward sizes, but the
  spectrum is of the **padded** image, so the frequency axes are scaled by the
  padded dimensions. Crop back before comparing against the source.
* The rest of the Fourier family (`CV Power Spectrum`, `CV Notch Reject Filter`)
  crops to an **even** size instead, because they build a matching mask. The two
  conventions do not mix.
* **Reference** none instantiates it; `19_fourier_playground.json` is the
  workflow that covers the DFT family.

---

## Noise and degradation

### CV Add Gaussian noise

Fills a same-shape canvas with `cv2.randn` at mean 128, sigma 25, then adds it
and subtracts 128 so the noise can go both ways inside a uint8 pipeline.

* **In** `image` - **Out** `noisy`, `noise`
* Sigma is **baked in at 25** on the `randn` node - open the blueprint to change
  it, it is not promoted.
* `cv2.randn` draws from cv2's one global RNG. Put a `CV Random Seed` node in the
  data path if you need the same noise twice - a seed widget alone will not do
  it, because execution order follows the links.
* **Reference** `09_denoise_playground.json`, inline copy `Add Gaussian noise`; a
  grayscale variant appears in the classifier exercises.

### CV Add salt & pepper

A uniform random field thresholded at both ends: values above 247 become salt,
below 8 become pepper - about 3% of each.

* **In** `image` - **Out** `noisy`, `salt`, `pepper`
* The two thresholds are baked in, and they are the *only* density control.
* Salt is painted with `bitwise_or` (so it whitens every channel) and pepper with
  `bitwise_and` on the inverse - correct for uint8, wrong for float data outside
  `[0, 255]`.
* **Reference** `09_denoise_playground.json`, inline copy `Add salt & pepper`
  (the median filter versus everything else).

---

## Frequency domain

All the spectral blueprints crop the image to an **even** height and width first
(`rows & -2`), because the mirror symmetry they rely on is only exact on even
dimensions. Feed them a single channel.

### CV Power Spectrum (PSD)

Squared DFT magnitude, both raw and `log(1 + PSD)`, each `fftshift`-ed to put DC
in the middle and stretched to `0..255` for viewing.

* **In** `image` - **Out** `psd`, `log_psd`
* **The DC term is zeroed before the stretch.** DC is orders of magnitude above
  everything else and would flatten the rest to black. So `psd` is a *display*
  product, not a measurement - do not integrate it.
* `log_psd` is the one to read peaks off; `psd` is almost always saturated.
* **Reference** `19_fourier_playground.json`.

### CV Detect Spectral Peaks

Thresholds the spectrum at `mean + sigmas * std`, finds connected components,
drops the DC blob, and emits an `Nx3` table of `(x, y, radius)` notches.

* **In** `spectrum`; `sigmas` FLOAT; `min_area` INT; `dc_radius` FLOAT;
  `radius` FLOAT - **Out** `notches` (Nx3), `count` INT
* Feed it `log_psd`, not `psd` - on a linear spectrum the mean and std are
  dominated by the low frequencies and nothing crosses the threshold.
* `sigmas` is the knob and it is image-dependent: too low and the whole
  low-frequency plateau counts as peaks. `dc_radius` must be big enough to
  swallow the central blob, or the filter notches out the image's own brightness.
* `radius` is written into every row - one size for all notches.
* **Reference** `19_fourier_playground.json` (side by side with the k-means
  detector).

### CV Detect Spectral Peaks (K-Means)

The knob-free sibling. It estimates the noise floor locally (box mean), measures
prominence, uses median/MAD to set an expected-maximum threshold over
`rows*cols` pixels, then runs **2-means on (peak prominence, area)** to split
genuine spikes from the floor.

* **In** `spectrum`; `dc_radius` FLOAT; `radius` FLOAT
* **Out** `notches` (Nx3), `count` INT
* No `sigmas`, no `min_area` - that is the point. It survives images where the
  sigma threshold has no good setting.
* It pads the feature table with two dummy rows because `cv2.kmeans` asserts on
  fewer samples than clusters, and drops them afterwards. Zero peaks is valid.
* Both feature columns are divided by their own spread before clustering, so
  area and prominence count equally.
* `cv2.kmeans` draws from the global RNG; the blueprint seeds it internally for
  repeatability.
* **Reference** `19_fourier_playground.json`.

### CV Notch Reject Filter (periodic noise)

Builds an all-ones filter, punches a disc at every notch, then **mirrors the
mask** (flip plus a one-pixel roll, on both axes) so each notch is removed
together with its conjugate, `ifftshift`s it into DFT order and multiplies.

* **In** `image`; `notches` (Nx3, from either detector) - **Out** `image`,
  `filter`
* The mirroring is what keeps the result real-valued. Hand-authored notch
  coordinates that ignore the conjugate produce ghosting.
* Output is saturated to `0..255` and then stretched - **saturate before
  stretch** is deliberate; the other order lets one outlier pixel crush the
  contrast of everything else.
* `filter` is the mask actually applied; preview it to check the notches landed
  where you meant. Published notch coordinates from a tutorial usually do not
  fit your copy of the image.
* **Reference** `19_fourier_playground.json` (twice - one per detector).

### CV Edge Taper (tanh window)

A separable sigmoid window `w(y) * w(x)` that fades the image to zero at the
border, so the DFT does not see the wrap-around discontinuity as a giant edge.

* **In** `image`; `gamma` FLOAT (flat-top width); `beta` FLOAT (roll-off
  softness) - **Out** `image`, `window`
* Algebraically a `tanh` window, written out as `1/(1+exp(2q))` terms.
* Without it every Wiener deconvolution shows ringing that follows the frame
  edges rather than the subject - the classic "picture frame" artefact.
* `window` is the mask, previewable; multiply your own data by it if you want the
  taper without the filter.
* **Reference** `19_fourier_playground.json`. Also nested inside both
  motion-deblur blueprints.

### CV Wiener Filter (deblur)

Out-of-focus deconvolution: build a **disc** PSF of the given radius, pad to
`getOptimalDFTSize`, and divide in the frequency domain by `|H|^2 + 1/SNR`.

* **In** `image` (one channel); `radius` INT; `value` FLOAT (the SNR)
* **Out** `image` (saturated uint8), `raw` (before the stretch)
* Blueprint defaults: `radius = 6`, `SNR = 750`.
* **`radius` is resolution-bound.** It is the blur disc in *this image's* pixels;
  a radius copied from a tutorial whose photo was 9x larger will be ~9x too big.
  Measure it on your own image.
* One channel only - use the colour wrapper for a colour image.
* **Reference** reached through `CV Wiener Filter (deblur, color)` in
  `19_fourier_playground.json`.

### CV Wiener Filter (deblur, color)

Splits BGR, runs the mono deblur on each channel, restacks, then normalizes
**all three channels together** before saturating.

* **In** `image`; `radius` INT; `value` FLOAT - **Out** `image`
* Defaults `radius = 6`, `SNR = 250`.
* The joint normalize is what keeps the colour balance; normalizing per channel
  tints the result.
* **Nests** `CV Wiener Filter (deblur)`, three times.
* **Reference** `19_fourier_playground.json`.

### CV Wiener Filter (motion deblur)

The same Wiener core with a **line** PSF: a degenerate ellipse of axes
`(0, LEN/2)` rotated by `90 - THETA`, plus an edge taper on the input.

* **In** `image` (one channel); `LEN` INT (smear length, px); `THETA` FLOAT
  (smear angle, deg); `SNR` FLOAT - **Out** `image`, `raw`
* Working values: `LEN = 21`, `THETA = 15`, `SNR = 50`.
* `THETA` is measured the way the OpenCV tutorial measures it; the internal
  `90 - THETA` is the conversion to the ellipse's own rotation. Do not
  pre-convert.
* `LEN` is in this image's pixels - same resolution warning as the disc variant.
* **Nests** `CV Edge Taper (tanh window)`.
* **Reference** `19_fourier_playground.json`.

### CV Wiener Filter (motion deblur, color)

Per-channel motion deblur with the same joint normalize as the other colour
wrapper.

* **In** `image`; `LEN`; `THETA`; `SNR` - **Out** `image`
* **Nests** `CV Wiener Filter (motion deblur)`, and through it the edge taper.
* **Reference** `19_fourier_playground.json`.

---

## Segmentation, clustering and stylization

### CV Anisotropic Segmentation (GST)

The gradient structure tensor tutorial as a graph: Sobel derivatives, box-filter
the three tensor components over a `W x W` window, solve for the eigenvalues, and
emit coherency `(l1-l2)/(l1+l2)` plus orientation in degrees. `segmented` is the
AND of "coherent enough" and "oriented within the band".

* **In** `src`; `W` INT; `C_Thr` FLOAT; `LowThr` FLOAT; `HighThr` FLOAT
* **Out** `coherency` (0..1 float), `orientation` (degrees), `segmented` (uint8
  0/255)
* `W` is the integration window - it sets the scale of the texture you are
  looking for, and it is the input that matters most.
* Orientation is `0.5 * phase` and therefore lives in **0..180**, not 0..360: it
  is a direction, not a vector. `LowThr`/`HighThr` are in that range.
* Single channel in. Coherency is 0/0 in flat regions; the division is guarded
  but the value there is meaningless, which is exactly what `C_Thr` removes.
* **Reference** `27_anisotropic_segmentation.json`.

### CV K-Means Colors

`cv2.kmeans` over the flattened float32 pixel stack, plus a palette strip of
`64 x (64*K)` swatches sorted by coverage.

* **In** `image` (IMAGE/MASK/NPARRAY); `colors` INT (default 5); `attempts` INT
  (8); `seed` INT (0)
* **Out** `quantized` IMAGE, `palette` IMAGE, `values` NPARRAY `(1, K, C)`
* Uses `KMEANS_PP_CENTERS`. `CV_32F` is asserted by cv2, so the blueprint casts.
* The palette and `values` are in **coverage order** - most common colour first.
* `seed` seeds cv2's global RNG so the clusters repeat; `attempts` is
  cv2.kmeans' own restart count (higher is more stable, linearly slower).
* For an IMAGE input the quantized output keeps the same 8-bit range.
* **Reference** `24_kmeans_clusters.json`.

### CV K-Means Mask Clusters

Measures every region of a mask, z-scores the chosen property columns, and
k-means them into groups - one output MASK per group, largest first.

* **In** `mask` MASK; `clusters` INT (5); `properties` STRING; `scale` COMBO
  (standardized / raw / image-relative); `attempts` INT (10); `seed` INT
* **Out** `cluster_masks` MASK batch, `bboxes` BOUNDING_BOX, `labels`,
  `features`, `cluster_map`
* A single mask is split into its connected blobs; a MASK *batch* is treated as
  one region per frame.
* `properties` picks the feature columns (centroid, area, aspect ratio,
  circularity, orientation, mean colour...). `no position | area` clusters by
  **size alone**.
* **The `properties` checkboxes live on the Region Properties node INSIDE the
  subgraph.** A promoted subgraph input is a bare socket with no widget, so to
  choose columns from outside, drop a `CV Region Property Flags` node on the
  canvas and wire its STRING into `properties`. Unconnected, the inner value
  applies.
* `scale` decides whether columns are comparable: standardized (z-score) weighs
  each column equally; raw lets a raw area outweigh a raw centroid.
* `k` is clamped to the region count, so asking for more clusters than blobs is
  safe.
* **Reference** `24_kmeans_clusters.json` (twice).

### CV K-Means Points

The point-set sibling: clusters an `Nx1x2` point array into `k` groups.

* **In** `points`; `k` INT (3); `attempts` INT (8); `seed` INT
* **Out** `labels`, `centers` (Kx1x2), `mean_sq_distance` FLOAT, `success`
  BOOLEAN
* **Zero points is valid.** `cv2.kmeans` asserts on an empty sample set, so the
  data path substitutes a one-point dummy and the result path throws it away:
  `success` goes false, `labels` and `centers` come back empty, nothing raises.
* `k` is clamped to the point count.
* `mean_sq_distance` is cv2's compactness divided by N - use it to compare two
  values of `k`.
* **Reference** `24_kmeans_clusters.json`, `exercise_clock_reading.json`
  (clustering hand bearings into 3 groups).

### CV Snap Mask To Labels

Rounds a rough mask onto the edges of a segmentation - most usefully
superpixels, whose boundaries already sit on real object edges. It averages the
binarized mask over each label region (so a region's average **is** the covered
fraction), fills every region at or above `coverage_percent`, and clears the
rest. Regions are never partly filled, which is what makes the outline follow the
segmentation.

* **In** `mask` MASK; `labels` NPARRAY (any integer `[H,W]` label map);
  `num_labels` INT; `coverage_percent` FLOAT; `mask_threshold` FLOAT
* **Out** `mask` MASK, `coverage` NPARRAY (per-region fraction, 0..1), `table`
  (one row per label), `counts` (pixel area per label)
* **The two thresholds deliberately do not share a range.** `mask_threshold` is
  `0.0-1.0` because a ComfyUI MASK is 0-1 and it is compared against the mask's
  own pixel values; it is **exclusive** (`>`), so a flat 0.3 mask passes nothing
  at 0.3, and it only matters for soft or antialiased masks. `coverage_percent`
  is `0-100` because it is an area percentage; it is **inclusive** (`>=`), so a
  region covered exactly 50% survives at 50.
* `coverage_percent = 0` fills **every** region, including ones the mask never
  touched (0% >= 0%). For "every region the mask touches at all" use something
  like 0.001. `100` fills only fully covered regions.
* Preview `coverage` as a heat map to pick the threshold.
* Not superpixel-specific: watershed, connected components, HFS and k-means label
  maps all work. Wire `num_labels` from the segmenter's region count to pin the
  table height (0 derives it).
* The mask may be a different resolution from the label map - it is
  area-resampled onto the label grid, so the fractions stay exact.
* **Reference** `25_saliency_superpixels.json`: saliency -> largest connected
  blob is the rough mask, SLICO superpixels supply `labels` **and** `num_labels`,
  run at `coverage_percent = 25`, `mask_threshold = 0.5`.

### CV Superpixel Paint

Posterizes an image by superpixel: optional edge-preserving presmooth, SLICO
regions, mean colour per region (or a k-means palette), then darkened boundaries
drawn back over the result.

* **In** `image`; `region_size` INT; `presmooth` BOOLEAN; `posterize` BOOLEAN;
  `palette_size` INT; `algorithm` COMBO; `iterations` INT; `switch` BOOLEAN;
  `alpha` FLOAT; `ksize_x` INT
* **Out** `image` IMAGE, `labels`, `palette`, `contours`, `count`
* `region_size` drives the SLIC family; SEEDS and ScanSegment are driven by a
  superpixel *count* instead - the node inside feeds each algorithm the one it
  uses, so changing `algorithm` changes which knob is live.
* `posterize` off = mean colour per region; on = those region colours are
  themselves k-means'ed down to `palette_size`.
* `alpha` is the boundary darkness, `ksize_x` the line width.
* **Reference** `16_npr_filters_playground.json` (five instances - the knob sweep
  is the demo).

### CV Cartoonify

Three nodes: bilateral filter to flatten colour, adaptive threshold for ink
outlines, multiply-blend the two.

* **In** `src` - **Out** `IMAGE`
* Baked in: `bilateralFilter(d=87, sigmaColor=150, sigmaSpace=75)` and
  `adaptiveThreshold(blockSize=199, C=23)`. A bilateral `d` of 87 is very large
  and very slow on a big image - shrink first, or lower it.
* The blend is `ImageBlend` in multiply mode at full strength, so the ink is hard
  black. Lower its factor for a softer line.
* **Reference** `16_npr_filters_playground.json`.

---

## Matting and clean plate

The clean-plate family is the shipped answer to "matte without a green screen":
one take plus one photograph of the same scene without the subject. It leans on
`CV Local Linear Fit` (per-pixel windowed regression of take on plate) -
where the subject is opaque the local slope goes to 0, so `1 - slope` is alpha.

### CV Clean Plate Matte

The core, NPARRAY in and out: local regression for a first alpha, reconstruct the
foreground colour `F = intercept / alpha`, inpaint `F` outward from trustworthy
donor pixels, then solve alpha properly from the projection of `C-B` onto `F-B`.

* **In** `shot`, `plate` (both NPARRAY) - **Out** `alpha`, `foreground`,
  `confidence`, `plate_texture`, `colour_donors`
* `confidence` is the magnitude of `F - B` in grey levels - **alpha is only
  trustworthy where foreground and background actually differ.** A subject the
  same colour as the wall behind it cannot be matted, and this output is how you
  see that.
* `colour_donors` shows which pixels seeded the inpaint (locally flat, safely
  inside the subject). If it is nearly empty, the matte will be mush.
* Both inputs must be **the same size and already aligned**; use `CV Plate Align`
  first if the camera moved.
* Baked in: regression window 5 px, pooled across channels; the alpha dead zone
  is 0.12 and "opaque" is 0.9.
* **Reference** `exercise_clean_plate_matting.json` (three instances).

### CV Clean Plate Matte (RGBA)

The same core with IMAGE in and a straight-alpha RGBA out, ready for
`JoinImageWithAlpha` consumers.

* **In** `shot` IMAGE, `plate` IMAGE - **Out** `rgba` IMAGE, `alpha` MASK,
  `confidence`
* The alpha is **straight, not premultiplied** - the colour is unmixed, so
  compositing is `F*a + B*(1-a)`.
* **Reference** `exercise_clean_plate_matting.json`.

### CV Clean Plate Matte (Align + Cleanup, RGBA)

The full pipeline: SIFT + RANSAC registration of the plate onto the take, a
photometric gain/bias fit, the matte core, then noise-floor and speckle cleanup
with a guided filter.

* **In** `shot` IMAGE; `plate` IMAGE; `align_plate` BOOLEAN; `clean_up` BOOLEAN;
  `value` STRING (the noise floor, default `0.01`)
* **Out** `rgba` IMAGE, `alpha` MASK, `plate_used` NPARRAY, `registered` BOOLEAN
* `registered` is the honest signal - when the homography fails the raw plate is
  used unchanged and `registered` goes false. Branch or stamp on it.
* `value` arrives as a STRING and is converted inside, because there is no
  nullable FLOAT widget. It is the alpha dead zone, rescaled as
  `(alpha - floor) / (1 - floor)`.
* Alignment only fixes a **static** scene shot from a slightly different
  position - it is a homography, not a solve for parallax.
* `plate_used` is the warped, exposure-matched plate: preview it when the matte
  looks wrong, since most failures are really registration failures.
* **Reference** `exercise_clean_plate_matting.json`.

### CV Plate Align (Geometric + Photometric)

The registration half on its own: SIFT on both, ratio-test match, RANSAC
homography plate -> take, warp, then `CV Photometric Align` for gain and bias
per channel.

* **In** `plate`, `shot` - **Out** `aligned`, `background_mask`, `coverage`,
  `found`
* `coverage` is where the warped plate actually has data - outside it the matte
  has nothing to compare against, so gate on it.
* Subject keypoints will not agree between the two images; that is expected and
  RANSAC is what removes them. A subject filling most of the frame leaves too few
  background keypoints and `found` goes false.
* Photometric align is gain+bias **per channel**, so it also corrects a white
  balance drift between the two shots.
* **Reference** `exercise_clean_plate_matting.json` carries an inline copy named
  `Plate Align (Geometric + Photometric)`.

### CV Matte Cleanup (Noise Floor + Speckle)

The cleanup half on its own: erode the coverage edge, apply a dead zone, keep the
confident core, morphological open to drop speckle, dilate that into a gate, and
guided-filter the alpha onto the take's edges inside the gate.

* **In** `alpha`, `guide` (the take), `coverage`, `noise_floor` FLOAT
* **Out** `alpha`, `gate`
* `noise_floor` is the dead zone: everything below it goes to 0 and the rest is
  rescaled, so raising it kills grain in the background *and* eats genuine soft
  edges. The "confident core" threshold is `2 * noise_floor`.
* `gate` is the region cleanup was allowed to touch - preview it when a wanted
  soft edge disappears.
* The guided filter is what snaps alpha to real edges, and it needs the **take**
  as guide, not the plate.
* **Reference** `exercise_clean_plate_matting.json`, inline copy
  `Matte Cleanup (Noise Floor + Speckle)`.

### CV Triangulation Matte

The textbook two-background matte: the same subject shot over two different
backgrounds gives alpha in closed form from the projection of `dC` onto `dB`.

* **In** `take_a`, `background_a`, `take_b`, `background_b`
* **Out** `alpha`, `foreground` (premultiplied), `confidence`
* `confidence` is the magnitude of `dB` - alpha is undefined where the two
  backgrounds are the same colour, so "two backgrounds" that barely differ is no
  better than one.
* All four inputs must be pixel-aligned; nothing here registers anything.
* **It needs a studio capture**: the same subject shot twice, over two
  different backgrounds. If you can only shoot one take plus an empty set, use
  `CV Clean Plate Matte` instead - it solves the same algebra from a single
  pair. No shipped workflow uses this blueprint for that reason.

---

## Compositing

### CV Exposure Match (Masked Per-Channel)

Matches one image's exposure to another **inside a shared mask**: per channel,
`(x - mean_correct) * (sigma_ref / sigma_correct) + mean_ref`.

* **In** `image_correct`, `image_reference`, `mask` - **Out** `corrected`
* The mask should be the **overlap** of the two views - statistics taken from
  non-overlapping content is what makes naive exposure matching drift.
* Per channel, so it corrects white balance as well as brightness. For brightness
  only, use `CV Photometric Align` in pooled mode.
* There is a std floor inside to stop a flat region (sigma near 0) exploding the
  gain.
* Works in float32 and the result can exceed `0..255`; it is **not** clipped -
  clamp or cast downstream.
* **Reference** `exercise_match_two_views_over_a_mask.json` (four instances),
  `exercise_seam_pipeline.json`.

### CV Multi-Band Blend (Laplacian pyramid)

Three-level Laplacian blend: build two detail bands plus the coarse Gaussian
level for both images, blend each band with the correspondingly downsampled
weight, and collapse.

* **In** `image_a`, `image_b`, `weight_b` (0..1, B's weight) - **Out** `blended`
* Fixed at **3 levels**. Deep enough for a soft seam, not for a hard one across a
  large image - a visible join means you need more levels than this blueprint
  has.
* `weight_b` should be a *soft* seam mask (distance transform or blurred), not a
  hard 0/1 mask; the whole point is that low frequencies cross over a wide band
  and high frequencies over a narrow one.
* Everything is float32 because the detail bands are signed.
* Both images and the weight must be the same size.
* **Reference** `exercise_seam_pipeline.json`.

### CV Paste Through Homography

Warps `content` into `scene` through a homography, warping an all-white coverage
plate the same way, and composites premultiplied so no dark rim appears at the
edge of the warp.

* **In** `content` IMAGE; `scene` IMAGE; `homography`; `source_width` INT;
  `source_height` INT; `content_mask` MASK; `scene_paintable` MASK
* **Out** `image` IMAGE, `mask` MASK (where the content landed), `warped`
* `source_width`/`source_height` are the frame the homography maps **from** - the
  resolution the reference was measured at, which is not necessarily the
  content's own size. `CV Find Planar Object` outputs exactly these.
* Both masks are optional: unconnected means "all of it". `scene_paintable`
  restricts where the paste may land (a wall, a poster, a screen).
* The premultiplied composite is why this is a blueprint and not two nodes -
  `scene * (1 - coverage) + warped`, with the coverage warped identically.
* **Superseded for most uses by the `CV Paste Through Warp` node**, which is
  what `47_paste_into_scene.json` actually uses. The blueprint remains as the
  graph-level version, for when you need to intervene between the warp and the
  composite.
* **Reference** `47_paste_into_scene.json` (the node);
  `36_feature_matching_playground.json` for the homography that feeds it.

---

## Matching, tracking and pose

### CV Match Template

`cv2.matchTemplate` plus `minMaxLoc` and the template's size, turned into a
`BOUNDING_BOX` around the best match.

* **In** `image`, `templ` (both accept NPARRAY/IMAGE/MASK/LATENT)
* **Out** `BOUNDING_BOX`, `nparray` (the score map)
* The method is fixed on the node inside. With `TM_SQDIFF` the *minimum* is the
  match and this blueprint takes the maximum - check before switching methods.
* No scale or rotation invariance. For scale, use the
  `CV Match Template Multi-Scale` node (workflow 38).
* **Reference** `37_match_template.json`, inline copy `CV Match Template`.

### CV Find Planar Object (Reference in Scene)

Detects a planar object (poster, book cover, screen) in a scene photo: optional
downscale of either side, one detector on both, ratio-test matching, RANSAC
homography, projected corner quad, filled footprint mask and a `found` gate.

* **In** `reference` IMAGE; `scene` IMAGE; `detector`; `max_features`;
  `scale_reference` + `reference_megapixels`; `scale_scene` + `scene_megapixels`;
  `ratio`; `reproj_threshold`; `min_inliers`; `draw_matches`; `matcher`;
  `reference_mask`; `scene_mask`
* **Out** `found`, `homography`, `mask`, `quad`, `inlier_count`,
  `scene_width/height`, `reference_width/height`, `matches_vis`
* The inputs fall into four groups that do not interact - keypoints, matching,
  fit, diagnostics - and that is the order to tune them in. A loose fit cannot
  rescue bad keypoints.
* **`max_features = 0` means "unlimited" for SIFT and "keep zero" for ORB.** ORB
  at 0 finds nothing at all; give it ~20000.
* Both images must use the **same** detector - descriptors from two different
  ones are not comparable, which is why there is one knob and not two.
* The scale gates **only ever shrink** (megapixels are the core resampler's
  1024x1024 unit); asking for 2 MP of a 0.2 MP image is a no-op, not an upscale.
  Neither gate changes the coordinates you get back: `homography`, `quad` and
  `mask` are always in the ORIGINAL images' pixels.
* `ratio` is Lowe's test and **lower is stricter** (0.7-0.8 usual) - the main
  defence against repeated texture. `min_inliers` is the false-positive gate:
  below it the blueprint reports `found = false` and returns the identity
  homography with an empty mask instead of a plausible lie.
* `mask` is already gated on `found`; `quad` deliberately is not, because
  identity-projected corners are visibly absurd rather than subtly wrong.
* `matches_vis` is the only output NOT in original coordinates - it shows the
  matched (possibly downscaled) pair, because that is where its keypoints live.
  Turning `draw_matches` off skips the drawing entirely and passes the scene
  through.
* Triage: empty or fanned-out `matches_vis` = a keypoint problem; lines agree but
  `found` is false = raise `reproj_threshold` or lower `min_inliers`; `found` is
  true but the quad is wrong = lower `ratio` and raise `min_inliers`.
* **Reference** `36_feature_matching_playground.json`.

### CV Hue Track Step

One frame of meanShift tracking on a hue x saturation back-projection. Not a
tracker - the per-frame *step* of one: instantiate it once per frame and chain
`window` into the next instance's `window_in`.

* **In** `frame` IMAGE (one frame, not a batch); `histogram` NPARRAY;
  `window_in` NPARRAY `(x, y, w, h)`
* **Out** `measurement`, `center`, `found`, `window`, `density`
* `histogram` is the colour model: build it **once** with `CV CalcHist` over
  channels `0, 1`, bins `30, 32`, ranges `0, 180, 0, 256`, peak normalised to
  255. The same histogram feeds every instance.
* `measurement` is the centre when found and an **empty array** when not - feed
  it to an estimator that accepts "no measurement this frame". `center` is always
  populated but goes stale when the track is lost (Track Window echoes its input
  window), so it is for drawing, not for deciding. `found` is the real signal.
* `density` is the gated back-projection the step climbed on - preview it to tune
  the lost threshold, or to see something else out-scoring the target.
* Baked in, not promoted: the trustworthy-hue floor `S >= 110, V >= 60` (hue is
  meaningless on grey and dark pixels); `min_density = 22.2`, the lost threshold
  in mean back-projection value (0-255); meanShift with a fixed-size window, 10
  iterations, epsilon 1.0. Switch to CamShift inside if the target's apparent
  size changes as it moves.
* **Reference** `40_kalman_tracking.json`: a red bicycle across
  `pedestrian_area.mp4`, frames 21-28, eight chained instances feeding a Kalman
  chain. Learn the model once at frame 21, seed the window once at
  `(411, 198, 50, 50)`, chain `window`, send `measurement` to
  `CV Kalman Filter Step`. From the fifth frame the bike passes behind
  pedestrians, density falls ~23 -> ~20, `found` goes false and the raw marker
  freezes 56 px behind; the filter, fed nothing, coasts on the estimated velocity
  and stays 3-6x closer.

### CV Rapid Pose Refine

One RAPID iteration: sample control points on the model silhouette, search along
their normals for the strongest image gradient, and solve PnP (plain or RANSAC)
from the resulting 3D-2D correspondences.

* **In** `img`; `pts3d`; `tris`; `K`; `rvec`; `tvec`; `num` INT; `len` INT;
  `use_ransac` BOOLEAN; `reprojection_error` FLOAT
* **Out** `found`, `rvec`, `tvec`, `correspondences`, `search_lines`, `bundle`,
  `cols`, `scores`, `inliers`
* It **refines** a pose, it does not find one - the incoming `rvec`/`tvec` must
  already be close, and the projected silhouette must be near the real edges or
  the search locks onto the wrong gradient.
* `len` is the half-length of each search line in pixels: too short and it never
  reaches the edge, too long and it finds background clutter.
* `search_lines` and `correspondences` are the debug outputs - draw them when a
  refinement diverges.
* Chain several instances for several iterations.
* **Reference** `41_rapid_tracking_playground.json`.

### CV Pick Point On Mesh

Turns clicked image pixels into 3D points in the **model's own frame**: render
metric depth of the mesh at the current pose, read the depth under each pixel,
back-project to camera space, then apply the inverse pose.

* **In** `pts3d`, `tris`, `K`, `rvec`, `tvec`, `points_2d`, `frame`
* **Out** `points_3d`, `valid`, `count`, `depth`, `silhouette`
* A click that misses the mesh has no depth - `valid` marks it rather than
  silently placing the point at zero.
* The output is in model coordinates, so the annotation stays attached when the
  object moves in later frames. That is the whole point of the blueprint.
* `silhouette` doubles as a check that the pose you passed in is the pose you
  think it is.
* **Reference** `63_ar_model_annotation.json`, `42_rapid_model_tracking.json`.

### CV Annotate Model

Projects 3D anchor points into the frame, hides the ones facing away or behind
scene geometry, and draws markers plus captions.

* **In** `frame`; `points_3d`; `rvec`; `tvec`; `K`; `scene_depth`; `labels`;
  `depth_tolerance` FLOAT
* **Out** `overlay`, `points_2d`, `visible`
* Occlusion is decided by comparing projected depth against `scene_depth` within
  `depth_tolerance`: too tight and anchors flicker, too loose and far-side
  anchors show through the object.
* Hidden anchors come back as **NaN** in `points_2d`; the drawing nodes skip NaN
  rather than clamping it to a corner. Downstream maths must expect NaN.
* `labels` is a string array matched by index to `points_3d`.
* **Reference** `63_ar_model_annotation.json`, `42_rapid_model_tracking.json`.

### CV Draw 3D Measurement

Two 3D points in; a dimension line with end caps and the distance rendered at the
midpoint out.

* **In** `frame`; `points_3d` (exactly two); `rvec`; `tvec`; `K`
* **Out** `overlay`, `distance` FLOAT, `points_2d`
* The distance is computed **in 3D, in the pose's own units** - millimetres if
  the model is in millimetres. It is not a pixel measurement, so it does not
  change with viewing angle.
* The caption is rounded for display; `distance` carries the full value.
* **Reference** `63_ar_model_annotation.json`, `42_rapid_model_tracking.json`.

### CV Stabilize Frame (Align to Reference)

Aligns one frame onto a reference by either ECC (direct photometric) or features
plus a RANSAC homography, with a `found` gate and a quality score.

* **In** `frame` IMAGE; `reference` IMAGE; `method` COMBO; `motion_type` COMBO;
  `min_correlation` FLOAT; `detector` COMBO
* **Out** `aligned` IMAGE, `matrix`, `found`, `score`
* `motion_type` deliberately excludes HOMOGRAPHY on the ECC path, because
  `warpAffine` cannot take a 3x3 - the ECC path returns a 2x3, the feature path a
  3x3, and `matrix` is one or the other depending on `method`.
* `score` is the ECC correlation on one path and the RANSAC inlier count on the
  other, so **the number means different things**; `min_correlation` only gates
  the ECC one.
* When `found` is false the frame passes through untouched rather than being
  warped by a bad fit.
* ECC is slow and needs a decent starting overlap; features cope with larger
  motion but need texture.
* **Reference** none instantiates it; `exercise_image_registration_ecc.json`
  covers the same ground with the underlying nodes.

### CV Trajectory Plot (Top-Down)

Draws an XZ (top-down) plot of an estimated trajectory against ground truth, both
mapped through a **shared** extent so the two are comparable, with axes, a grid,
a scale bar, a time ramp and direction arrows.

* **In** `estimate`; `truth`; `gap_mask`; `units` STRING - **Out** `plot`
* Truth is optional: unconnected, the merge yields an empty array and only the
  estimate is drawn.
* The shared extent is the point - plotting each track to its own bounds hides
  exactly the drift you are trying to see.
* `gap_mask` marks frames with no estimate, so the path breaks rather than
  interpolating across a dropout.
* Top-down means the vertical axis is **Z**, not Y, and a left turn must draw as
  a left turn (the node behind it had a mirroring bug once).
* **Reference** `exercise_visual_odometry.json` (four shipped stills, runs out
  of the box), `exercise_visual_odometry_video.json` (the ground-truth
  poses ship in `example_inputs/`; **the 48 MB clip does not** - the workflow's
  Note says how to encode it from KITTI raw drive `2011_09_26_drive_0009`, and
  its *Data sources* MarkdownNote carries the links. The data is
  [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) -
  **non-commercial** - which is why it stays out of the repo:
  [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php)).

### CV Transform Points (Scale / Rotate / Translate)

Builds `Rz . Ry . Rx` from three Euler angles in degrees, packs it with a
translation into a 4x4, and applies it to a point set after a uniform scale.

* **In** `points`; `scale`; `rotate_x_deg`; `rotate_y_deg`; `rotate_z_deg`;
  `translate_x/y/z` - **Out** `points`, `matrix`
* The order is fixed: **scale, then rotate, then translate**, with rotations
  applied Z then Y then X. A different order needs a different graph.
* Angles are degrees on the input and converted inside - do not pre-convert.
* `matrix` is the 4x4 it built, so the same transform can be reused elsewhere.
* No shipped workflow uses it.

### CV Polygon IoU (convex)

`cv2.intersectConvexConvex` plus two `contourArea` calls:
`inter / (A + B - inter)`.

* **In** `poly_a`, `poly_b` - **Out** `iou` FLOAT, `intersection_area` FLOAT,
  `intersection_points`
* **Convex only.** A concave polygon gives a wrong answer silently, not an error.
* Point order matters to `contourArea`'s sign; the blueprint takes the areas as
  returned, so pass consistently wound contours.
* Disjoint polygons give IoU 0 and an empty `intersection_points` - valid, not an
  error.
* **Reference** `31_convex_geometry_playground.json`, inline copy
  `Polygon IoU (convex)`.

---

## Video

These four take a whole IMAGE batch. The two that loop (`CV Flow Interpolate`,
`CV Foreground Mask`) use the core fold/ITEM_LIST nodes and are **linear in frame
count** - not free on a long clip.

### CV Background Plate (Video)

Temporal median (or another statistic) over the batch: whatever is static becomes
the plate, whatever moves averages out.

* **In** `frames` IMAGE; `statistic` COMBO; `denoise_ksize` INT
* **Out** `plate` NPARRAY, `plate_image` IMAGE
* **Median, not mean** - a mean smears every moving object into a ghost. Median
  is the default and usually the right answer.
* It needs the subject to actually move across the clip; someone standing still
  for the whole take becomes part of the plate.
* `denoise_ksize = 1` disables the median blur (it is a kernel size, not a flag).
* **Reference** none instantiates it; `28_background_subtract_bgsegm.json` and
  `exercise_background_subtraction.json` use the same `CV Temporal Reduce`
  node directly.

### CV Foreground Mask (Video)

Per-frame foreground masks, by either an adaptive background subtractor over the
whole batch or a static median plate plus a sigma threshold, then morphological
open and close.

* **In** `frames` IMAGE; `method` COMBO; `algorithm` COMBO; `warmup_frames` INT;
  `plate_threshold` FLOAT; `cleanup_ksize` INT
* **Out** `foreground` MASK batch, `plate` NPARRAY, `foreground_fraction` FLOAT
* `plate_threshold` is **in sigma**, not grey levels.
* The static path needs a mostly empty scene; the adaptive path needs
  `warmup_frames` before its output means anything, and slowly absorbs anything
  that stops moving.
* The fold converts MASK to IMAGE and back because core has no MASK batcher -
  harmless, but it is why the graph looks longer than the idea.
* `foreground_fraction` is a cheap sanity check: near 1.0 means the plate is
  wrong, near 0 means the threshold is too high.
* **Reference** none instantiates it; `exercise_background_subtraction.json`
  builds the same thing.

### CV Flow Interpolate (Video)

Doubles the frame rate: for each consecutive pair compute flow both ways, pull
both frames to time `t` with `CV Flow Map` plus remap, and cross-fade.

* **In** `frames` IMAGE; `t` FLOAT; `flow_preset` COMBO; `border_mode` COMBO
* **Out** `frames_2x` IMAGE, `frame_count` INT
* Output is **2N-1** frames, not 2N - the last original frame has no successor to
  interpolate with, and the fold appends it separately.
* `border_mode` deliberately has no TRANSPARENT option: the remap destination is
  a fresh buffer, so TRANSPARENT would leave garbage.
* Interpolating over an occlusion or a cut produces the usual flow-warp tearing;
  this is a two-frame method with no occlusion reasoning.
* **Reference** none instantiates it;
  `exercise_optical_flow_interpolation_video.json` and
  `exercise_optical_flow_interpolation_comparison.json` build it inline. For the
  learned alternative see `CV Frame Interpolate (Video, DNN Model)`.

### CV Motion Segmentation (Flow Residual)

Separates object motion from camera motion: dense DIS flow, fit a global RANSAC
affine (the camera), subtract it, threshold the residual, clean up, keep the
largest blob, optionally refine with GrabCut.

* **In** `frame_a`, `frame_b`; `finest_scale` INT; `flow_preset` COMBO;
  `refinement_iterations` INT; `threshold_px` FLOAT; `cleanup_ksize` INT;
  `refine_grabcut` BOOLEAN
* **Out** `mask`, `mask_out` MASK, `dominant_motion`, `found`, `area` INT,
  `camera_matrix`
* `threshold_px` is residual motion **in pixels**, so it scales with how fast the
  object moves relative to the camera. It is the knob.
* Only the **largest** moving blob survives; two independently moving objects
  give you one of them.
* An affine camera model assumes a distant or planar background. Strong parallax
  leaks into the residual and reads as false motion.
* GrabCut refinement snaps the mask to colour edges and costs real time; its
  seeds are an erode/dilate pair of the rough mask.
* **Reference** none instantiates it;
  `exercise_optical_flow_motion_segmentation.json` builds the same pipeline.

---

## DNN model blueprints

Each of these wraps `cv2.dnn` primitives around one ONNX file. The model note
lives on the blueprint's own canvas; the URLs are repeated here. `cv2.dnn` is
**inference only** - none of these can be fine-tuned.

Models go in `ComfyUI/models/onnx` unless stated otherwise.

### CV Dense Flow (DNN Model)

RAFT optical flow: pack the two frames as a 2-image blob, forward, reshape
NCHW -> `(1, H, W, 2)`, rescale the field from network pixels to source pixels,
and split into magnitude and angle in the same output slots as the DIS node.

* **In** `frame_a`, `frame_b` IMAGE; `model`; `net_width`; `net_height`;
  `input_names` STRING; `input_scale` FLOAT; `engine` COMBO
* **Out** `flow`, `magnitude`, `angle`
* **Model** `optical_flow_estimation_raft_2023aug.onnx` (61 MB fp32) from
  <https://huggingface.co/opencv/optical_flow_estimation_raft>; the 47 MB
  int8-block-quantized export in the same repo also works.
* The net's output order is **not stable** - that is what the "order is not
  stable!" title on the pick node means. Check it if you swap models.
* It can emit **NaN**. The blueprint runs `checkRange` and, on failure, re-runs
  the reversed pair and negates the result (`B->A` flow is the negative of
  `A->B`). NaN is also zeroed first so it never poisons the graph.
* `net_width`/`net_height` must match what the export expects; the flow is
  rescaled back to the source resolution for you.
* **Reference** `76_raft_optical_flow_dnn.json`.

### CV Frame Interpolate (DNN Model)

RIFE: pad both frames to a multiple of 32, stack them with a constant timestep
plane into a 7-channel input, forward, crop back.

* **In** `frame_a`, `frame_b` IMAGE; `model`; `t` FLOAT; `engine`
* **Out** `middle` IMAGE, `middle_array` NPARRAY
* **Model** `rife_v4.9.onnx` (21 MB fp32) from
  <https://huggingface.co/FuryTMP/RIFE_v4.9>. **Not** the export at
  `FuryTMP/RIFE_fp32` - cv2.dnn cannot run that graph.
* The 32-multiple padding is mandatory; the net rejects other sizes. The crop
  afterwards is why the output matches the input exactly.
* `t` is the timestep in `[0, 1]`, fed in as a full plane filled with exactly
  `t` - not as a scalar input.
* Input must be RGB in `0..1`, which the first node does.
* **Reference** `77_rife_frame_interpolation.json` (three instances).

### CV Frame Interpolate (Video, DNN Model)

The same net over a whole clip at an integer multiplier, using a fold over output
slots: slot `j` maps to pair `j / multiplier` and sub-step `j % multiplier`, and
sub-step 0 passes the **original** frame through instead of synthesising it.

* **In** `frames` IMAGE; `model`; `multiplier` INT; `engine`
* **Out** `frames_out` IMAGE, `frame_count` INT
* Output is `(N-1) * multiplier + 1` frames.
* Originals are preserved exactly at every multiple - the `k == 0` switch is what
  guarantees it.
* One forward pass per synthesised frame: the cost is linear and real.
* **Model** as above.
* **Reference** `77_rife_frame_interpolation.json`.

### CV Image To Image (DNN Model)

The generic image->image wrapper: blob at native size with `swap_rb`, one forward
pass, NCHW -> HWC, then a postprocess choice.

* **In** `image` IMAGE; `model`; `scale` FLOAT; `postprocess` COMBO; `engine`;
  `backend`; `target`
* **Out** `image` IMAGE, `array` NPARRAY
* `scale` is the **input** range knob: `1.0` for a raw 0-255 model
  (fast-neural-style), `1/255` for a 0-1 model (NAFNet-style). Getting it wrong
  produces a plausible-looking wrong result, not an error.
* `postprocess` picks between "clip only" and "x255 then clip". Both round and
  clamp to 0-255 **without** an absolute value - `convertScaleAbs` would reflect
  negative pixels back to bright and corrupt them, and style models run well
  outside `[0,255]` on both sides.
* Some nets round each output side up to a multiple of 4, so the result can be a
  pixel or two larger than the input.
* **Model** `mosaic-9.onnx` from <https://huggingface.co/onnxmodelzoo/mosaic-9>;
  `udnie` / `candy-9` / `rain-princess-9` / `pointilism-9` use identical
  preprocessing.
* **Reference** `73_dnn_style_transfer.json` builds this from the same
  primitives.

### CV Image To Image (Batch, DNN Model)

The same, but the whole IMAGE batch is encoded into one `(N,C,H,W)` blob and run
in a single forward pass.

* **In** `images` IMAGE; the rest as above - **Out** `images` IMAGE, `array`
* **Most ONNX exports accept batch = 1 only.** If the forward errors on a
  multi-frame batch, fall back to the single-image blueprint. That is the one
  real difference between the two.
* **Reference** `74_batch_dnn_style_transfer.json`.

### CV YOLO Detection

YOLOv4: letterbox to 608 keeping aspect, blob at 1/255 with BGR->RGB, forward all
outputs, pick the box and confidence tensors, decode and NMS, and load the COCO
names.

* **In** `image` IMAGE
* **Out** `bboxes` BOUNDING_BOX, `scores`, `labels`, `det_count`, `found`
* **Model** `yolov4.onnx` from
  <https://huggingface.co/opencv/opencv_contribution>. Labels:
  `models/labels/coco-labels-2014_2017.txt` (80 COCO classes in 2014/2017 naming
  - `motorcycle`/`airplane`/`couch`/`tv`).
* The letterbox is why boxes come back in the original image's coordinates;
  changing the 608 means changing the decode too.
* Zero detections is valid - `found` goes false and the arrays come back empty.
* **Reference** `69_yolov4_detection.json`.

### CV YOLO-Seg Inference

YOLO segmentation: letterbox to 640, one forward, decode the detection head to
boxes, and build instance masks as coefficients times prototypes.

* **In** `image` IMAGE
* **Out** `bboxes`, `masks` MASK, `scores`, `labels`, `det_count`, `found`
* **Model** `yolo26n-seg.onnx` from
  <https://huggingface.co/opencv/opencv_contribution>. Labels:
  `example_inputs/coco-labels-2014_2017.txt`.
* The two output tensors are picked by shape (`(1,N,6+C)` detection head,
  `(1,C,h,w)` prototypes) - another seg export may order them differently.
* Masks come out at prototype resolution and are resized, so expect them to be
  coarser than the boxes.
* **Reference** `70_yolo_seg.json`.

### CV Document Scan (SAM seg)

Segment the document with SAM3, take the largest contour, fit a quadrilateral,
recover the **true aspect ratio** from the perspective, and warp to a rectangle.

* **In** `image` IMAGE; `ckpt_name` COMBO
* **Out** `BOOLEAN` (found), `output` IMAGE (the rectified page), `output_1` MASK
* **Model** `sam3.1_multiplex_fp16.safetensors` from
  <https://huggingface.co/Comfy-Org/sam3.1> - it goes in
  `ComfyUI/models/checkpoints`, not `onnx`.
* **Nests** `Image Segmentation (SAM3)`.
* The target rectangle is derived from the quad's perspective rather than
  assumed. Focal length is unobservable on a one-vanishing-point quad, so the
  solve uses a focal ratio; a quad that is nearly a parallelogram carries almost
  no aspect information and the result approaches a guess.
* The quad fit falls back to the convex hull when the contour will not collapse
  to four corners.
* **Reference** `exercise_document_perspective.json`.

### CV Color Correct

Detect a colour checker (`cv2.mcc`), build a CCM from it and apply it - with a
switch so a failed detection passes `on_false` through instead of a wrong
correction.

* **In** `image` IMAGE; `on_false` IMAGE - **Out** `IMAGE`, `found` BOOLEAN
* `found` is the real signal. Wire `on_false` to the original image (or a stamped
  copy) so a miss is visible rather than silent.
* Uses `Basic data handling` dict nodes to carry the model, so that pack must be
  installed or the blueprint will not load.
* Needs a real chart in frame, reasonably flat-lit and large enough for the
  detector.
* **Reference** none instantiates it; `11_color_correction_basic.json` and
  `12_white_balance_and_tone.json` cover the same nodes.
