# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# Tooltip text extracted from the OpenCV documentation (Apache-2.0);
# copyright of that text remains with the OpenCV contributors.
# See LICENSE for the full GNU GPL v3 text.

RETURN_DOCS = {
	"buildOpticalFlowPyramid": "number of levels in constructed pyramid. Can be less than maxLevel.",
	"checkChessboard": "Whether a chessboard was found.",
	"computeECC": "The ECC similarity coefficient in the range [-1, 1].",
	"estimateAffine2D": "Output 2D affine transformation matrix $2 \\times 3$ or empty matrix if transformation could not be estimated. The returned matrix has the following form: $$ \\begin{bmatrix} a_{11} & a_{12} & b_1\\ a_{21} & a_{22} & b_2\\ \\end{bmatrix} $$ The function estimates an optimal 2D affine transformation between two 2D point sets using the selected robust algorithm. The computed transformation is then refined further (using only inliers) with the Levenberg-Marquardt method to reduce the re-projection error even more.",
	"estimateAffine3D": "Whether a solution was found. The function estimates an optimal 3D affine transformation between two 3D point sets using the RANSAC algorithm.",
	"estimateAffinePartial2D": "Output 2D affine transformation (4 degrees of freedom) matrix $2 \\times 3$ or empty matrix if transformation could not be estimated. The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust estimation. The computed transformation is then refined further (using only inliers) with the Levenberg-Marquardt method to reduce the re-projection error even more. Estimated transformation matrix is: $$ \\begin{bmatrix} \\cos(\\theta) \\cdot s & -\\sin(\\theta) \\cdot s & t_x \\ \\sin(\\theta) \\cdot s & \\cos(\\theta) \\cdot s & t_y \\end{bmatrix} $$ Where $ \\theta $ is the rotation angle, $ s $ the scaling factor and $ t_x, t_y $ are translations in $ x, y $ axes respectively.",
	"estimateChessboardSharpness": "Scalar(average sharpness, average min brightness, average max brightness,0)",
	"estimateTranslation2D": "A 2D translation vector $[t_x, t_y]^T$ as `cv::Vec2d`. If the translation could not be estimated, both components are set to NaN and, if @p inliers is provided, the mask is filled with zeros. \\par Converting to a 2x3 transformation matrix: $$ \\begin{bmatrix} 1 & 0 & t_x\\ 0 & 1 & t_y \\end{bmatrix} $$",
	"findChessboardCorners": "True if all of the corners are found and placed in a certain order (row by row, left to right in every row). Otherwise, if the function fails to find all the corners or reorder them, it returns false. The function attempts to determine whether the input image is a view of the chessboard pattern and locate the internal chessboard corners. For example, a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black squares touch each other. The detected coordinates are approximate, and to determine their positions more accurately, the function calls #cornerSubPix. You also may use the function #cornerSubPix with different parameters if returned coordinates are not accurate enough. Sample usage of detecting and drawing chessboard corners: :",
	"findFile": "Returns path (absolute or relative to the current directory) or empty string if file is not found",
	"getFontScaleFromHeight": "The fontSize to use for cv::putText",
	"getOptimalNewCameraMatrix": "new_camera_matrix Output new camera intrinsic matrix. The function computes and returns the optimal new camera intrinsic matrix based on the free scaling parameter. By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original image pixels if there is valuable information in the corners alpha=1 , or get something in between. When alpha>0 , the undistorted result is likely to have some black pixels corresponding to \"virtual\" pixels outside of the captured distorted image. The original camera intrinsic matrix, distortion coefficients, the computed new camera intrinsic matrix, and newImageSize should be passed to #initUndistortRectifyMap to produce the maps for #remap .",
	"getTextSize": "The size of a box that contains the specified text.",
	"haveImageReader": "true if an image reader for the specified file is available and the file can be opened, false otherwise.",
	"haveImageWriter": "true if an image writer for the specified extension is available, false otherwise.",
	"imdecodeWithMetadata": "The decoded image as a cv::Mat object. If decoding fails, the function returns an empty matrix.",
	"imdecodeanimation": "Returns true if the buffer was successfully loaded and frames were extracted; returns false otherwise.",
	"imencodeanimation": "Returns true if the animation was successfully encoded; returns false otherwise.",
	"imreadWithMetadata": "The loaded image as a cv::Mat object. If the image cannot be read, the function returns an empty matrix.",
	"imreadanimation": "Returns true if the file was successfully loaded and frames were extracted; returns false otherwise.",
	"imwrite": "true if the image is successfully written to the specified file; false otherwise.",
	"imwriteanimation": "Returns true if the animation was successfully saved; returns false otherwise.",
	"kmeans": "The function returns the compactness measure that is computed as $$\\sum _i \\| \\texttt{samples} i - \\texttt{centers} { \\texttt{labels} _i} \\| ^2$$ after every attempt. The best (minimum) value is chosen and the corresponding labels and the compactness value are returned by the function. Basically, you can use only the core of the function, set the number of attempts to 1, initialize labels each time using a custom algorithm, pass them with the ( flags = #KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best (most-compact) clustering.",
	"sampsonDistance": "The computed Sampson distance.",
	"selectROI": "selected ROI or empty rect if selection canceled.",
	"setLogLevel": "previous logging level",
	"solveCubic": "number of real roots. It can be -1 (all real numbers), 0, 1, 2 or 3.",
	"solveLP": "One of cv::SolveLPResult",
	"rapid.rapid": "ratio, the fraction of search lines that produced a usable correspondence (0..1) - a health signal, not an accuracy measure; the REFINED rvec/tvec to use for the next track step; and rmsd, the 2D reprojection difference in PIXELS. A low rmsd with a wrong pose is possible, so read it together with ratio.",
	"rapid.extractControlPoints": "The sampled control points: ctl2d are their pixel positions along the projected model's silhouette, ctl3d the matching object-space points (the pair order follows the measured Python binding, not the C++ parameter order).",
	"rapid.extractLineBundle": "bundle is num x (2*len+1) x 3 image intensities sampled along each control point's silhouette-normal search line; locations are the pixel coordinates of those samples.",
	"rapid.findCorrespondencies": "cols - the offset of the strongest gradient position along each search line (-1 = no edge found) - and scores, a per-line match confidence in 0..255.",
	"rapid.convertCorrespondencies": "pts2d, the image-space point each matched search line landed on (aligned with extractControlPoints' ctl3d, ready for cv2.solvePnP); mask reports which control points produced a valid correspondence.",
	"threshold": "the computed threshold value if Otsu's or Triangle methods used.",
	"thresholdWithMask": "the computed threshold value if Otsu's or Triangle methods used.",
}
