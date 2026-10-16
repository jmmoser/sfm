/**
 * Shared data types used across every module. See docs/ARCHITECTURE.md for the
 * coordinate-frame and memory-layout conventions these types encode.
 *
 * Conventions (summary):
 *  - Matrices are ROW-MAJOR Float64Array. Mat3 = 9 elements, Mat34 = 12, Mat4 = 16.
 *  - Camera frame is OpenCV-style: +x right, +y down, +z forward (optical axis).
 *  - Pose = world-to-camera rigid transform: x_cam = R * X_world + t.
 *    Camera centre C = -R^T t.
 *  - Pixel coordinates: the CENTRE of the top-left pixel is (0, 0) (OpenCV
 *    convention). COLMAP export adds +0.5.
 *  - All keypoints / intrinsics of a keyframe are expressed in the pixel frame of
 *    the keyframe's full-resolution image (FrameInfo.width x FrameInfo.height),
 *    even if features were extracted on a downscaled copy.
 *  - Quaternions are [w, x, y, z] (Hamilton, w first), unit norm.
 *  - Angles are radians unless the name says Deg.
 *  - Timestamps are milliseconds in the performance.now() time domain.
 */

export type Vec2 = Float64Array; // length 2
export type Vec3 = Float64Array; // length 3
export type Vec4 = Float64Array; // length 4
export type Mat3 = Float64Array; // length 9, row-major
export type Mat34 = Float64Array; // length 12, row-major 3x4
export type Mat4 = Float64Array; // length 16, row-major
export type Quat = Float64Array; // length 4, [w, x, y, z]

/** Rigid world-to-camera transform. x_cam = R * X_world + t. */
export interface Pose {
  R: Mat3;
  t: Vec3;
}

/**
 * Camera model "RADIAL" (identical to COLMAP's RADIAL model but with pixel
 * centres at integer coordinates):
 *   x = Xc/Zc, y = Yc/Zc, r2 = x^2 + y^2, d = 1 + k1*r2 + k2*r2^2
 *   u = f * d * x + cx,  v = f * d * y + cy
 */
export interface CameraIntrinsics {
  width: number;
  height: number;
  f: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
}

/** 8-bit or float single-channel image, row-major, no padding. */
export interface GrayImage {
  width: number;
  height: number;
  /** Uint8Array (0..255) or Float32Array (0..1). */
  data: Uint8Array | Float32Array;
}

/** Interleaved RGBA8 image, row-major, no padding (like ImageData). */
export interface RGBAImage {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

export const DESCRIPTOR_SIZE = 128;

/**
 * Local features of one image (SIFT-like, RootSIFT descriptors).
 * keypoints: 4 floats per feature [x, y, scale, orientation]
 *   - x, y in full-resolution keyframe pixel coordinates (see header).
 *   - scale = detection sigma in full-resolution pixels.
 *   - orientation in radians, image frame (x right, y down; angle measured from +x towards +y).
 * descriptors: DESCRIPTOR_SIZE bytes per feature. RootSIFT (L1-normalise SIFT,
 *   element-wise sqrt => unit L2 norm), quantised as min(255, round(512 * v)).
 * scores: detector response per feature (higher = stronger).
 */
export interface FeatureSet {
  count: number;
  keypoints: Float32Array;
  descriptors: Uint8Array;
  scores: Float32Array;
  /** Dimensions of the full-resolution image the coordinates refer to. */
  width: number;
  height: number;
}

/** Putative or verified matches as flat index pairs [i0, j0, i1, j1, ...]. */
export type MatchPairs = Uint32Array;

export type TwoViewConfig =
  | 'calibrated' // general motion, E is reliable
  | 'uncalibrated' // F reliable but E inconsistent (bad intrinsics)
  | 'planar' // dominant plane, with translation
  | 'panoramic' // (near) pure rotation, no useful baseline
  | 'planar_or_panoramic'
  | 'degenerate'; // not enough inliers

export interface TwoViewGeometry {
  config: TwoViewConfig;
  /** Inlier matches (pairs of feature indices) w.r.t. the selected model. */
  inliers: MatchPairs;
  E?: Mat3;
  F?: Mat3;
  H?: Mat3;
  /** Relative pose of image 2 w.r.t. image 1 (x2 = R x1 + t), |t| = 1. */
  R?: Mat3;
  t?: Vec3;
  /** Median triangulation angle (radians) of inliers in front of both cameras. */
  triAngle?: number;
}

/** One inertial sample, already normalised to the device frame (see sensors/motion). */
export interface ImuSample {
  /** ms, performance.now() domain. */
  t: number;
  /** Angular velocity, rad/s, device frame. */
  gyro: [number, number, number];
  /** Linear acceleration without gravity (m/s^2), device frame; null if unavailable. */
  accel: [number, number, number] | null;
  /** Specific force incl. gravity (m/s^2), device frame, spec sign (flat face-up => +z). */
  accelG: [number, number, number] | null;
}

/**
 * Device frame (W3C DeviceMotion): +x right edge, +y top edge, +z out of the screen
 * (portrait, natural orientation). Camera frame: OpenCV (x right, y down, z forward
 * out of the back of the phone) w.r.t. the *processed image orientation*.
 */
export interface GeoFix {
  t: number; // ms, performance.now() domain
  lat: number;
  lon: number;
  alt: number | null;
  accuracy: number; // metres (1-sigma-ish horizontal)
  altitudeAccuracy: number | null;
}

/** Metadata recorded for every captured keyframe. */
export interface FrameInfo {
  id: number;
  /** Capture time (ms, performance.now() domain). */
  t: number;
  /** Full-resolution image dimensions (after rotation to the session's canonical orientation). */
  width: number;
  height: number;
  /** Laplacian-variance style sharpness (higher = sharper), on the analysis image. */
  sharpness: number;
  /** Rotation from device frame to camera frame for this frame (Mat3). */
  R_cam_device: Mat3;
  /** Gravity direction (unit, pointing DOWN) in the camera frame, if known. */
  gravityCam: Vec3 | null;
  /** Compass heading in degrees clockwise from true north (device top), if known. */
  heading: number | null;
  /** GPS fix nearest in time, if any. */
  geo: GeoFix | null;
}

/** Point cloud with optional attributes; all arrays tightly packed. */
export interface PointCloud {
  count: number;
  positions: Float32Array; // 3 * count
  colors: Uint8Array; // 3 * count, sRGB
  normals?: Float32Array; // 3 * count, unit
  /** Optional per-point radius (world units) for splat rendering. */
  radii?: Float32Array; // count
  /** Optional per-point confidence in [0,1]. */
  confidence?: Float32Array; // count
}

export interface TriangleMesh {
  vertexCount: number;
  triangleCount: number;
  positions: Float32Array; // 3 * vertexCount
  normals: Float32Array; // 3 * vertexCount
  colors: Uint8Array; // 3 * vertexCount
  indices: Uint32Array; // 3 * triangleCount
}

/** Scale information for the reconstruction. */
export interface MetricInfo {
  /** True when the reconstruction is in metres (from IMU and/or GPS). */
  metric: boolean;
  /** Source of scale. */
  source: 'none' | 'imu' | 'gps' | 'imu+gps';
  /** Estimated relative 1-sigma scale uncertainty, e.g. 0.05 = 5 %. */
  relativeUncertainty: number | null;
  /** Whether +Z of the world frame is aligned with "up" (opposite gravity). */
  gravityAligned: boolean;
  /** Whether +Y points to (true) north (requires compass or GPS). */
  northAligned: boolean;
  /** WGS84 origin of the local ENU frame when georeferenced. */
  originLatLonAlt: [number, number, number] | null;
}

export interface ProgressEvent {
  stage: string;
  /** 0..1 within the stage, or -1 for indeterminate. */
  progress: number;
  message?: string;
}

export type ProgressCallback = (e: ProgressEvent) => void;

/** Cooperative cancellation token passed through long-running work. */
export interface CancelToken {
  readonly cancelled: boolean;
}
