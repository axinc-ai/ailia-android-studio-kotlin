/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

/**
 * AILIA library for human pose estimation and human face landmarks extraction (native API)
 */
public class AiliaPoseEstimator
{
    /**
	 * Creates a estimator instance.
	 *
	 * @param netHandle A network instance handle
	 * @param algorithm {@link AiliaPoseEstimatorAlgorithm}
	 * @return An estimator instance handle
	 * @throws AiliaException
	 */
	public static native long Create(long netHandle, int algorithm) throws AiliaException;

	/**
	 * Destroys the estimator instance.
	 *
	 * @param handle An estimator instance handle
	 */
	public static native void Destroy(long handle);

	/**
	 * Set the detection threshold.
	 *
	 * @param handle An estimator instance handle
	 * @param threshold The detection threshold (for example, 0.1f) (The smaller it is, the easier the detection will be and the more detected objects found.)
	 * @throws AiliaException
	 */
	public static native void SetThreshold(long handle, float threshold) throws AiliaException;

	/**
	 * Performs human pose estimation and human face landmarks extraction.
	 *
	 * @param handle An estimator instance handle
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @throws AiliaException
	 */
	public static native void Compute(long handle, byte[] src, int srcStride, int srcWidth, int srcHeight, int srcFormat) throws AiliaException;

	/**
	 * Gets the number of detection results.
	 *
	 * @param handle An estimator instance handle
	 * @return The number of objects. Set to 1 or 0 for human face landmarks.
	 * @throws AiliaException
	 */
	public static native int GetObjectCount(long handle) throws AiliaException;

	/**
	 * Gets the results of the human pose estimation.
	 *
	 * @param handle An estimator instance handle
	 * @param objIndex Object index
	 * @param version {@link AiliaPoseEstimatorObjectPose#version}
	 * @return Object information
	 * @throws AiliaException
	 */
	public static native AiliaPoseEstimatorObjectPose GetObjectPose(long handle, int objIndex, int version) throws AiliaException;

	/**
	 * Gets the results of the human face landmarks extraction.
	 *
	 * @param handle An estimator instance handle
	 * @param objIndex Object index
	 * @param version {@link AiliaPoseEstimatorObjectFace#version}
	 * @return Object information
	 * @throws AiliaException
	 */
	public static native AiliaPoseEstimatorObjectFace GetObjectFace(long handle, int objIndex, int version) throws AiliaException;

	/**
	 * Gets the results of the human up pose estimation.
	 *
	 * @param handle An estimator instance handle
	 * @param objIndex Object index
	 * @param version {@link AiliaPoseEstimatorObjectUpPose#version}
	 * @return Object information
	 * @throws AiliaException
	 */
	public static native AiliaPoseEstimatorObjectUpPose GetObjectUpPose(long handle, int objIndex, int version) throws AiliaException;
}
