/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

/**
 * AILIA feature extraction library (native API)
 */
public class AiliaFeatureExtractor {
	/**
	 * @deprecated Prease use {@link AiliaFeatureExtractorDistanceType#L2NORM}
	 */
    @Deprecated public static final int DISTANCE_L2NORM = 0;

	/**
	 * Creates a feature extraction instance.
	 *
	 * @param netHandle A network instance handle
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @param layerName The name of the layer corresponding to the feature (fc1 for VGG16 and NULL for the last layer)
	 * @return A feature extraction instance handle
	 * @throws AiliaException
	 */
    public static native long Create(long netHandle, int format, int channel, int range, String layerName) throws AiliaException;

	/**
	 * It destroys the feature extraction instance.
	 *
	 * @param handle A feature extraction instance handle
	 */
    public static native void Destroy(long handle);

	/**
	 * Performs feature extraction.
	 *
	 * @param handle A feature extraction instance handle
	 * @param dst The feature data
	 * @param dstSize The size of the dst (in byte)
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @throws AiliaException
	 */
    public static native void Compute(long handle, float[] dst, int dstSize, byte[] src, int srcStride, int srcWidth, int srcHeight, int srcFormat) throws AiliaException;

	/**
	 * Computes distances in feature space.
	 *
	 * @param handle A feature extraction instance handle
	 * @param distanceType The type of the distance in feature space {@link AiliaFeatureExtractorDistanceType}
	 * @param feature1 The feature
	 * @param feature1Size The size of the feature1 (in bytes)
	 * @param feature2 The other feature
	 * @param feature2Size The size of the feature2 (in bytes)
	 * @return A distance in feature space
	 * @throws AiliaException
	 */
    public static native float Match(long handle, int distanceType, float[] feature1, int feature1Size, float[] feature2, int feature2Size) throws AiliaException;
}
