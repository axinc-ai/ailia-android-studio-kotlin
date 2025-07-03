/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

/**
 * Wrapper class for feature extractor.
 */
public class AiliaFeatureExtractorModel implements AutoCloseable {
    private static final int SIZEOF_FLOAT = 4;

    private long fextractorHandle;

	/**
	 * Constructor
	 *
	 * @param netHandle A network instance handle
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @param layerName The name of the layer corresponding to the feature (fc1 for VGG16 and NULL for the last layer)
	 * @throws AiliaException
	 * @see {@link AiliaFeatureExtractor#Create(long, int, int, int, String)}
	 */
    public AiliaFeatureExtractorModel(long netHandle, AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range, String layerName) throws AiliaException {
        fextractorHandle = AiliaFeatureExtractor.Create(netHandle, format.getValue(), channel.getValue(), range.getValue(), layerName);
    }

	/**
	 * It destroys the feature extraction instance.
	 *
	 * @see {@link AiliaFeatureExtractor#Destroy(long)}
	 */
    public void close() {
        AiliaFeatureExtractor.Destroy(fextractorHandle);
    }

	/**
	 * Performs feature extraction.
	 *
	 * @param dst The feature data
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @throws AiliaException
	 * @see {@link AiliaFeatureExtractor#Compute(long, float[], int, byte[], int, int, int, int)}
	 */
    public void compute(float[] dst, byte[] src, int srcStride, int srcWidth, int srcHeight, AiliaImageFormat srcFormat) throws AiliaException {
        AiliaFeatureExtractor.Compute(fextractorHandle, dst, dst.length * SIZEOF_FLOAT, src, srcStride, srcWidth, srcHeight, srcFormat.getValue());
    }

	/**
	 * Computes distances in feature space.
	 *
	 * @param distanceType The type of the distance in feature space {@link AiliaFeatureExtractorDistanceType}
	 * @param feature1 The feature
	 * @param feature2 The other feature
	 * @return A distance in feature space
	 * @throws AiliaException
	 * @see {@link AiliaFeatureExtractor#Match(long, int, float[], int, float[], int)}
	 */
	public float match(AiliaFeatureExtractorDistanceType distanceType, float[] feature1, float[] feature2) throws AiliaException {
        return AiliaFeatureExtractor.Match(fextractorHandle, distanceType.getValue(), feature1, feature1.length * SIZEOF_FLOAT, feature2, feature2.length * SIZEOF_FLOAT);
    }

	/**
	 * @deprecated {@link} {@link #match(AiliaFeatureExtractorDistanceType, float[], float[])}
	 */
    @Deprecated public float match(int distanceType, float[] feature1, float[] feature2) throws AiliaException {
        return AiliaFeatureExtractor.Match(fextractorHandle, distanceType, feature1, feature1.length * SIZEOF_FLOAT, feature2, feature2.length * SIZEOF_FLOAT);
    }
}
