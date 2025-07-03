/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

/**
 * Wrapper class for classifier
 */
public class AiliaClassifierModel implements AutoCloseable {
    private long classifierHandle;

	/**
	 * Constructor
	 *
	 * @param netHandle A network instance handle
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @see {@link AiliaClassifier#Create(long, int, int, int)}
	 * @throws AiliaException
	 */
    public AiliaClassifierModel(long netHandle, AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range) throws AiliaException {
        classifierHandle = AiliaClassifier.Create(netHandle, format.getValue(), channel.getValue(), range.getValue());
    }

	/**
	 * Destroys the classifier instance.
	 *
	 * @see {@link AiliaClassifier#Destroy(long)}
	 */
    public void close() {
        AiliaClassifier.Destroy(classifierHandle);
    }

	/**
	 * Performs object classification.
	 *
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @param maxClassCount The maximum number of classification results
	 * @throws AiliaException
	 * @see {@link AiliaClassifier#Compute(long, byte[], int, int, int, int, int)}
	 */
    public void compute(byte[] src, int srcStride, int srcWidth, int srcHeight, AiliaImageFormat srcFormat, int maxClassCount) throws AiliaException {
        AiliaClassifier.Compute(classifierHandle, src, srcStride, srcWidth, srcHeight, srcFormat.getValue(), maxClassCount);
    }

	/**
	 * Gets the number of classification results.
	 *
	 * @return The number of classes
	 * @throws AiliaException
	 * @see {@link AiliaClassifier#GetClassCount(long)}
	 */
    public int getClassCount() throws AiliaException {
        return AiliaClassifier.GetClassCount(classifierHandle);
    }

	/**
	 * Gets the classification results.
	 *
	 * @param index Index of class
	 * @return Class information
	 * @throws AiliaException
	 * @see {@link AiliaClassifier#GetClass(long, int, int)}
	 */
    public AiliaClassifierClass getClass(int index) throws AiliaException {
        return AiliaClassifier.GetClass(classifierHandle, index, AiliaClassifierClass.version);
    }
}
