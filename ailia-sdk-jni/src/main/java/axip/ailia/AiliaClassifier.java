/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

/**
 * AILIA object classification library (native API)
 */
public class AiliaClassifier {
	/**
	 * Creates a classifier instance.
	 *
	 * @param netHandle A network instance handle
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @return A handle of classifier instance
	 * @throws AiliaException
	 */
    public static native long Create(long netHandle, int format, int channel, int range) throws AiliaException;

	/**
	 * Destroys the classifier instance.
	 *
	 * @param handle A classifier instance handle
	 */
    public static native void Destroy(long handle);

	/**
	 * Performs object classification.
	 *
	 * @param handle A classifier instance handle
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @param maxClassCount The maximum number of classification results
	 * @throws AiliaException
	 */
    public static native void Compute(long handle, byte[] src, int srcStride, int srcWidth, int srcHeight, int srcFormat, int maxClassCount) throws AiliaException;

	/**
	 * Gets the number of classification results.
	 *
	 * @param handle A classifier instance handle
	 * @return The number of classes
	 * @throws AiliaException
	 */
	public static native int GetClassCount(long handle) throws AiliaException;

	/**
	 * Gets the classification results.
	 *
	 * <p>If ailiaPredict is not run at all, the method throws {@link AiliaException}
	 * The classification results are sorted in the order of estimated probability.</p>
	 * @param handle A classifier instance handle
	 * @param clsIdx Index of class
	 * @param version {@link AiliaClassifierClass#version}
	 * @return Class information
	 * @throws AiliaException
	 */
    public static native AiliaClassifierClass GetClass(long handle, int clsIdx, int version) throws AiliaException;
}
