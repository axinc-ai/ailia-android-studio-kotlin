/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

import java.util.EnumSet;

/**
 * Wrapper class for detector.
 */
public class AiliaDetectorModel implements AutoCloseable {
    private long detectorHandle;

	/**
	 * Constructor
	 *
	 * @param netHandle A network instance handle
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @param algorithm {@link AiliaDetectorAlgorithm}
	 * @param categoryCount The number of detection categories (specify 20 for VOC or 80 for COCO, etc.)
	 * @param flags {@link AiliaDetectorFlags}
	 * @throws AiliaException
	 * @see {@link AiliaDetector#Create(long, int, int, int, int, int, int)}
	 */
	public AiliaDetectorModel(long netHandle, AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range, AiliaDetectorAlgorithm algorithm, int categoryCount, EnumSet<AiliaDetectorFlags> flags) throws AiliaException {
        int flag_value = 0;
		for (AiliaDetectorFlags f : flags) {
			flag_value |= f.getValue();
		}
		detectorHandle = AiliaDetector.Create(netHandle, format.getValue(), channel.getValue(), range.getValue(), algorithm.getValue(), categoryCount, flag_value);
    }
	/**
	 * Constructor
	 * @deprecated Prease use {@link #AiliaDetectorModel(long, AiliaNetworkImageFormat, AiliaNetworkImageChannel, AiliaNetworkImageRange, AiliaDetectorAlgorithm, int, EnumSet)}
	 */
    @Deprecated public AiliaDetectorModel(long netHandle, AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range, int algorithm, int categoryCount, int flags) throws AiliaException {
        detectorHandle = AiliaDetector.Create(netHandle, format.getValue(), channel.getValue(), range.getValue(), algorithm, categoryCount, flags);
    }

	/**
	 * Destroys the detector instance.
	 *
	 * @see {@link AiliaDetector#Destroy(long)}
	 */
    public void close() {
        AiliaDetector.Destroy(detectorHandle);
    }

	/**
	 * Performs object detection.
	 *
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @param threshold The detection threshold (for example, 0.1f) (The smaller it is, the easier the detection will be and the more detected objects found.)
	 * @param iou Iou threshold (for example, 0.45f) (The smaller it is, the fewer detected objects found, as duplication is not allowed.)
	 * @throws AiliaException
	 * @see {@link AiliaDetector#Compute(long, byte[], int, int, int, int, float, float)}
	 */
    public void compute(byte[] src, int srcStride, int srcWidth, int srcHeight, AiliaImageFormat srcFormat, float threshold, float iou) throws AiliaException {
        AiliaDetector.Compute(detectorHandle, src, srcStride, srcWidth, srcHeight, srcFormat.getValue(), threshold, iou);
    }

	/**
	 * Gets the number of detection results.
	 *
	 * @return The number of objects
	 * @throws AiliaException
	 * @see {@link AiliaDetector#GetObjectCount(long)}
	 */
    public int getObjectCount() throws AiliaException {
        return AiliaDetector.GetObjectCount(detectorHandle);
    }

	/**
	 * Gets the detection results.
	 *
	 * @param index Object index
	 * @return Object information
	 * @throws AiliaException
	 * @see {@link AiliaDetector#GetObject(long, int, int)}
	 */
    public AiliaDetectorObject getObject(int index) throws AiliaException {
        return AiliaDetector.GetObject(detectorHandle, index, AiliaDetectorObject.version);
    }

	/**
	 * Sets the anchor information (anchors or biases) for YoloV2 or other systems.
	 *
	 * @param anchors The anchor dimensions (the shape, height and width of the detection box)
	 * @throws AiliaException
	 * @see {@link AiliaDetector#SetAnchors(long, float[], int)}
	 */
    public void setAnchors(float[] anchors) throws AiliaException {
        AiliaDetector.SetAnchors(detectorHandle, anchors, anchors.length / 2);
    }
}
