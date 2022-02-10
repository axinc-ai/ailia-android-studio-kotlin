/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

/**
 * AILIA object detection library (native API)
 */
public class AiliaDetector {
	/**
	 * @deprecated Please use ({@link AiliaDetectorAlgorithm#YOLOV1})
	 */
    @Deprecated public static final int ALGORITHM_YOLOV1 = 0;

    /**
	 * @deprecated Please use ({@link AiliaDetectorAlgorithm#YOLOV2})
	 */
    @Deprecated public static final int ALGORITHM_YOLOV2 = 1;

	/**
	 * @deprecated Please use ({@link AiliaDetectorAlgorithm#YOLOV3})
	 */
    @Deprecated  public static final int ALGORITHM_YOLOV3 = 2;

	/**
	 * @deprecated Please use ({@link AiliaDetectorAlgorithm#YOLOV4})
	 */
    @Deprecated public static final int ALGORITHM_YOLOV4 = 3;

	/**
	 * @deprecated Please use ({@link AiliaDetectorAlgorithm#SSD})
	 */
    @Deprecated public static final int ALGORITHM_SSD = 8;

	/**
	 * @deprecated Please use ({@link AiliaDetectorFlags#NORMAL})
	 */
    @Deprecated public static final int FLAG_NORMAL = 0;

	/**
	 * Creates a detector instance.
	 *
	 * @param netHandle A network instance handle
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @param algorithm The network algorithm ({@link AiliaDetectorAlgorithm})
	 * @param category_count The number of detection categories (specify 20 for VOC or 80 for COCO, etc.)
	 * @param flags Additional flags ({@link AiliaDetectorFlags})
	 * @return A handle of detector instance
	 * @throws AiliaException
	 */
    public static native long Create(long netHandle, int format, int channel, int range, int algorithm, int category_count, int flags) throws AiliaException;

	/**
	 * Destroys the detector instance.
	 *
	 * @param handle A detector instance handle
	 */
	public static native void Destroy(long handle);

	/**
	 * Performs object detection.
	 *
	 * @param handle A detector instance handle
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @param threshold The detection threshold (for example, 0.1f) (The smaller it is, the easier the detection will be and the more detected objects found.)
	 * @param iou Iou threshold (for example, 0.45f) (The smaller it is, the fewer detected objects found, as duplication is not allowed.)
	 * @throws AiliaException
	 */
    public static native void Compute(long handle, byte[] src, int srcStride, int srcWidth, int srcHeight, int srcFormat, float threshold, float iou) throws AiliaException;

	/**
	 * Gets the number of detection results.
	 *
	 * @param handle A detector instance handle
	 * @return The number of objects
	 * @throws AiliaException
	 */
    public static native int GetObjectCount(long handle) throws AiliaException;

	/**
	 * Gets the detection results.
	 *
	 * <p>If ailiaPredict is not run at all, the method throws {@link AiliaException}.
	 * The detection results are sorted in the order of estimated probability.</p>
	 * @param handle A detector instance handle
	 * @param objIdx Object index
	 * @param version {@link AiliaDetectorObject#version}
	 * @return Object information
	 * @throws AiliaException
	 */
    public static native AiliaDetectorObject GetObject(long handle, int objIdx, int version) throws AiliaException;

	/**
	 * Sets the anchor information (anchors or biases) for YoloV2 or other systems.
	 *
	 * <p>YoloV2 and other systems perform object detection with multiple detection boxes determined during training. By using this API function to set the shape of the detection box determined during training, correct inferences can be made.
	 * The {x, y, x, y ...} format is used for anchor storage.
	 * If {@value anchorsCount} has a value of 5, then anchors is a 10-dimensional array.</p>
	 * @param handle A detector instance handle
	 * @param anchors The anchor dimensions (the shape, height and width of the detection box)
	 * @param anchorsCount The number of anchors (half of the anchors array size)
	 * @throws AiliaException
	 */
    public static native void SetAnchors(long handle, float[] anchors, int anchorsCount) throws AiliaException;

	/**
	 * Sets the size of the input image for YoloV3 model.
	 *
	 * <p>The same YoloV3 model can be used for any input image size that is a multiple of 32.
	 * You can use this API if you want to choose the input image size, for example to reduce the calculation complexity.
	 * It must be called between {@link #Create(long, int, int, int, int, int, int)} and {@link #Compute(long, byte[], int, int, int, int, float, float)}.
	 * If this API is not used, a default size of 416x416 is assumed.
	 * If used with some model other than YoloV3, it will throw {@link AiliaException}</p>
	 * @param handle A detector instance handle
	 * @param input_width Width of the model's input image
	 * @param input_height Height of the model's input image
	 * @throws AiliaException
	 */
    public static native void SetInputShape(long handle, int input_width, int input_height) throws AiliaException;
}
