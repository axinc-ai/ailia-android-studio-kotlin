/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

/**
 * Wrapper class for pose estimator
 */
public class AiliaPoseEstimatorModel implements AutoCloseable
{
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#ACCULUS_POSE}
	 */
	@Deprecated public static final int  ALGORITHM_ACCULUS_POSE = 0;
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#ACCULUS_FACE}
	 */
	@Deprecated public static final int  ALGORITHM_ACCULUS_FACE = 1;
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#ACCULUS_UPPOSE}
	 */
	@Deprecated public static final int  ALGORITHM_ACCULUS_UPPOSE = 2;
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#ACCULUS_UPPOSE_FPGA}
	 */
	@Deprecated public static final int  ALGORITHM_ACCULUS_UPPOSE_FPGA = 3;
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#ACCULUS_HAND}
	 */
	@Deprecated public static final int  ALGORITHM_ACCULUS_HAND = 5;
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#OPEN_POSE}
	 */
	@Deprecated public static final int  ALGORITHM_OPEN_POSE = 10;
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#LW_HUMAN_POSE}
	 */
	@Deprecated public static final int  ALGORITHM_LW_HUMAN_POSE = 11;
    /**
	 * @deprecated Please use {@link AiliaPoseEstimatorAlgorithm#OPEN_POSE_SINGLE_SCALE}
	 */
	@Deprecated public static final int  ALGORITHM_OPEN_POSE_SINGLE_SCALE = 12;

    private long estimatorHandle;

	/**
	 * Creates a estimator instance.
	 *
	 * @param netHandle A network instance handle
	 * @param algorithm {@link AiliaPoseEstimatorAlgorithm}
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#Create(long, int)}
	 */
	public AiliaPoseEstimatorModel(long netHandle, AiliaPoseEstimatorAlgorithm algorithm) throws AiliaException {
        estimatorHandle = AiliaPoseEstimator.Create(netHandle, algorithm.getValue());
    }
	/**
	 * @deprecated Please use {@link #AiliaPoseEstimatorModel(long, AiliaPoseEstimatorAlgorithm)}
	 */
    @Deprecated public AiliaPoseEstimatorModel(long netHandle, int algorithm) throws AiliaException {
        estimatorHandle = AiliaPoseEstimator.Create(netHandle, algorithm);
    }

	/**
	 * Destroys the estimator instance.
	 *
	 * @see {@link AiliaFeatureExtractor#Destroy(long)}
	 */
    public void close()
    {
        AiliaPoseEstimator.Destroy(estimatorHandle);
    }

	/**
	 * Set the detection threshold.
	 *
	 * @param threshold The detection threshold (for example, 0.1f) (The smaller it is, the easier the detection will be and the more detected objects found.)
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#SetThreshold(long, float)}
	 */
	public void setThreshold(float threshold) throws AiliaException
	{
		AiliaPoseEstimator.SetThreshold(estimatorHandle, threshold);
	}

	/**
	 * Performs human pose estimation and human face landmarks extraction.
	 *
	 * @param src Image data (32 bpp)
	 * @param srcStride The number of bytes in 1 line
	 * @param srcWidth Image width
	 * @param srcHeight Image height
	 * @param srcFormat Image format ({@link AiliaImageFormat})
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#Compute(long, byte[], int, int, int, int)}
	 */
    public void compute(byte[] src, int srcStride, int srcWidth, int srcHeight, AiliaImageFormat srcFormat) throws AiliaException
    {
        AiliaPoseEstimator.Compute(estimatorHandle, src, srcStride, srcWidth, srcHeight, srcFormat.getValue());
    }

	/**
	 * Gets the number of detection results.
	 * @return The number of objects. Set to 1 or 0 for human face landmarks.
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#GetObjectCount(long)}
	 */
    public int getObjectCount() throws AiliaException
    {
        return AiliaPoseEstimator.GetObjectCount(estimatorHandle);
    }

	/**
	 * Gets the results of the human pose estimation.
	 *
	 * @param objIndex Object index
	 * @return Object information
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#GetObjectPose(long, int, int)}
	 */
    public AiliaPoseEstimatorObjectPose getObjectPose(int objIndex) throws AiliaException
    {
        return AiliaPoseEstimator.GetObjectPose(estimatorHandle, objIndex, AiliaPoseEstimatorObjectPose.version);
    }

	/**
	 * Gets the results of the human face landmarks extraction.
	 *
	 * @param objIndex Object index
	 * @return Object information
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#GetObjectFace(long, int, int)}
	 */
    public AiliaPoseEstimatorObjectFace getObjectFace(int objIndex) throws AiliaException
    {
        return AiliaPoseEstimator.GetObjectFace(estimatorHandle, objIndex, AiliaPoseEstimatorObjectFace.version);
    }

	/**
	 * Gets the results of the human up pose estimation.
	 *
	 * @param objIndex Object index
	 * @return Object information
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#GetObjectUpPose(long, int, int)}
	 */
    public AiliaPoseEstimatorObjectUpPose getObjectUpPose(int objIndex) throws AiliaException
    {
        return AiliaPoseEstimator.GetObjectUpPose(estimatorHandle, objIndex, AiliaPoseEstimatorObjectUpPose.version);
    }
}
