/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

import java.util.EnumSet;
import java.util.LinkedList;
import java.util.List;

/**
 * Wrapper class for Ailia.
 */
public class AiliaModel implements AutoCloseable {
	private long netHandle;

	/**
	 * Gets the available computational environments.
	 * @return List of Environment
	 * @throws AiliaException
	 */
	static public List<AiliaEnvironment> getEnvironments() throws AiliaException {
		int len = Ailia.GetEnvironmentCount();
		LinkedList<AiliaEnvironment> envs = new LinkedList<>();
		for (int i = 0; i < len; i++) {
			envs.push(Ailia.GetEnvironment(i, AiliaEnvironment.version));
		}
		return envs;
	}

	/**
	 * Create mode flags for {@link Ailia#SetMemoryMode(long, int)} / Constructor.
	 * @param reduceConstant Releases the intermediate buffer that is a constant such as weight
	 * @param ignoreInputWithInitializer Disable the input specified initializer and release the intermediate buffer that becomes a constant such as weight.
	 * @param reduceInterstage Release intermediate buffer during inference
	 * @param reuseInterstage Infer by sharing the intermediate buffer.
	 * @return Enum set of {@link AiliaMemoryMode}
	 * @see {@link Ailia#SetMemoryMode(long, int)}
	 * @throws AiliaException
	 */
	static public EnumSet<AiliaMemoryMode> getMemoryMode(boolean reduceConstant, boolean ignoreInputWithInitializer, boolean reduceInterstage, boolean reuseInterstage) throws AiliaException {
		EnumSet<AiliaMemoryMode> flag = EnumSet.noneOf(AiliaMemoryMode.class);
		if(reduceConstant) flag.add(AiliaMemoryMode.REDUCE_CONSTANT);
		if(ignoreInputWithInitializer) flag.add(AiliaMemoryMode.REDUCE_CONSTANT_WITH_INPUT_INITIALIZER);
		if(reduceInterstage) flag.add(AiliaMemoryMode.REDUCE_INTERSTAGE);
		if(reuseInterstage) flag.add(AiliaMemoryMode.REUSE_INTERSTAGE);
		return flag;
	}

	private void setMemoryMode(EnumSet<AiliaMemoryMode> memoryMode) throws AiliaException {
		int mode = 0;
		for (AiliaMemoryMode m : memoryMode) {
			mode |= m.getValue();
		}
		Ailia.SetMemoryMode(netHandle, mode);
	}

	private void init(int envId, int numThread, EnumSet<AiliaMemoryMode> memoryMode, byte[] streamData, byte[] weightData) throws AiliaException {
		netHandle = Ailia.Create(envId, numThread);
		this.setMemoryMode(memoryMode);
		Ailia.OpenStreamMem(netHandle, streamData, streamData.length);
		Ailia.OpenWeightMem(netHandle, weightData, weightData.length);
	}

	private void init(int envId, int numThread, EnumSet<AiliaMemoryMode> memoryMode, String streamPath, String weightPath) throws AiliaException {
		netHandle = Ailia.Create(envId, numThread);
		this.setMemoryMode(memoryMode);
		Ailia.OpenStreamFile(netHandle, streamPath);
		Ailia.OpenWeightFile(netHandle, weightPath);
	}

	/**
	 * Constructor
	 *
	 * @param envId The ID of the inference backend used for computation (obtained by {@link #getEnvironments()})
	 * @param numThread The upper limit on the number of threads
	 * @param streamData An array of the data in the protobuf file
	 * @param weightData An array of the data in the protobuf/onnx file
	 * @throws AiliaException
	 * @see {@link Ailia#Create(int, int)}
	 * @see {@link Ailia#OpenStreamMem(long, byte[], int)}
	 * @see {@link Ailia#OpenWeightMem(long, byte[], int)}
	 */
	public AiliaModel(int envId, int numThread, byte[] streamData, byte[] weightData) throws AiliaException {
		init(envId, numThread, EnumSet.of(AiliaMemoryMode.OPTIMAIZE_DEFAULT), streamData, weightData);
	}

	/**
	 * Constructor
	 *
	 * @param envId The ID of the inference backend used for computation (obtained by {@link #getEnvironments()})
	 * @param numThread The upper limit on the number of threads
	 * @param streamPath The path name to the prototxt file
	 * @param weightPath The path name to the protobuf/onnx file
	 * @throws AiliaException
	 * @see {@link Ailia#Create(int, int)}
	 * @see {@link Ailia#OpenStreamFile(long, String)}
	 * @see {@link Ailia#OpenWeightFile(long, String)}
	 */
	public AiliaModel(int envId, int numThread, String streamPath, String weightPath) throws AiliaException {
		init(envId, numThread, EnumSet.of(AiliaMemoryMode.OPTIMAIZE_DEFAULT), streamPath, weightPath);
	}

	/**
	 * Constructor
	 *
	 * @param envId The ID of the inference backend used for computation (obtained by {@link #getEnvironments()})
	 * @param numThread The upper limit on the number of threads
	 * @param memoryMode Memory mode (Calucurate by {@link #getMemoryMode(boolean, boolean, boolean, boolean)})
	 * @param streamData An array of the data in the protobuf file
	 * @param weightData An array of the data in the protobuf/onnx file
	 * @throws AiliaException
	 * @see {@link Ailia#Create(int, int)}
	 * @see {@link Ailia#OpenStreamMem(long, byte[], int)}
	 * @see {@link Ailia#OpenWeightMem(long, byte[], int)}
	 */
	public AiliaModel(int envId, int numThread, EnumSet<AiliaMemoryMode> memoryMode, byte[] streamData, byte[] weightData) throws AiliaException {
		init(envId, numThread, memoryMode, streamData, weightData);
	}

	/**
	 * Constructor
	 *
	 * @param envId The ID of the inference backend used for computation (obtained by {@link #getEnvironments()})
	 * @param numThread The upper limit on the number of threads
	 * @param memoryMode Memory mode (Calucurate by {@link #getMemoryMode(boolean, boolean, boolean, boolean)})
	 * @param streamPath The path name to the prototxt file
	 * @param weightPath The path name to the protobuf/onnx file
	 * @throws AiliaException
	 * @see {@link Ailia#Create(int, int)}
	 * @see {@link Ailia#OpenStreamFile(long, String)}
	 * @see {@link Ailia#OpenWeightFile(long, String)}
	 */
	public AiliaModel(int envId, int numThread, EnumSet<AiliaMemoryMode> memoryMode, String streamPath, String weightPath) throws AiliaException {
		init(envId, numThread, memoryMode, streamPath, weightPath);
	}

	/**
	 * Creates a classifier object.
	 *
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @return The instance of a classfier wrapper class.
	 * @throws AiliaException
	 * @see {@link AiliaClassifier#Create(long, int, int, int)}
	 */
	public AiliaClassifierModel createClassifier(AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range) throws AiliaException {
		return new AiliaClassifierModel(netHandle, format, channel, range);
	}

	/**
	 * Creates a detector object.
	 *
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @param algorithm The network algorithm {@link AiliaDetectorAlgorithm}
	 * @param categoryCount The number of detection categories (specify 20 for VOC or 80 for COCO, etc.)
	 * @param flags Additional flags {@link AiliaDetectorFlags}
	 * @return The instance of a detector wrapper class.
	 * @throws AiliaException
	 * @see {@link AiliaDetector#Create(long, int, int, int, int, int, int)}
	 */
	public AiliaDetectorModel createDetector(AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range, AiliaDetectorAlgorithm algorithm, int categoryCount, EnumSet<AiliaDetectorFlags> flags) throws AiliaException {
		return new AiliaDetectorModel(netHandle, format, channel, range, algorithm, categoryCount, flags);
	}

	/**
	 * @deprecated Please use {@link #createDetector(AiliaNetworkImageFormat, AiliaNetworkImageChannel, AiliaNetworkImageRange, AiliaDetectorAlgorithm, int, EnumSet<AiliaDetectorFlags>)}
	 */
	@Deprecated public AiliaDetectorModel createDetector(AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range, int algorithm, int categoryCount, int flags) throws AiliaException {
		return new AiliaDetectorModel(netHandle, format, channel, range, algorithm, categoryCount, flags);
	}

	/**
	 * Creates a feature extractor object.
	 *
	 * @param format The network image format ({@link AiliaNetworkImageFormat})
	 * @param channel The network image channel ({@link AiliaNetworkImageChannel})
	 * @param range The network image range ({@link AiliaNetworkImageRange})
	 * @param layerName The name of the layer corresponding to the feature (fc1 for VGG16 and NULL for the last layer)
	 * @return The instance of a feature extractor wrapper class.
	 * @throws AiliaException
	 * @see {@link AiliaFeatureExtractor#Create(long, int, int, int, String)}
	 */
	public AiliaFeatureExtractorModel createFeatureExtractor(AiliaNetworkImageFormat format, AiliaNetworkImageChannel channel, AiliaNetworkImageRange range, String layerName) throws AiliaException {
		return new AiliaFeatureExtractorModel(netHandle, format, channel, range, layerName);
	}

	/**
	 * Creates a pose estimator object.
	 *
	 * @param algorithm {@link AiliaPoseEstimatorAlgorithm}
	 * @return The instance of pose estimtor wrapper class.
	 * @throws AiliaException
	 * @see {@link AiliaPoseEstimator#Create(long, int)}
	 */
	public AiliaPoseEstimatorModel createPoseEstimator(AiliaPoseEstimatorAlgorithm algorithm) throws AiliaException {
		return new AiliaPoseEstimatorModel(netHandle, algorithm);
	}

	/**
	 * @deprecated Psease use {@link #createPoseEstimator(AiliaPoseEstimatorAlgorithm)}
	 */
	@Deprecated public AiliaPoseEstimatorModel createPoseEstimator(int algorithm) throws AiliaException {
		return new AiliaPoseEstimatorModel(netHandle, algorithm);
	}

	/**
	 * Performs the inferences and provides the inference result.
	 *
	 * @param dest The result is stored in the inference result destination buffer as float data in the order of X, Y, Z, and W. The buffer has the same size as the network file outputSize.
	 * @param destSize The size of the dest (in byte)
	 * @param src The input is stored as float data in the order of the inference data X, Y, Z, and W. The input has the same size as the network file inputSize.
	 * @param srcSize The size of the src (in byte)
	 * @throws AiliaException
	 * @see {@link Ailia#Predict(long, float[], int, float[], int)}
	 */
	public void predict(float[] dest, int destSize, float[] src, int srcSize) throws AiliaException {
		Ailia.Predict(netHandle,dest,destSize,src,srcSize);
	}

	/**
	 * @deprecated Psease use {@link #predict(float[], int, float[], int)}
	 */
	@Deprecated public void Predict(float[] dest, int destSize, float[] src, int srcSize) throws AiliaException {
		Ailia.Predict(netHandle,dest,destSize,src,srcSize);
	}

	/**
	 * Changes the shape of the input data during inference.
	 *
	 * @param shape Shape information for the input data
	 * @throws AiliaException
	 * @see {@link Ailia#SetInputShape(long, AiliaShape, int)}
	 */
	public void setInputShape(AiliaShape shape) throws AiliaException {
		Ailia.SetInputShape(netHandle, shape, AiliaShape.version);
	}

	/**
	 * Changes the shape of the input data during inference.
	 *
	 * @param shape An array of shape
	 * @throws AiliaException
	 * @see {@link Ailia#SetInputShapeND(long, int[], int)}
	 */
	public void setInputShapeND(int[] shape) throws AiliaException {
		Ailia.SetInputShapeND(netHandle, shape, shape.length);
	}

	/**
	 * Gets the shape of the input data during inference.
	 *
	 * @return The shape of the input data
	 * @throws AiliaException
	 * @see {@link Ailia#GetInputShape(long, int)}
	 */
	public AiliaShape getInputShape() throws AiliaException {
		return Ailia.GetInputShape(netHandle, AiliaShape.version);
	}

	/**
	 * Gets the shape of the input data during inference.
	 *
	 * @return The array of shape
	 * @throws AiliaException
	 * @see {@link Ailia#GetInputDim(long)}
	 * @see {@link Ailia#GetInputShapeND(long, int[], int)}
	 */
	public int[] getInputShapeND() throws AiliaException {
		int dim = Ailia.GetInputDim(netHandle);
		int[] shape = new int[dim];
		Ailia.GetInputShapeND(netHandle, shape, dim);
		return shape;
	}

	/**
	 * Change the shape of the blob given by its index
	 *
	 * @param shape New shape of the blob
	 * @param blobIndex Index referencing the blob to reshape
	 * @param version Shape version({@link AiliaShape#version})
	 * @throws AiliaException
	 * @see {@link Ailia#SetInputBlobShape(long, AiliaShape, int, int)}
	 */
	public void setInputBlobShape(AiliaShape shape, int blobIndex, int version) throws AiliaException {
		Ailia.SetInputBlobShape(netHandle, shape, blobIndex, version);
	}

	/**
	 * Change the shape of the blob given by its index
	 *
	 * @param shape The array of the new shape.
	 * @param blobIndex Index referencing the blob to reshape
	 * @throws AiliaException
	 * @see {@link Ailia#SetInputBlobShapeND(long, int[], int, int)}
	 */
	public void setInputBlobShapeND(int[] shape, int blobIndex) throws AiliaException {
		Ailia.SetInputBlobShapeND(netHandle, shape, shape.length, blobIndex);
	}

	/**
	 * Gets the shape of the internal data (blob) during inference.
	 *
	 * @param blobIndex Index referencing the blob to reshape
	 * @return The shape of the internal data (blob).
	 * @throws AiliaException
	 * @see {@link Ailia#GetBlobShape(long, int, int)}
	 */
	public AiliaShape getBlobShape(int blobIndex) throws AiliaException {
		return Ailia.GetBlobShape(netHandle, blobIndex, AiliaShape.version);
	}

	/**
	 * Gets the shape of the internal data (blob) during inference.
	 * @param blobIndex Index referencing the blob to reshape
	 * @return The array of shape
	 * @throws AiliaException
	 * @see {@link Ailia#GetBlobShapeND(long, int[], int, int)}
	 */
	public int[] getBlobShapeND(int blobIndex) throws AiliaException {
		int dim = Ailia.GetBlobDim(netHandle, blobIndex);
		int[] shape = new int[dim];
		Ailia.GetBlobShapeND(netHandle, shape, dim, blobIndex);
		return shape;
	}

	/**
	 * Gets the shape of the output data during inference.
	 *
	 * @return The shape of the output.
	 * @throws AiliaException
	 * @see {@link Ailia#GetOutputShape(long, int)}
	 */
	public AiliaShape getOutputShape() throws AiliaException {
		return Ailia.GetOutputShape(netHandle, AiliaShape.version);
	}

	/**
	 * Gets the shape of the output data during inference.
	 *
	 * @return The array of the shape
	 * @throws AiliaException
	 * @see {@link Ailia#GetOutputShapeND(long, int[], int)}
	 */
	public int[] getOutputShapeND() throws AiliaException {
		int dim = Ailia.GetOutputDim(netHandle);
		int[] shape = new int[dim];
		Ailia.GetOutputShapeND(netHandle, shape, dim);
		return shape;
	}

	/**
	 * Get the number of input data blobs.
	 *
	 * @return The number of input blobs
	 * @throws AiliaException
	 * @see {@link Ailia#GetInputBlobCount(long)}
	 */
	public int getInputBlobCount() throws AiliaException {
		return Ailia.GetInputBlobCount(netHandle);
	}

	/**
	 * Get the blob index of the input data.
	 *
	 * @param inputBlobIndex Index among the input blobs (between 0 and {@link #getInputBlobCount()}-1)
	 * @return Index of the blob
	 * @throws AiliaException
	 * @see {@link Ailia#GetBlobIndexByInputIndex(long, int)}
	 */
	public int getBlobIndexByInputIndex(int inputBlobIndex) throws AiliaException {
		return Ailia.GetBlobIndexByInputIndex(netHandle, inputBlobIndex);
	}

	/**
	 * Get the number of output data blobs.
	 *
	 * @return The number of output blobs
	 * @throws AiliaException
	 * @see {@link Ailia#GetOutputBlobCount(long)}
	 */
	public int getOutputBlobCount() throws AiliaException {
		return Ailia.GetOutputBlobCount(netHandle);
	}

	/**
	 * Get the blob index of the output data blob.
	 *
	 * @param outputBlobIndex Index among output blobs (between 0 and {@link #getOutputBlobCount()}-1)
	 * @return Blob index
	 * @throws AiliaException
	 * @see {@link Ailia#GetBlobIndexByOutputIndex(long, int)}
	 */
	public int getBlobIndexByOutputIndex(int outputBlobIndex) throws AiliaException {
		return Ailia.GetBlobIndexByOutputIndex(netHandle, outputBlobIndex);
	}

	/**
	 * It destroys the network instance.
	 * @see {@link Ailia#Destroy(long)}
	 */
	public void close() {
		Ailia.Destroy(netHandle);
	}

	/**
	 * Gets A network instance handle
	 * @return A network instance handle
	 */
	public long getHandle() {
		return netHandle;
	}

	/**
	 * Performs the inferences
	 * @throws AiliaException
	 */
	public void update() throws AiliaException {
		Ailia.Update(netHandle);
	}

	/**
	 * Set the input blob data.
	 *
	 * @param shape Shape information for the input data
	 * @param src The input is stored as float data in the order of the inference data X, Y, Z, and W. The input has the same size as the network file inputSize.
	 * @param srcSize The size of the src (in byte)
	 * @param idx The input idx of input blob
	 * @throws AiliaException
	 */
	public void setInputBlobData(float[] src, int srcSize, int idx) throws AiliaException {
		Ailia.SetInputBlobData(netHandle, src, srcSize, idx);
	}

	/**
	 * Get the output blob data.
	 *
	 * @param dest The result is stored in the inference result destination buffer as float data in the order of X, Y, Z, and W. The buffer has the same size as the network file outputSize.
	 * @param destSize The size of the dest (in byte)
	 * @param idx The input idx of blob
	 * @throws AiliaException
	 */
	public void getBlobData(float[] dest, int destSize, int idx) throws AiliaException {
		Ailia.GetBlobData(netHandle, dest, destSize, idx);
	}

	/**
	 * Get the error detail
	 *
	 * @param dest The result is stored in the inference result destination buffer as float data in the order of X, Y, Z, and W. The buffer has the same size as the network file outputSize.
	 * @param destSize The size of the dest (in byte)
	 * @param idx The input idx of blob
	 * @throws AiliaException
	 */
	public String getErrorDetail() throws AiliaException {
		return Ailia.GetErrorDetail(netHandle);
	}
}
