/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;
/**
 * AILIA inference library (native API)
 */
public class Ailia {
	/**
	 * Number of threads
	 */
	public static final int MULTITHREAD_AUTO = 0;
	/**
	 * Automatic setup of the inference backend
	 */
	public static final int ENVIRONMENT_ID_AUTO = -1;

	/**
	 * Do not release the intermediate buffer
	 * @deprecated Please use {@link AiliaMemoryMode#NO_OPTIMIZATION}
	 */
	@Deprecated public static final int MEMORY_NO_OPTIMIZATION                        = 0;
	/**
	 * Releases the intermediate buffer that is a constant such as weight
	 * @deprecated Please use {@link AiliaMemoryMode#REDUCE_CONSTANT}
	 */
	@Deprecated public static final int MEMORY_REDUCE_CONSTANT                        = 1;
	/**
	 * Disable the input specified initializer and release the intermediate buffer that becomes a constant such as weight.
	 * @deprecated Please use {@link AiliaMemoryMode#REDUCE_CONSTANT_WITH_INPUT_INITIALIZER}
	 */
	@Deprecated public static final int MEMORY_REDUCE_CONSTANT_WITH_INPUT_INITIALIZER = 2;
	/**
	 * Release intermediate buffer during inference
	 * @deprecated Please use {@link AiliaMemoryMode#REDUCE_INTERSTAGE}
	 */
	@Deprecated public static final int MEMORY_REDUCE_INTERSTAGE                      = 4;
	/**
	 * Infer by sharing the intermediate buffer. When used with {@link #MEMORY_REDUCE_INTERSTAGE}, the sharable intermediate buffer is not opened.
	 * @deprecated Please use {@link AiliaMemoryMode#REUSE_INTERSTAGE}
	 */
	@Deprecated public static final int MEMORY_REUSE_INTERSTAGE                       = 8;
	/**
	 *
	 * @deprecated Please use {@link AiliaMemoryMode#OPTIMAIZE_DEFAULT}
	 */
	@Deprecated public static final int MEMORY_OPTIMAIZE_DEFAULT					  = MEMORY_REDUCE_CONSTANT;

	/**
	 * Creates a network instance.
	 *
	 * <p>If the inference backend is set to automatic, CPU mode is used, while if BLAS is available, it uses BLAS.
	 * Note that if BLAS is used, num_thread may be ignored.</p>
	 * @param envId The ID of the inference backend used for computation (obtained by {@link #GetEnvironment(int, int)}). It is selected automatically if {@link #ENVIRONMENT_ID_AUTO} is specified.
	 * @param numThread The upper limit on the number of threads (It is set automatically if {@link #MULTITHREAD_AUTO} is specified.)
	 * @return Handle of network instance.
	 * @throws AiliaException
	 */
	public static native long Create(int envId, int numThread) throws AiliaException;

	/**
	 * Initializes the network instance. (Read from file)
	 *
	 * <p>This function reads the network instance from a file and initializes it.</p>
	 * @param handle A network instance handle
	 * @param path The path name to the prototxt file
	 * @throws AiliaException
	 */
	public static native void OpenStreamFile(long handle, String path) throws AiliaException;

	/**
	 * Initializes the network instance. (Read from memory)
	 *
	 * <p>This function reads the network instance from memory and initializes it.</p>
	 * @param handle A network instance handle
	 * @param buf An array of the data in the protobuf file
	 * @param buf_size The data size of the prototxt file
	 * @throws AiliaException
	 */
	public static native void OpenStreamMem(long handle, byte[] buf, int buf_size) throws AiliaException;

	/**
	 * Reads weights into a network instance. (Read from file)
	 *
	 * <p>This function reads weights into the network instance from a file.</p>
	 * @param handle A network instance handle
	 * @param path The path name to the protobuf/onnx file
	 * @throws AiliaException
	 */
	public static native void OpenWeightFile(long handle, String path) throws AiliaException;

	/**
	 * Reads weights into a network instance. (Read from memory)
	 *
	 * This function reads weights into the network instance from memory.
	 * @param handle A network instance handle
	 * @param buf An array of the data in the protobuf/onnx file
	 * @param buf_size The data size of the protobuf/onnx file
	 * @throws AiliaException
	 */
	public static native void OpenWeightMem(long handle, byte[] buf, int buf_size) throws AiliaException;

	/**
	 * It destroys the network instance.
	 *
	 * @param handle A network instance handle
	 */
	public static native void Destroy(long handle);

	/**
	 * Changes the shape of the input data during inference.
	 *
	 * <p>This function changes the input shape defined in prototxt.
	 * The shape must have the same rank as the one contained in prototxt.
	 * Note that {@link AiliaException} may be throw if the weights are dependent on the input shapes, among other reasons.
	 * The dimension of shape that defined in prototxt has 5 or more, please use {@link #SetInputShapeND(long, int[], int)}.</p>
	 * @param handle A network instance handle
	 * @param shape Shape information for the input data
	 * @param version Shape version({@link AiliaShape#version})
	 * @throws AiliaException
	 */
	public static native void SetInputShape(long handle, AiliaShape shape, int version) throws AiliaException;

	/**
	 * Changes the shape of the input data during inference.
	 *
	 * <p>This function changes the input shape defined in prototxt.
	 * The shape must have the same rank as the one contained in prototxt.
	 * Note that {@link AiliaException} may be throw if the weights are dependent on the input shapes, among other reasons.</p>
	 * @param handle A network instance handle
	 * @param shape An array of shape
	 * @param dim The size of shape.
	 * @throws AiliaException
	 */
	public static native void SetInputShapeND(long handle, int[] shape, int dim) throws AiliaException;

	/**
	 * Gets the shape of the input data during inference.
	 *
	 * <p>When dimension of shape is 5 or more, please use {@link #GetInputDim(long)} and {@link #GetInputShapeND(long, int[], int)}.
	 * When shape is not settled, this method stores 0 at unsettled dimension and otherwise stores valid value.</p>
	 * @param handle A network instance handle
	 * @param version Shape version({@link AiliaShape#version})
	 * @return The shape of the input data
	 * @throws AiliaException
	 */
	public static native AiliaShape GetInputShape(long handle, int version) throws AiliaException;

	/**
	 * Gets the dimension of the input data during inference.
	 *
	 * @param handle A network instance handle
	 * @return The dimension of the input data.
	 */
	public static native int GetInputDim(long handle);

	/**
	 * Gets the shape of the input data during inference.
	 *
	 * <p>When shape is not settled, this function stores 0 at unsettled dimension and otherwise stores valid value.</p>
	 * @param handle A network instance handle
	 * @param shape The storage array of the shape. (It stores dim-1, dim-2, ... ,1, 0 order.)
	 * @param dim The size of shape
	 */
	public static native void GetInputShapeND(long handle, int[] shape, int dim);

	/**
	 * Gets the shape of the output data during inference.
	 *
	 * <p>When dimension of shape is 5 or more, please use {@link #GetOutputDim(long)} and {@link #GetOutputShapeND(long, int[], int)}.</p>
	 * @param handle A network instance handle
	 * @param version Shape version({@link AiliaShape#version})
	 * @return The shape of the output.
	 * @throws AiliaException
	 */
	public static native AiliaShape GetOutputShape(long handle, int version) throws AiliaException;

	/**
	 * Gets the dimension of the output data during inference.
	 *
	 * @param handle A network instance handle
	 * @return The dimension of the output.
	 */
	public static native int GetOutputDim(long handle);

	/**
	 * Gets the shape of the output data during inference.
	 *
	 * @param handle A network instance handle
	 * @param shape The storage array of the shape. (It stores dim-1, dim-2, ... ,1, 0 order.)
	 * @param dim The size of shape
	 */
	public static native void GetOutputShapeND(long handle, int[] shape, int dim);

	/**
	 * Performs the inferences and provides the inference result.
	 *
	 * @param handle A network instance handle
	 * @param dest The result is stored in the inference result destination buffer as float data in the order of X, Y, Z, and W. The buffer has the same size as the network file outputSize.
	 * @param destSize The size of the dest (in byte)
	 * @param src The input is stored as float data in the order of the inference data X, Y, Z, and W. The input has the same size as the network file inputSize.
	 * @param srcSize The size of the src (in byte)
	 * @throws AiliaException
	 */
	public static native void Predict(long handle, float[] dest, int destSize, float[] src, int srcSize) throws AiliaException;

	/**
	 * Gets the amount of internal data (blob) during inference.
	 *
	 * @param handle A network instance handle
	 * @return The number of blobs
	 * @throws AiliaException
	 */
	public static native int GetBlobCount(long handle) throws AiliaException;

	/**
	 * Gets the shape of the internal data (blob) during inference.
	 *
	 * @param handle A network instance handle
	 * @param blobIdx The index of the blob (0 to {@link #GetBlobCount(long)}-1)
	 * @param version Shape version({@link AiliaShape#version})
	 * @return The shape of the internal data (blob).
	 * @throws AiliaException
	 */
	public static native AiliaShape GetBlobShape(long handle, int blobIdx, int version) throws AiliaException;

	/**
	 * Gets the dimension of the internal data (blob) during inference.
	 *
	 * @param handle A network instance handle
	 * @param blobIdx The index of the blob (0 to {@link #GetBlobCount(long)}-1)
	 * @return The dimension of the internal data (blob).
	 */
	public static native int GetBlobDim(long handle, int blobIdx);

	/**
	 * Gets the shape of internal data (blob) during inference.
	 *
	 * @param handle A network instance handle
	 * @param shape The storage array of the shape. (It stores dim-1, dim-2, ... ,1, 0 order.)
	 * @param dim The size of shape
	 * @param blobIdx The index of the blob (0 to {@link #GetBlobCount(long)}-1)
	 */
	public static native void GetBlobShapeND(long handle, int[] shape, int dim, int blobIdx);

	/**
	 * Gets the internal data (blob) during inference.
	 *
	 * @param handle A network instance handle
	 * @param dest The result is stored in the inference result destination buffer as float data in the order of X, Y, Z, and W.
	 * @param destSize The size of the dest (in byte)
	 * @param blobIdx The index of the blob (0 to {@link #GetBlobCount(long)}-1)
	 * @throws AiliaException
	 */
	public static native void GetBlobData(long handle, float[] dest, int destSize, int blobIdx) throws AiliaException;

	/**
	 * Searches by name for the index of the internal data (blob) during inference and returns it.
	 *
	 * @param handle A network instance handle
	 * @param name The name of the blob to search for
	 * @return The index of the blob (0 to {@link #GetBlobCount(long)}-1)
	 * @throws AiliaException
	 */
	public static native int FindBlobIndexByName(long handle, String name) throws AiliaException;

	/**
	 * Searches by index for the name of the internal data (blob) during inference and returns it.
	 *
	 * @param handle A network instance handle
	 * @param blobIdx The index of the blob (0 to {@link #GetBlobCount(long)}-1)
	 * @return The blob name
	 * @throws AiliaException
	 */
	public static native String FindBlobNameByIndex(long handle, int blobIdx) throws AiliaException;

	/**
	 * Shows the name and shape of each blob.
	 *
	 * @param handle A network instance handle
	 * @return The summary string
	 * @throws AiliaException
	 */
	public static native String Summary(long handle) throws AiliaException;

	/**
	 * Get the number of input data blobs.
	 *
	 * @param handle A network instance handle
	 * @return The number of input blobs
	 * @throws AiliaException
	 */
	public static native int GetInputBlobCount(long handle) throws AiliaException;

	/**
	 * Get the blob index of the input data.
	 *
	 * @param handle A network instance handle
	 * @param inputBlobIdx Index among the input blobs (between 0 and {@link #GetInputBlobCount(long)}-1)
	 * @return Index of the blob (between 0 and {@link #GetBlobCount(long)}-1)
	 * @throws AiliaException
	 */
	public static native int GetBlobIndexByInputIndex(long handle, int inputBlobIdx) throws AiliaException;

	/**
	 * Provides the specified blob with the input data.
	 *
	 * <p>This method is used to specify the input on networks with multiple inputs.</p>
	 * @param handle A network instance handle
	 * @param src　The inference data is stored as float data in the order of X, Y, Z, and W.
	 * @param srcSize The size of the inference data (in byte)
	 * @param blobIdx The index of the blob for input
	 * @throws AiliaException
	 */
	public static native void SetInputBlobData(long handle, float[] src, int srcSize, int blobIdx) throws AiliaException;

	/**
	 * Change the shape of the blob given by its index
	 *
	 * <p>This is useful to change the network input shape in a context where there are several input blobs.
	 * If dimension of shape has 5 or more, please use {@link #SetInputBlobShapeND(long, int[], int, int)}</p>
	 * @param handle A network instance handle
	 * @param shape New shape of the blob
	 * @param blobIdx Index referencing the blob to reshape
	 * @param version Shape version({@link AiliaShape#version})
	 * @see #SetInputShape(long, AiliaShape, int)
	 * @throws AiliaException
	 */
	public static native void SetInputBlobShape(long handle, AiliaShape shape, int blobIdx, int version) throws AiliaException;

	/**
	 * Change the shape of the blob given by its index
	 *
	 * <p>This is useful to change the network input shape in a context where there are several input blobs.</p>
	 * @param handle A network instance handle
	 * @param shape The array of the new shape. (It stores dim-1, dim-2, ... ,1, 0 order.)
	 * @param dim The size of shape
	 * @param blobIdx Index referencing the blob to reshape
	 * @see #SetInputShapeND(long, int[], int)
	 * @throws AiliaException
	 */
	public static native void SetInputBlobShapeND(long handle, int[] shape, int dim, int blobIdx) throws AiliaException;

	/**
	 * Makes inferences with the input data specified in advance.
	 *
	 * <p>This function is used when, for example, the input is provided with {@link #SetInputBlobData(long, float[], int, int)}.
	 * Get the inference result with  {@link #GetBlobData(long, float[], int, int)}</p>
	 * @param handle A network instance handle
	 * @throws AiliaException
	 */
	public static native void Update(long handle) throws AiliaException;

	/**
	 * Get the number of output data blobs.
	 *
	 * @param handle A network instance handle
	 * @return The number of output blobs
	 * @throws AiliaException
	 */
	public static native int GetOutputBlobCount(long handle) throws AiliaException;

	/**
	 * Get the blob index of the output data blob.
	 * @param handle A network instance handle
	 * @param outputBlobIdx Index among output blobs (between 0 and {@link #GetOutputBlobCount(long)}-1)
	 * @return Blob index (between 0 and {@link #GetBlobCount(long)}-1)
	 * @throws AiliaException
	 */
	public static native int GetBlobIndexByOutputIndex(long handle, int outputBlobIdx) throws AiliaException;

	/**
	 * Specifies a temporary cache directory.
	 *
	 * <p>This system uses the specified cache directory to generate and store machine code optimized for each inference backend.
	 * Call only once at the start of execution of ailia. It ignores any second and subsequent calls, and automatically returns success.
	 * There is no particular problem if it is called from multiple threads, as it provides exclusive control internally.
	 * Some functions, such as RenderScript in an Android environment, cannot be used until this API function is called.
	 * Specify the file path obtained by {@link android.content.Context#getCacheDir()} for cache_dir.
	 * You cannot specify the path to external storage due to Permission restrictions from RenderScript.</p>
	 * @param cache_path Temporary cache directory
	 * @throws AiliaException
	 */
	public static native void SetTemporaryCachePath(String cache_path) throws AiliaException;

	/**
	 * Gets the number of available computational environments (CPU, GPU).
	 *
	 * @return The number of computational environment information
	 * @throws AiliaException
	 */
	public static native int GetEnvironmentCount() throws AiliaException;

	/**
	 * Gets the list of computational environments.
	 *
	 * @param env_id The index of the computational environment information (0 to {@link #GetEnvironmentCount()}-1)
	 * @param version {@link AiliaEnvironment#version}
	 * @return The computational environment information (valid until the AILIANetwork instance is destroyed)
	 * @throws AiliaException
	 */
	public static native AiliaEnvironment GetEnvironment(int env_id, int version) throws AiliaException;

	/**
	 * Gets the selected computational environment.
	 *
	 * @param handle A network instance handle
	 * @param version {@link AiliaEnvironment#version}
	 * @return The computational environment information (valid until the AILIANetwork instance is destroyed)
	 * @throws AiliaException
	 */
	public static native AiliaEnvironment GetSelectedEnvironment(long handle, int version) throws AiliaException;

	/**
	 * Set the memory usage policy for inference
	 *
	 * <p> hange the memory usage policy.
	 * If a value other than {@link #MEMORY_NO_OPTIMIZATION} is specified,
	 * the intermediate buffer secured during inference will be released, so the memory usage during inference can be reduced.
	 * Must be specified immediately after ailiaCreate. It cannot be changed after calling ailiaOpen.
	 * If you specify to release the intermediate buffer, calling {@link #GetBlobData(long, float[], int, int)} for the corresponding blob will thrown {@link AiliaException}.</p>
	 * @param handle A network instance handle
	 * @param mode Memory mode (Multiple specifications possible with logical sum) {@link AiliaMemoryMode} (Default :{@link AiliaMemoryMode#OPTIMAIZE_DEFAULT})
	 * @throws AiliaException
	 */
	public static native void SetMemoryMode(long handle, int mode) throws AiliaException;

	/**
	 * Returns the details of errors.
	 *
	 * <p>The return value does not have to be released.
	 * The string is valid until the next ailia API function is called.</p>
	 * @param handle A network instance handle
	 * @return Error details
	 */
	public static native String GetErrorDetail(long handle);

	/**
	 * Get the version of the library.
	 *
	 * <p>The return value does not have to be released.</p>
	 * @return Version number
	 */
	public static native String GetVersion();
}
