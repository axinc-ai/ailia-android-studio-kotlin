/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public class AiliaException extends Exception {
	/**
	 * Successful
	 * @deprecated Please use{@link AiliaStatus#SUCCESS}
	 */
	@Deprecated public static final int STATUS_SUCCESS              =   0;
	/**
	 * Incorrect argument
	 * @deprecated Please use{@link AiliaStatus#INVALID_ARGUMENT}
	 */
	@Deprecated public static final int STATUS_INVALID_ARGUMENT     =  -1;
	/**
	 * File access failed.
	 * @deprecated Please use{@link AiliaStatus#ERROR_FILE_API}
	 */
	@Deprecated public static final int STATUS_ERROR_FILE_API       =  -2;
	/**
	 * Incorrect struct version
	 * @deprecated Please use{@link AiliaStatus#INVALID_VERSION}
	 */
	@Deprecated public static final int STATUS_INVALID_VERSION      =  -3;
	/**
	 * A corrupt file was passed.
	 * @deprecated Please use{@link AiliaStatus#BROKEN}
	 */
	@Deprecated public static final int STATUS_BROKEN               =  -4;
	/**
	 * Insufficient memory
	 * @deprecated Please use{@link AiliaStatus#MEMORY_INSUFFICIENT}
	 */
	@Deprecated public static final int STATUS_MEMORY_INSUFFICIENT  =  -5;
	/**
	 * Thread creation failed.
	 * @deprecated Please use{@link AiliaStatus#THREAD_ERROR}
	 */
	@Deprecated public static final int STATUS_THREAD_ERROR         =  -6;
	/**
	 * The internal status of the decoder is incorrect.
	 * @deprecated Please use{@link AiliaStatus#INVALID_STATE}
	 */
	@Deprecated public static final int STATUS_INVALID_STATE        =  -7;
	/**
	 * Unsupported network
	 * @deprecated Please use{@link AiliaStatus#UNSUPPORT_NET}
	 */
	@Deprecated public static final int STATUS_UNSUPPORT_NET        =  -9;
	/**
	 * Incorrect layer weight, parameter, or input or output shape
	 * @deprecated Please use{@link AiliaStatus#INVALID_LAYER}
	 */
	@Deprecated public static final int STATUS_INVALID_LAYER        = -10;
	/**
	 * The content of the parameter file is invalid.
	 * @deprecated Please use{@link AiliaStatus#INVALID_PARAMINFO}
	 */
	@Deprecated public static final int STATUS_INVALID_PARAMINFO    = -11;
	/**
	 * The specified element was not found.
	 * @deprecated Please use{@link AiliaStatus#NOT_FOUND}
	 */
	@Deprecated public static final int STATUS_NOT_FOUND            = -12;
	/**
	 * A layer parameter not supported by the GPU was given.
	 * @deprecated Please use{@link AiliaStatus#GPU_UNSUPPORT_LAYER}
	 */
	@Deprecated public static final int STATUS_GPU_UNSUPPORT_LAYER  = -13;
	/**
	 * Error during processing on the GPU
	 * @deprecated Please use{@link AiliaStatus#GPU_ERROR}
	 */
	@Deprecated public static final int STATUS_GPU_ERROR            = -14;
	/**
	 * Unimplemented error
	 * @deprecated Please use{@link AiliaStatus#UNIMPLEMENTED}
	 */
	@Deprecated public static final int STATUS_UNIMPLEMENTED        = -15;
	/**
	 * Operation not allowed
	 * @deprecated Please use{@link AiliaStatus#PERMISSION_DENIED}
	 */
	@Deprecated public static final int STATUS_PERMISSION_DENIED    = -16;
	/**
	 * Model Expired
	 * @deprecated Please use{@link AiliaStatus#EXPIRED}
	 */
	@Deprecated public static final int STATUS_EXPIRED              = -17;
	/**
	 * The shape is not yet determined
	 * @deprecated Please use{@link AiliaStatus#UNSETTLED_SHAPE}
	 */
	@Deprecated public static final int STATUS_UNSETTLED_SHAPE      = -18;
	/**
	 * Deleted due to optimization
	 * @deprecated Please use{@link AiliaStatus#DATA_HIDDEN}
	 */
	@Deprecated public static final int STATUS_DATA_HIDDEN          = -19;
	/**
	 * No valid license found
	 * @deprecated Please use{@link AiliaStatus#LICENSE_NOT_FOUND}
	 */
	@Deprecated public static final int STATUS_LICENSE_NOT_FOUND    = -20;
	/**
	 * License is broken
	 * @deprecated Please use{@link AiliaStatus#LICENSE_BROKEN}
	 */
	@Deprecated public static final int STATUS_LICENSE_BROKEN       = -21;
	/**
	 * License expired
	 * @deprecated Please use{@link AiliaStatus#LICENSE_EXPIRED}
	 */
	@Deprecated public static final int STATUS_LICENSE_EXPIRED      = -22;
	/**
	 * Dimension of shape is 5 or more.
	 * @deprecated Please use{@link AiliaStatus#NDIMENSION_SHAPE}
	 */
	@Deprecated public static final int STATUS_NDIMENSION_SHAPE     = -23;
	/**
	 * Unknown error
	 * @deprecated Please use{@link AiliaStatus#OTHER_ERROR}
	 */
	@Deprecated public static final int STATUS_OTHER_ERROR          = -128;
	/**
	 * @deprecated
	 */
	@Deprecated static final int STATUS_DATA_REMOVED         = STATUS_DATA_HIDDEN;

	private AiliaStatus status;
	public AiliaException(String message) {
		super(message);
		this.status = AiliaStatus.OTHER_ERROR;
	}
	public AiliaException(String message, int code) {
		super(message);
		this.status = AiliaStatus.fromInt(code);
	}
	public AiliaException(String message, AiliaStatus status) {
		super(message);
		this.status = status;
	}
	/**
	 * Get the error code.
	 * @return STATUS_XXX
	 * @deprecated please use {@link #getStatus()}
	 */
	@Deprecated public int getErrorCode() {
		return this.status.getValue();
	}
	public AiliaStatus getStatus() {
		return this.status;
	}

}
