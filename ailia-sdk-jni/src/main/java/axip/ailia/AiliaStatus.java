/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaStatus {
	/**
     * Successful
     */
    SUCCESS(0),
	/**
     * Incorrect argument
     */
    INVALID_ARGUMENT( -1),
	/**
     * File access failed.
     */
    ERROR_FILE_API( -2),
	/**
     * Incorrect struct version
     */
    INVALID_VERSION( -3),
	/**
     * A corrupt file was passed.
     */
    BROKEN( -4),
	/**
     * Insufficient memory
     */
    MEMORY_INSUFFICIENT( -5),
	/**
     * Thread creation failed.
     */
    THREAD_ERROR( -6),
	/**
     * The internal status of the decoder is incorrect.
     */
    INVALID_STATE( -7),
	/**
     * Unsupported network
     */
    UNSUPPORT_NET( -9),
	/**
     * Incorrect layer weight, parameter, or input or output shape
     */
    INVALID_LAYER(-10),
	/**
     * The content of the parameter file is invalid.
     */
    INVALID_PARAMINFO(-11),
	/**
     * The specified element was not found.
     */
    NOT_FOUND(-12),
	/**
     * A layer parameter not supported by the GPU was given.
     */
    GPU_UNSUPPORT_LAYER(-13),
	/**
     * Error during processing on the GPU
     */
    GPU_ERROR(-14),
	/**
     * Unimplemented error
     */
    UNIMPLEMENTED(-15),
	/**
     * Operation not allowed
     */
    PERMISSION_DENIED(-16),
	/**
     * Model Expired
     */
    EXPIRED(-17),
	/**
     * The shape is not yet determined
     */
    UNSETTLED_SHAPE(-18),
	/**
     * Deleted due to optimization
     */
    DATA_HIDDEN(-19),
	/**
     * No valid license found
     */
    LICENSE_NOT_FOUND(-20),
	/**
     * License is broken
     */
    LICENSE_BROKEN(-21),
	/**
     * License expired
     */
    LICENSE_EXPIRED(-22),
	/**
     * Dimension of shape is 5 or more.
     */
    NDIMENSION_SHAPE(-23),
	/**
     * Unknown error
     */
    OTHER_ERROR(-128);

    private int value;
    private AiliaStatus(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static AiliaStatus fromInt(int value) {
        for (AiliaStatus status : AiliaStatus.values()) {
            if (status.value == value) return status;
        }
        return OTHER_ERROR;
    }
}
