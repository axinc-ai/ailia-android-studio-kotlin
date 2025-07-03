/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaNetworkImageRange {
	/**
	 * 0 to 255
	*/
    UNSIGNED_INT8(0),
	/**
	 * -128 to 127
	*/
    SIGNED_INT8  (1),
	/**
	 * 0.0 to 1.0
	*/
    UNSIGNED_FP32(2),
	/**
	 * -1.0 to 1.0
	*/
    SIGNED_FP32  (3),
	/**
	 * ImageNet mean&std normalize
	*/
    IMAGENET     (4);

    private int value;
    private AiliaNetworkImageRange(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
