/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaMemoryMode {
    /**
	 * Do not release the intermediate buffer
	 */
	NO_OPTIMIZATION(0),
	/**
	 * Releases the intermediate buffer that is a constant such as weight
	 */
	REDUCE_CONSTANT(1),
	/**
	 * Disable the input specified initializer and release the intermediate buffer that becomes a constant such as weight.
	 */
	REDUCE_CONSTANT_WITH_INPUT_INITIALIZER(2),
	/**
	 * Release intermediate buffer during inference
	 */
	REDUCE_INTERSTAGE(4),
	/**
	 * Infer by sharing the intermediate buffer. When used with {@link #REDUCE_INTERSTAGE}, the sharable intermediate buffer is not opened.
	 */
	REUSE_INTERSTAGE(8),
	OPTIMAIZE_DEFAULT(1);

    private int value;
    private AiliaMemoryMode(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

}
