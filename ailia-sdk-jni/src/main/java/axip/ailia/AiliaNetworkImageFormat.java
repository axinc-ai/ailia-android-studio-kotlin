/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaNetworkImageFormat {
	/**
	 * BGR order
	*/
    BGR(0),
	/**
	 * RGB order
	*/
    RGB(1),
	/**
	 * Gray Scale (1ch)
	*/
    GRAY(2),
	/**
	 * Equalized Gray Scale (1ch)
	*/
	GRAY_EQUALIZE(3);

    private int value;
    private AiliaNetworkImageFormat(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
