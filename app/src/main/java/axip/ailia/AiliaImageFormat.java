/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaImageFormat {
	/**
	 * RGBA order
	 */
    RGBA(0x00),
	/**
	 * BGRA order
	 */
    BGRA(0x01),
	/**
	 * RGBA order (Bottom to top)
	 */
    RGBA_B2T(0x10),
	/**
	 * BGRA order (Bottom to top)
	 */
    BGRA_B2T(0x11);

    private int value;
    private AiliaImageFormat(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

}
