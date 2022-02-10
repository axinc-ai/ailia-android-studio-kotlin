/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public class AiliaClassifierClass {
    public static final int version = 1;

	/**
	 * Classification category number
	 */
    public int category;
	/**
	 * Estimated probability (0 to 1)
	 */
    public float prob;

    AiliaClassifierClass(int category, float prob) {
        this.category = category;
        this.prob = prob;
    }
}
