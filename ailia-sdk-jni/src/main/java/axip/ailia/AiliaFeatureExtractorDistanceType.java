/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaFeatureExtractorDistanceType {

    L2NORM(0);

    private int value;
    private AiliaFeatureExtractorDistanceType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
