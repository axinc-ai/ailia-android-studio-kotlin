/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaPoseEstimatorAlgorithm {
    ACCULUS_POSE(0),
    ACCULUS_FACE(1),
    ACCULUS_UPPOSE(2),
    ACCULUS_UPPOSE_FPGA(3),
    ACCULUS_HAND(5),
    OPEN_POSE(10),
    LW_HUMAN_POSE(11),
    OPEN_POSE_SINGLE_SCALE(12);

    private int value;
    private AiliaPoseEstimatorAlgorithm(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
