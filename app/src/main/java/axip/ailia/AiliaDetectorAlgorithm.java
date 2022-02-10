/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public enum AiliaDetectorAlgorithm {

    YOLOV1(0),
    YOLOV2(1),
    YOLOV3(2),
    YOLOV4(3),
	SSD(8);

    private int value;
    private AiliaDetectorAlgorithm(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
