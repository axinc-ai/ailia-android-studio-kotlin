/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public class AiliaInvalidArgumentException extends AiliaException {
    public AiliaInvalidArgumentException(String message) {
        super(message, AiliaStatus.INVALID_ARGUMENT);
    }
}
