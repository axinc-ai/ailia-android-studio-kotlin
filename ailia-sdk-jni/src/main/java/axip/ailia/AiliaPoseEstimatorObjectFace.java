/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public class AiliaPoseEstimatorObjectFace
{
    public static final int version = 1;

	public static final int KEYPOINT_COUNT = 68;

	/**
	 * Detected human face landmarks. The array index corresponding to a human face landmark number.
	 */
	public AiliaPoseEstimatorKeypoint[] points;
	/**
	 * The confidence of this object
	 */
    public float totalScore;

	AiliaPoseEstimatorObjectFace(AiliaPoseEstimatorKeypoint[] points, float totalScore)
	{
		this.points = new AiliaPoseEstimatorKeypoint[KEYPOINT_COUNT];

		this.totalScore = totalScore;
		for(int i = 0; i < KEYPOINT_COUNT; ++i)
		{
			this.points[i] = points[i];
		}
    }
}
