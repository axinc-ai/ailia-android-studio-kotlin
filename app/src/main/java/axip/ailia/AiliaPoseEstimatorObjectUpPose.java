/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public class AiliaPoseEstimatorObjectUpPose
{
    public static final int version = 1;

	public static final int KEYPOINT_NOSE = 0;
	public static final int KEYPOINT_EYE_LEFT = 1;
	public static final int KEYPOINT_EYE_RIGHT = 2;
	public static final int KEYPOINT_EAR_LEFT = 3;
	public static final int KEYPOINT_EAR_RIGHT = 4;
	public static final int KEYPOINT_SHOULDER_LEFT = 5;
	public static final int KEYPOINT_SHOULDER_RIGHT = 6;
	public static final int KEYPOINT_ELBOW_LEFT = 7;
	public static final int KEYPOINT_ELBOW_RIGHT = 8;
	public static final int KEYPOINT_WRIST_LEFT = 9;
	public static final int KEYPOINT_WRIST_RIGHT = 10;
	public static final int KEYPOINT_HIP_LEFT = 11;
	public static final int KEYPOINT_HIP_RIGHT = 12;
	public static final int KEYPOINT_SHOULDER_CENTER = 13;
	public static final int KEYPOINT_BODY_CENTER = 14;
	public static final int KEYPOINT_COUNT = 15;

	/**
	 * Detected body joint positions. The array index corresponding to a body joint number.
	 */
	public AiliaPoseEstimatorKeypoint[] points;
	/**
	 * The confidence of this object
	 */
    public float totalScore;
	/**
	 * The number of body joint positions properly detected in {@value points}
	 */
	public int numValidPoints;
	/**
	 * A unique ID for this object in the time direction. An integer value of 1 or more.
	 */
	public int id;
	/**
	 * Euler angles for this object: yaw, pitch, and roll (in radians). Currently, only yaw is supported. If the angles are not detected, they are set to {@link Float#MAX_VALUE}.
	 */
	public float[] angle;

	AiliaPoseEstimatorObjectUpPose(AiliaPoseEstimatorKeypoint[] points, float totalScore, int numValidPoints, int id, float[] angle)
	{
		this.points = new AiliaPoseEstimatorKeypoint[KEYPOINT_COUNT];
		this.angle = new float[3];

		this.totalScore = totalScore;
		this.numValidPoints = numValidPoints;
		this.id = id;
		for(int i = 0; i < KEYPOINT_COUNT; ++i)
		{
			this.points[i] = points[i];
		}
		for(int i = 0; i < 3; ++i)
		{
			this.angle[i] = angle[i];
		}
    }
}
