/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public class AiliaPoseEstimatorKeypoint
{
	/**
	 * Input image X coordinate (0.0, 1.0)
	 */
	public float x;
	/**
	 * Input image Y coordinate (0.0, 1.0)
	 */
	public float y;
	/**
	 * Valid only for human pose estimation. The local Z coordinate is estimated when the center of the body is defined as coordinate 0. The unit (scale) is the same as that for X.
	 */
	public float z_local;
	/**
	 * The confidence of this point. If the value is 0.0F, then this point is not available as it has not been detected yet.
	 */
	public float score;
	/**
	 * The default is 0. If this point has not been detected and can be interpolated by other points, the x and y values are then interpolated and the value of interpolated is set to 1.
	 */
	public int interpolated;

	AiliaPoseEstimatorKeypoint(float x, float y, float z_local, float score, int interpolated)
	{
		this.x = x;
		this.y = y;
		this.z_local = z_local;
		this.score = score;
		this.interpolated = interpolated;
    }
}
