/* Copyright 2018-2021 AXELL CORPORATION */
/* Updated July 28, 2021*/

package axip.ailia;

public class AiliaEnvironment {
    public static final int version = 2;

    public static final int TYPE_CPU    = 0;
    public static final int TYPE_BLAS   = 1;
    public static final int TYPE_GPU    = 2;
    public static final int TYPE_REMOTE = 3;

    public static final int BACKEND_NONE = 0;
    public static final int BACKEND_AMP  = 1;
    public static final int BACKEND_CUDA = 2;
    public static final int BACKEND_MPS  = 3;
    public static final int BACKEND_RENDERSCRIPT = 4;
    public static final int BACKEND_VULKAN = 6;

    public static final int PROPERTY_NORMAL	= 0;
    public static final int PROPERTY_LOWPOWER = 1;
    public static final int PROPERTY_FP16 = 2;

	/**
	 * The ID to identify the inference backend (passed to {@link Ailia#Create(int, int)} as an argument)
	 */
    public int id;
	/**
	 * The type of the inference backend (TYPE_CPU, BLAS, or GPU)
	 */
    public int type;
	/**
	 * The device name.
	 */
    public String name;
	/**
	 * Computational (hardware) backend enabled by this environment (BACKEND_XXX)
	 */
    public int backend;
	/**
	 * Additional property (low-power etc) of the environment (PROPERTY_XXX)
	 */
    public int props;

    public AiliaEnvironment(int id, int type, String name, int backend, int props) {
        this.id = id;
        this.type = type;
        this.name = name;
        this.backend = backend;
        this.props = props;
    }
}
