package jp.axinc.ailia_kotlin

//import android.R
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.Options
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import axip.ailia.*
import org.json.JSONObject
import java.io.*


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_main)
        ailia_init(savedInstanceState);
    }

    @Throws(IOException::class)
    fun inputStreamToByteArray(`in`: InputStream): ByteArray? {
        val bout = ByteArrayOutputStream()
        BufferedOutputStream(bout).use { out ->
            val buf = ByteArray(128)
            var n = 0
            while (`in`.read(buf).also { n = it } > 0) {
                out.write(buf, 0, n)
            }
        }
        return bout.toByteArray()
    }

    @Throws(IOException::class)
    fun loadRawImage(bmp: Bitmap): ByteArray {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val bout = ByteArrayOutputStream()
        val out = DataOutputStream(bout)
        for (i in pixels) {
            out.writeByte(i shr 16 and 0xff) //r
            out.writeByte(i shr 8 and 0xff) //g
            out.writeByte(i shr 0 and 0xff) //b
            out.writeByte(i shr 24 and 0xff) //a
        }
        return bout.toByteArray()
    }

    @Throws(IOException::class)
    fun loadRawFile(resourceId: Int): ByteArray? {
        val resources = this.resources
        resources.openRawResource(resourceId).use { `in` -> return inputStreamToByteArray(`in`) }
    }

    fun ailia_pose_estimator(envId: Int): Boolean {
        return try {
            val ailia = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                loadRawFile(R.raw.lightweight_human_pose_proto),
                loadRawFile(R.raw.lightweight_human_pose_weight)
            )
            val poseEstimator =
                ailia.createPoseEstimator(AiliaPoseEstimatorModel.ALGORITHM_LW_HUMAN_POSE)
            val imageId: Int = R.raw.person
            val options = Options()
            options.inScaled = false
            val bmp = BitmapFactory.decodeResource(this.resources, imageId, options)
            val img = loadRawImage(bmp)
            val w = bmp.width
            val h = bmp.height
            poseEstimator.compute(img, w * 4, w, h, AiliaImageFormat.RGBA)
            val objCount = poseEstimator.objectCount
            Log.i("AILIA_Main", "objCount (human count) = $objCount")
            if (objCount != 0) {
                val pose = poseEstimator.getObjectPose(0)
                Log.i("AILIA_Main", "total score = " + pose.totalScore)
                Log.i(
                    "AILIA_Main",
                    "angle[3] = {" + pose.angle[0] + ", " + pose.angle[1] + ", " + pose.angle[2] + "}"
                )
                for (i in 0 until AiliaPoseEstimatorObjectPose.KEYPOINT_COUNT) {
                    val p = pose.points[i]
                    Log.i(
                        "AILIA_Main",
                        "keypoint[" + i + "] = {x: " + p.x + ", y: " + p.y + ", z_local: " + p.z_local + ", score: " + p.score + ", interpolated: " + p.interpolated + "}"
                    )
                }
            } else {
                Log.i("AILIA_Error", "No object detected.")
                return false
            }
            true
        } catch (e: Exception) {
            Log.i("AILIA_Error", e.javaClass.name + ": " + e.message)
            false
        }
    }

    protected fun ailia_init(savedInstanceState: Bundle?) {
        try {
            // Detect GPU Environment
            Ailia.SetTemporaryCachePath(cacheDir.absolutePath)
            val envList = AiliaModel.getEnvironments()
            var selectedEnv = envList[0]
            for (env in envList) {
                Log.i(
                    "AILIA_Main",
                    "Environment " + env.id + ": ( type: " + env.type + ", name: " + env.name + ")"
                )
                if (env.type == AiliaEnvironment.TYPE_GPU && env.props and AiliaEnvironment.PROPERTY_FP16 == 0) {
                    selectedEnv = env
                    break
                }
            }
            Log.i(
                "AILIA_Main",
                "Selected environment id: " + selectedEnv.id + " (" + selectedEnv.name + ")"
            )

            // Samples
            var success = true
            success = success and ailia_pose_estimator(selectedEnv.id)

            // ForTest
            if (success) {
                Log.i("AILIA_Main", "Success")
            }
        } catch (e: Exception) {
            Log.i("AILIA_Error", e.javaClass.name + ": " + e.message)
        }
    }

    //Important : load ailia library
    companion object {
        init {
            System.loadLibrary("ailia")
        }
    }
}