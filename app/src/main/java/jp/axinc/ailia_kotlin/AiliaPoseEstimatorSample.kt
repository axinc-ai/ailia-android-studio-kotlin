package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import android.widget.ImageView
import axip.ailia.Ailia
import axip.ailia.AiliaEnvironment
import axip.ailia.AiliaImageFormat
import axip.ailia.AiliaModel
import axip.ailia.AiliaPoseEstimatorAlgorithm
import axip.ailia.AiliaPoseEstimatorModel
import axip.ailia.AiliaPoseEstimatorObjectPose
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.nio.ByteBuffer

class AiliaPoseEstimatorSample {
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

    fun ailia_environment(cache_dir: String) : AiliaEnvironment {
        // Detect GPU Environment
        Ailia.SetTemporaryCachePath(cache_dir)
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
        return selectedEnv;
    }
    fun ailia_pose_estimator(envId: Int, proto: ByteArray?, model: ByteArray?, bmp: Bitmap, image: ImageView): Boolean {
        return try {
            //create ailia pose estimator
            val ailia = AiliaModel(
                envId,
                Ailia.MULTITHREAD_AUTO,
                proto,
                model
            )
            val poseEstimator =
                AiliaPoseEstimatorModel(ailia.handle, AiliaPoseEstimatorAlgorithm.LW_HUMAN_POSE)

            //get test image
            val img = loadRawImage(bmp)
            val w = bmp.width
            val h = bmp.height

            //display test image
            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(img))
            image.setImageBitmap(bitmap)

            //create canvas
            val canvas = Canvas(bitmap)
            val paint = Paint()
            canvas.drawARGB(0, 0, 0, 0)
            paint.color = Color.parseColor("#FFFFFF")

            //run
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

                    //display result
                    var r:Float = 8.0f
                    canvas.drawCircle(
                        (bitmap.getWidth() * p.x).toFloat(),
                        (bitmap.getHeight() * p.y).toFloat(),
                        r,
                        paint
                    )
                }
            } else {
                Log.i("AILIA_Error", "No object detected.")
                return false
            }

            poseEstimator.close()
            ailia.close()
            true
        } catch (e: Exception) {
            Log.i("AILIA_Error", e.javaClass.name + ": " + e.message)
            false
        }
    }
}