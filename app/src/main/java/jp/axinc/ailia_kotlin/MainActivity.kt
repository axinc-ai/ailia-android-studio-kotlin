package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.Options
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import axip.ailia.*
import axip.ailia_tflite.*
import org.json.JSONObject
import java.io.*
//import android.R
import android.R.attr

import android.widget.ImageView

import java.nio.ByteBuffer

import android.R.attr.height

import android.R.attr.width
import android.view.View
import android.graphics.PorterDuff

import android.graphics.PorterDuffXfermode

import android.graphics.Color

import android.graphics.Rect

import android.graphics.Paint

import android.graphics.Canvas





class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        ailia_test(savedInstanceState);
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
    fun loadRawFile(resourceId: Int): ByteArray? {
        val resources = this.resources
        resources.openRawResource(resourceId).use { `in` -> return inputStreamToByteArray(`in`) }
    }



    protected fun ailia_test(savedInstanceState: Bundle?) {
        try {
            // Load test image
            val imageId: Int = R.raw.person
            val options = Options()
            options.inScaled = false
            val bmp = BitmapFactory.decodeResource(this.resources, imageId, options)

            // Create result image
            val image = findViewById<View>(R.id.imageView) as ImageView

            // Pose Estimation
            var proto: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_proto)
            var model: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_weight)
            var pose_estimator_sample = AiliaPoseEstimatorSample();
            var selectedEnv = pose_estimator_sample.ailia_environment(cacheDir.absolutePath)
            var success = pose_estimator_sample.ailia_pose_estimator(selectedEnv.id, proto, model, bmp, image)

            // Tokenizer
            var tokenizer_sample = AiliaTokenizerSample()
            tokenizer_sample.ailia_tokenize()

            // TFLite Object Detection
            var tflite_model: ByteArray? = loadRawFile(R.raw.yolox_tiny)
            var tflite_sample = AiliaTFLiteSample()
            tflite_sample.yolox_main(tflite_model, bmp)
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