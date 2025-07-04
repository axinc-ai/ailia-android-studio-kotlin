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

    protected fun ailia_test(savedInstanceState: Bundle?) {
        try {
            // Load test image
            val options = Options()
            options.inScaled = false
            val person_bmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)
            val clock_bmp = BitmapFactory.decodeResource(this.resources, R.raw.clock, options)

            // Create result image
            val image = findViewById<View>(R.id.imageView) as ImageView

            //get test image
            val img = loadRawImage(person_bmp)
            val w = person_bmp.width
            val h = person_bmp.height

            //display test image
            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(img))
            image.setImageBitmap(bitmap)

            //create canvas
            val canvas = Canvas(bitmap)
            val paint = Paint()
            canvas.drawARGB(0, 0, 0, 0)
            paint.color = Color.parseColor("#FFFFFF")

            val paint2 = Paint().apply {
                style = Paint.Style.STROKE // 塗りつぶしを無効にし、枠線のみを描画
                color = Color.RED // 境界線の色を設定
                strokeWidth = 5f // 境界線の太さを設定
            }

            val text = Paint().apply {
                color = Color.BLACK // テキストの色
                textSize = 50f // テキストサイズ
                isAntiAlias = true // アンチエイリアシングを有効にすることで、テキストをなめらかに表示
            }

            // Pose Estimation
            var proto: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_proto)
            var model: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_weight)
            var pose_estimator_sample = AiliaPoseEstimatorSample();
            var selectedEnv = pose_estimator_sample.ailia_environment(cacheDir.absolutePath)
            var success = pose_estimator_sample.ailia_pose_estimator(selectedEnv.id, proto, model, img, canvas, paint, w, h)

            // Tokenizer
            var tokenizer_sample = AiliaTokenizerSample()
            tokenizer_sample.ailia_tokenize()

            // TFLite Classification
            var mobilenet_model: ByteArray? = loadRawFile(R.raw.mobilenetv2)
            var tflite_classification_sample = AiliaTFLiteClassificationSample()
            tflite_classification_sample.classification(mobilenet_model, clock_bmp)

            // TFLite Object Detection
            var yolox_model: ByteArray? = loadRawFile(R.raw.yolox_tiny)
            var tflite_detection_sample = AiliaTFLiteObjectDetectionSample()
            tflite_detection_sample.detection(yolox_model, person_bmp, canvas, paint2, text, w, h)
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