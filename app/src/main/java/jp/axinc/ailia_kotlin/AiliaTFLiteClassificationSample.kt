package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log
import android.widget.ImageView
import axip.ailia_tflite.AiliaTFLite
import java.io.File
import kotlin.math.exp
import kotlin.math.pow

class AiliaTFLiteClassificationSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    private fun loadImage(inputTensorType: Int, inputBuffer: ByteArray, inputShape: IntArray, bitmap : Bitmap, quantScale: Float, quantZeroPoint: Long): ByteArray {
        Log.i(TAG, ""+inputShape[0].toString()+" "+inputShape[1].toString()+ " "+inputShape[2].toString()+ " "+inputShape[3].toString())
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputShape[2], inputShape[1], true)
        val channels = inputShape[3]
        val buffer = ByteArray(inputShape[1] * inputShape[2] * channels)

        val pixels = IntArray(inputShape[1] * inputShape[2])
        scaledBitmap.getPixels(pixels, 0, inputShape[2], 0, 0, inputShape[2], inputShape[1])

        for (y in 0 until inputShape[1]) {
            for (x in 0 until inputShape[2]) {
                val pixel = pixels[y * inputShape[2] + x]
                val r : Int = (pixel shr 16) and 0xFF
                val g : Int = (pixel shr 8) and 0xFF
                val b : Int = pixel and 0xFF
                val r2 : Int = maxOf(0.toInt(), minOf(((r / 127.5f - 1.0f) / quantScale + quantZeroPoint).toInt(), 255.toInt()))
                val g2 : Int = maxOf(0.toInt(), minOf(((g / 127.5f - 1.0f) / quantScale + quantZeroPoint).toInt(), 255.toInt()))
                val b2 : Int = maxOf(0.toInt(), minOf(((b / 127.5f - 1.0f) / quantScale + quantZeroPoint).toInt(), 255.toInt()))
                buffer[(y * inputShape[2] + x) * channels + 0] = r2.toByte()
                buffer[(y * inputShape[2] + x) * channels + 1] = g2.toByte()
                buffer[(y * inputShape[2] + x) * channels + 2] = b2.toByte()
            }
        }
        return buffer
    }

    private fun dequantUint8(
        value: Byte, quantScale: Float, quantZeroPoint: Long, tensorType: Int
    ): Float {
        return if (tensorType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_INT8) {
            ((value.toInt() - quantZeroPoint).toFloat() * quantScale)
        } else {
            (((value.toInt() and 0xFF) - quantZeroPoint).toFloat() * quantScale)
        }
    }

    private var tflite: AiliaTFLite? = null
    private var isInitialized = false
    private var inputShape: IntArray? = null
    private var inputTensorIndex: Int = -1
    private var outputTensorIndex: Int = -1
    private var outputShape: IntArray? = null
    private var outputType: Int = -1
    private var quantScale: Float = 1.0f
    private var quantZeroPoint: Long = 0L
    private var lastClassificationResult: String = ""

    fun initializeClassification(modelData: ByteArray?, env: Int = AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE): Boolean {
        if (modelData == null || modelData.isEmpty()) {
            Log.e(TAG, "Model data is null or empty")
            return false
        }

        if (isInitialized) {
            releaseClassification()
        }

        return try {
            tflite = AiliaTFLite()
            if (!tflite!!.open(modelData, env)) {
                Log.e(TAG, "Failed to open TFLite model")
                releaseClassification()
                return false
            }

            if (!tflite!!.allocateTensors()) {
                Log.e(TAG, "Failed to allocate tensors")
                releaseClassification()
                return false
            }

            inputTensorIndex = tflite!!.getInputTensorIndex(0)
            if (inputTensorIndex < 0) {
                Log.e(TAG, "Invalid input tensor index: $inputTensorIndex")
                releaseClassification()
                return false
            }

            inputShape = tflite!!.getInputTensorShape(0) ?: run {
                Log.e(TAG, "Failed to get input tensor shape")
                releaseClassification()
                return false
            }

            outputTensorIndex = tflite!!.getOutputTensorIndex(0)
            if (outputTensorIndex < 0) {
                Log.e(TAG, "Invalid output tensor index: $outputTensorIndex")
                releaseClassification()
                return false
            }

            outputShape = tflite!!.getOutputTensorShape(0) ?: run {
                Log.e(TAG, "Failed to get output tensor shape")
                releaseClassification()
                return false
            }

            outputType = tflite!!.getOutputTensorType(0)

            val quantCount = tflite!!.getTensorQuantizationCount(outputTensorIndex)
            if (quantCount != 1) {
                Log.e(TAG, "Unexpected quantization count: $quantCount")
                releaseClassification()
                return false
            }

            quantScale = tflite!!.getTensorQuantizationScale(outputTensorIndex)?.get(0) ?: 1.0f
            quantZeroPoint = tflite!!.getTensorQuantizationZeroPoint(outputTensorIndex)?.get(0) ?: 0L

            isInitialized = true
            Log.i(TAG, "Classification initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize classification: ${e.javaClass.name}: ${e.message}")
            releaseClassification()
            false
        }
    }

    fun processClassification(bitmap: Bitmap): Long {
        if (!isInitialized || tflite == null || inputShape == null || outputShape == null) {
            Log.e(TAG, "Classification not initialized properly")
            return -1
        }

        return try {
            val inputTensorType = tflite!!.getInputTensorType(0)
            val inputQuantScale = tflite!!.getTensorQuantizationScale(inputTensorIndex)?.get(0) ?: 1.0f
            val inputQuantZeroPoint = tflite!!.getTensorQuantizationZeroPoint(inputTensorIndex)?.get(0) ?: 0L
            
            val inputBuffer = loadImage(inputTensorType, ByteArray(inputShape!![1] * inputShape!![2] * inputShape!![3]), inputShape!!, bitmap, inputQuantScale, inputQuantZeroPoint)

            if (!tflite!!.setTensorData(inputTensorIndex, inputBuffer)) {
                Log.e(TAG, "Failed to set input tensor data")
                return -1
            }

            val startTime = System.nanoTime()
            if (!tflite!!.predict()) {
                Log.e(TAG, "Predict failed")
                return -1
            }
            val endTime = System.nanoTime()

            val outputData = tflite!!.getTensorData(outputTensorIndex) ?: run {
                Log.e(TAG, "Failed to get output tensor data")
                return -1
            }

            lastClassificationResult = postProcess(inputShape!!, outputShape!!, outputData, outputType, quantScale, quantZeroPoint)

            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process classification: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun releaseClassification() {
        try {
            tflite?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing classification: ${e.javaClass.name}: ${e.message}")
        } finally {
            tflite = null
            isInitialized = false
            inputShape = null
            inputTensorIndex = -1
            outputTensorIndex = -1
            outputShape = null
            outputType = -1
            quantScale = 1.0f
            quantZeroPoint = 0L
            Log.i(TAG, "Classification released")
        }
    }

    private fun postProcess(
        inputShape: IntArray,
        outputShape: IntArray,
        outputBuffer: ByteArray,
        outputTensorType: Int,
        quantScale: Float,
        quantZeroPoint: Long,
    ): String {
        var max_c = 0.0f;
        var max_i = 0;
        for (i in 0 until 1000) {
            val c = dequantUint8(outputBuffer[i], quantScale, quantZeroPoint, outputTensorType)
            if (max_c < c) {
                max_c = c;
                max_i = i;
            }
        }
        val result = "${CocoAndImageNetLabels.IMAGENET_CATEGORY[max_i]} (${String.format("%.2f", max_c)})"
        Log.i(TAG, "class " + max_i.toString() + " " + CocoAndImageNetLabels.IMAGENET_CATEGORY[max_i] + " confidence " + max_c.toString())
        return result
    }
    
    fun getLastClassificationResult(): String {
        return lastClassificationResult
    }

}
