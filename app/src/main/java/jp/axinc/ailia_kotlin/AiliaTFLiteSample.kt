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

class AiliaTFLiteSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    private fun loadFile(filename: String): ByteArray? {
        return try {
            File(filename).readBytes()
        } catch (e: Exception) {
            Log.e(TAG, "Could not read file: $filename")
            null
        }
    }

    private fun loadImage(inputTensorType: Int, inputBuffer: ByteArray, inputShape: IntArray, bitmap : Bitmap): ByteArray {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputShape[2], inputShape[1], true)
        val channels = inputShape[3]
        val buffer = ByteArray(inputShape[1] * inputShape[2] * channels)

        val pixels = IntArray(inputShape[1] * inputShape[2])
        scaledBitmap.getPixels(pixels, 0, inputShape[2], 0, 0, inputShape[2], inputShape[1])

        for (y in 0 until inputShape[1]) {
            for (x in 0 until inputShape[2]) {
                val pixel = pixels[y * inputShape[2] + x]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                if (inputTensorType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_FLOAT32) {
                    val floatBuffer = inputBuffer as FloatArray
                    floatBuffer[(y * inputShape[2] + x) * channels + 0] = r.toFloat()
                    floatBuffer[(y * inputShape[2] + x) * channels + 1] = g.toFloat()
                    floatBuffer[(y * inputShape[2] + x) * channels + 2] = b.toFloat()
                } else {
                    buffer[(y * inputShape[2] + x) * channels + 0] = r.toByte()
                    buffer[(y * inputShape[2] + x) * channels + 1] = g.toByte()
                    buffer[(y * inputShape[2] + x) * channels + 2] = b.toByte()
                }
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

    fun yolox_main(modelData: ByteArray?, bitmap: Bitmap, canvas: Canvas, paint: Paint, w: Int, h: Int, env: Int = AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE): Boolean {
        if (modelData == null){
            Log.e(TAG, "Failed to open model data")
            return false;
        }

        val tflite = AiliaTFLite()
        if (!tflite.open(modelData, env)) {
            Log.e(TAG, "Failed to open TFLite model")
            return false
        }

        if (!tflite.allocateTensors()) {
            Log.e(TAG, "Failed to allocate tensors")
            tflite.close()
            return false
        }

        val inputTensorIndex = tflite.getInputTensorIndex(0)
        val inputShape = tflite.getInputTensorShape(0) ?: run {
            Log.e(TAG, "Failed to get input tensor shape")
            tflite.close()
            return false
        }

        val inputTensorType = tflite.getInputTensorType(0)
        val inputBuffer = loadImage(inputTensorType, ByteArray(inputShape[1] * inputShape[2] * inputShape[3]), inputShape, bitmap)

        if (!tflite.setTensorData(inputTensorIndex, inputBuffer)) {
            Log.e(TAG, "Failed to set input tensor data")
            tflite.close()
            return false
        }

        // Measure time
        val startTime = System.nanoTime()
        if (!tflite.predict()) {
            Log.e(TAG, "Predict failed")
            tflite.close()
            return false
        }
        val endTime = System.nanoTime()
        Log.i(TAG, "Inference time: ${(endTime - startTime) / 1000000} ms")

        val outputTensorIndex = tflite.getOutputTensorIndex(0)
        val outputShape = tflite.getOutputTensorShape(0) ?: run {
            Log.e(TAG, "Failed to get output tensor shape")
            tflite.close()
            return false
        }

        val outputType = tflite.getOutputTensorType(0)
        val outputData = tflite.getTensorData(outputTensorIndex) ?: run {
            Log.e(TAG, "Failed to get output tensor data")
            tflite.close()
            return false
        }

        val quantCount = tflite.getTensorQuantizationCount(outputTensorIndex)
        if (quantCount != 1) {
            Log.e(TAG, "Unexpected quantization count: $quantCount")
            tflite.close()
            return false
        }

        val quantScale = tflite.getTensorQuantizationScale(outputTensorIndex)?.get(0) ?: 1.0f
        val quantZeroPoint = tflite.getTensorQuantizationZeroPoint(outputTensorIndex)?.get(0) ?: 0L

        if (outputType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_FLOAT32) {
            postProcessYoloxFp32(inputShape, outputShape, outputData as FloatArray, outputType)
        } else if (outputType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_UINT8 ||
            outputType == AiliaTFLite.AILIA_TFLITE_TENSOR_TYPE_INT8) {
            postProcessYolox(inputShape, outputShape, outputData, outputType, quantScale, quantZeroPoint, canvas, paint, w, h)
        }

        tflite.close()
        return true
    }

    // COCO categories for object detection
    private val COCO_CATEGORY = arrayOf(
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    private fun postProcessYolox(inputShape: IntArray, outputShape: IntArray, outputBuffer: ByteArray, outputTensorType: Int,
                                 quantScale: Float, quantZeroPoint: Long, canvas: Canvas, paint: Paint, w: Int, h: Int) {
        val ih = inputShape[1]
        val iw = inputShape[2]
        val oh = arrayOf(ih / 8, ih / 16, ih / 32)
        val ow = arrayOf(iw / 8, iw / 16, iw / 32)
        val numCells = oh[0] * ow[0] + oh[1] * ow[1] + oh[2] * ow[2]
        val numElements = 5 + COCO_CATEGORY.size
        if (numCells != outputShape[1] || numElements != outputShape[2]) {
            Log.e(TAG, "Error! YOLOX output_shape[1,2] mismatch")
            return
        }

        var bufIndex = 0
        for (s in 0..2) {
            val stride = 2f.pow(3 + s)
            for (y in 0 until oh[s]) {
                for (x in 0 until ow[s]) {
                    var maxScore = 0.toByte()
                    var maxClass = 0

                    for (cls in 0 until COCO_CATEGORY.size) {
                        val score = outputBuffer[bufIndex + 5 + cls]
                        if (score > maxScore) {
                            maxScore = score
                            maxClass = cls
                        }
                    }

                    var score = dequantUint8(maxScore, quantScale, quantZeroPoint, outputTensorType)
                    val c = dequantUint8(outputBuffer[bufIndex + 4], quantScale, quantZeroPoint, outputTensorType)
                    score *= c

                    val detThreshold = 0.25f
                    if (score >= detThreshold) {
                        val cx = dequantUint8(outputBuffer[bufIndex + 0], quantScale, quantZeroPoint, outputTensorType)
                        val cy = dequantUint8(outputBuffer[bufIndex + 1], quantScale, quantZeroPoint, outputTensorType)
                        val w = dequantUint8(outputBuffer[bufIndex + 2], quantScale, quantZeroPoint, outputTensorType)
                        val h = dequantUint8(outputBuffer[bufIndex + 3], quantScale, quantZeroPoint, outputTensorType)

                        val bbCx = (cx + x) * stride
                        val bbCy = (cy + y) * stride
                        val bbW = exp(w) * stride + 1f
                        val bbH = exp(h) * stride + 1f

                        Log.i(TAG, "s=$s, x=$x, y=$y, class=[$maxClass, ${COCO_CATEGORY[maxClass]}], score=$score, " +
                                "cx=$cx, cy=$cy, w=$w, h=$h, c=$c, bb=[$bbCx,$bbCy,$bbW,$bbH]")

                        var r:Float = 8.0f
                        canvas.drawRect(
                            bbCx,
                            bbCy,
                            bbCx + bbW,
                            bbCy + bbH,
                            paint
                        )
                    }

                    bufIndex += numElements
                }
            }
        }
    }

    private fun postProcessYoloxFp32(inputShape: IntArray, outputShape: IntArray, outputBuffer: FloatArray, outputTensorType: Int) {
        val ih = inputShape[1]
        val iw = inputShape[2]
        val oh = arrayOf(ih / 8, ih / 16, ih / 32)
        val ow = arrayOf(iw / 8, iw / 16, iw / 32)
        val numCells = oh[0] * ow[0] + oh[1] * ow[1] + oh[2] * ow[2]
        val numElements = 5 + COCO_CATEGORY.size
        if (numCells != outputShape[1] || numElements != outputShape[2]) {
            Log.e(TAG, "Error! YOLOX output_shape[1,2] mismatch")
            return
        }

        var bufIndex = 0
        for (s in 0..2) {
            val stride = 2f.pow(3 + s)
            for (y in 0 until oh[s]) {
                for (x in 0 until ow[s]) {
                    var maxScore = 0f
                    var maxClass = 0

                    for (cls in 0 until COCO_CATEGORY.size) {
                        val score = outputBuffer[bufIndex + 5 + cls]
                        if (score > maxScore) {
                            maxScore = score
                            maxClass = cls
                        }
                    }

                    var score = maxScore
                    val c = outputBuffer[bufIndex + 4]
                    score *= c

                    val detThreshold = 0.5f
                    if (score >= detThreshold) {
                        val cx = outputBuffer[bufIndex + 0]
                        val cy = outputBuffer[bufIndex + 1]
                        val w = outputBuffer[bufIndex + 2]
                        val h = outputBuffer[bufIndex + 3]

                        val bbCx = (cx + x) * stride
                        val bbCy = (cy + y) * stride
                        val bbW = exp(w) * stride + 1f
                        val bbH = exp(h) * stride + 1f

                        Log.i(TAG, "s=$s, x=$x, y=$y, class=[$maxClass, ${COCO_CATEGORY[maxClass]}], score=$score, " +
                                "cx=$cx, cy=$cy, w=$w, h=$h, c=$c, bb=[$bbCx,$bbCy,$bbW,$bbH]")
                    }

                    bufIndex += numElements
                }
            }
        }
    }
}
