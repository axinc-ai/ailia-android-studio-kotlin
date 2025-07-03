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

class AiliaTFLiteObjectDetectionSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    private fun loadImage(inputTensorType: Int, inputBuffer: ByteArray, inputShape: IntArray, bitmap : Bitmap): ByteArray {
        Log.i(TAG, ""+inputShape[0].toString()+" "+inputShape[1].toString()+ " "+inputShape[2].toString()+ " "+inputShape[3].toString())
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
                buffer[(y * inputShape[2] + x) * channels + 0] = b.toByte()
                buffer[(y * inputShape[2] + x) * channels + 1] = g.toByte()
                buffer[(y * inputShape[2] + x) * channels + 2] = r.toByte()
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

    fun detection(modelData: ByteArray?, bitmap: Bitmap, canvas: Canvas, paint: Paint, w: Int, h: Int, env: Int = AiliaTFLite.AILIA_TFLITE_ENV_REFERENCE): Boolean {
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

        postProcessYolox(inputShape, outputShape, outputData, outputType, quantScale, quantZeroPoint, canvas, paint, w, h)

        tflite.close()
        return true
    }



    data class RectF(var left: Float, var top: Float, var right: Float, var bottom: Float) {
        fun width() = right - left
        fun height() = bottom - top
        fun area() = width() * height()
    }

    private fun postProcessYolox(
        inputShape: IntArray,
        outputShape: IntArray,
        outputBuffer: ByteArray,
        outputTensorType: Int,
        quantScale: Float,
        quantZeroPoint: Long,
        canvas: Canvas,
        paint: Paint,
        originalW: Int,
        originalH: Int
    ) {
        val ih = inputShape[1]
        val iw = inputShape[2]
        val oh = arrayOf(ih / 8, ih / 16, ih / 32)
        val ow = arrayOf(iw / 8, iw / 16, iw / 32)
        val numCells = oh[0] * ow[0] + oh[1] * ow[1] + oh[2] * ow[2]
        val numElements = 5 + CocoAndImageNetLabels.COCO_CATEGORY.size
        if (numCells != outputShape[1] || numElements != outputShape[2]) {
            Log.e(TAG, "Error! YOLOX output_shape[1,2] mismatch")
            return
        }

        val boxes = mutableListOf<RectF>()
        val scores = mutableListOf<Float>()
        val categories = mutableListOf<Int>()

        var bufIndex = 0
        for (s in 0..2) {
            val stride = 2f.pow(3 + s)
            for (y in 0 until oh[s]) {
                for (x in 0 until ow[s]) {
                    var maxScore = 0.toByte()
                    var maxClass = 0

                    for (cls in 0 until CocoAndImageNetLabels.COCO_CATEGORY.size) {
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

                        boxes.add(RectF(bbCx / iw, bbCy / ih, (bbCx + bbW) / iw, (bbCy + bbH) / ih))
                        scores.add(score)
                        categories.add(maxClass)
                    }

                    bufIndex += numElements
                }
            }
        }

        // Apply NMS
        val selectedIndices = applyNMS(boxes, scores, 0.25f, 0.45f)  // threshold and IOU threshold values

        for (i in selectedIndices) {
            val bbox = boxes[i]
            //paint.color = colors[categories[i] % colors.size] // Set the color based on the category
            canvas.drawRect(
                bbox.left * originalW,
                bbox.top * originalH,
                bbox.right * originalW,
                bbox.bottom * originalH,
                paint
            )
        }
    }

    private fun applyNMS(boxes: List<RectF>, scores: List<Float>, scoreThreshold: Float, iouThreshold: Float): List<Int> {
        val indices = scores.mapIndexed { index, score -> index to score }
            .filter { it.second > scoreThreshold }
            .sortedByDescending { it.second }
            .map { it.first }

        val selectedIndices = mutableListOf<Int>()
        val active = BooleanArray(indices.size) { true }

        for (i in indices.indices) {
            if (!active[i]) continue
            val index = indices[i]
            selectedIndices.add(index)
            for (j in i + 1 until indices.size) {
                if (active[j]) {
                    val otherIndex = indices[j]
                    val iou = computeIOU(boxes[index], boxes[otherIndex])
                    if (iou > iouThreshold) {
                        active[j] = false
                    }
                }
            }
        }
        return selectedIndices
    }

    private fun computeIOU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = maxOf(box1.left, box2.left)
        val intersectionTop = maxOf(box1.top, box2.top)
        val intersectionRight = minOf(box1.right, box2.right)
        val intersectionBottom = minOf(box1.bottom, box2.bottom)

        val intersectionWidth = maxOf(0f, intersectionRight - intersectionLeft)
        val intersectionHeight = maxOf(0f, intersectionBottom - intersectionTop)
        val intersectionArea = intersectionWidth * intersectionHeight

        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)

        val unionArea = box1Area + box2Area - intersectionArea
        return if (unionArea <= 0) 0f else intersectionArea / unionArea
    }

}
