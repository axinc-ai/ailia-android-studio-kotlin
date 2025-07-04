package jp.axinc.ailia_kotlin

import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log
import axip.ailia_tracker.AiliaTracker

class AiliaTrackerSample {
    companion object {
        private const val TAG = "AILIA_Main"
    }

    private var tracker: AiliaTracker? = null
    private var isInitialized = false

    fun initializeTracker(): Boolean {
        return try {
            if (isInitialized) {
                releaseTracker()
            }
            
            tracker = AiliaTracker()
            isInitialized = true
            Log.i(TAG, "Tracker initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize tracker: ${e.javaClass.name}: ${e.message}")
            releaseTracker()
            false
        }
    }

    fun processTracking(canvas: Canvas, paint: Paint, w: Int, h: Int): Long {
        if (!isInitialized || tracker == null) {
            Log.e(TAG, "Tracker not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            
            val result = tracker!!.addTarget(
                category = 0,
                prob = 0.9f,
                x = 0.1f,
                y = 0.1f,
                w = 0.2f,
                h = 0.3f
            )

            if (result == 0) {
                Log.d(TAG, "Successfully added target")
            } else {
                Log.e(TAG, "Failed to add target: $result")
                Log.e(TAG, "Error detail: ${tracker!!.getErrorDetail()}")
                return -1
            }

            val computeResult = tracker!!.compute(w, h)
            if (computeResult == 0) {
                Log.d(TAG, "Successfully computed tracking")

                val objectCount = tracker!!.getObjectCount()
                Log.d(TAG, "Object count: $objectCount")

                for (i in 0 until objectCount) {
                    val obj = tracker!!.getObject(i)
                    obj?.let {
                        Log.d(TAG, "Object $i: id=${it.id}, category=${it.category}, prob=${it.prob}, x=${it.x}, y=${it.y}, w=${it.w}, h=${it.h}")
                        
                        canvas.drawRect(
                            (w * it.x).toFloat(),
                            (h * it.y).toFloat(),
                            (w * (it.x + it.w)).toFloat(),
                            (h * (it.y + it.h)).toFloat(),
                            paint
                        )
                    }
                }
            } else {
                Log.e(TAG, "Failed to compute tracking: $computeResult")
                return -1
            }

            val endTime = System.nanoTime()
            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e(TAG, "Failed to process tracking: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun releaseTracker() {
        try {
            tracker?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing tracker: ${e.javaClass.name}: ${e.message}")
        } finally {
            tracker = null
            isInitialized = false
            Log.i(TAG, "Tracker released")
        }
    }

    fun byte_track(): Boolean {
        Log.d(TAG, "Starting ailia tracker JNI sample")

        val tracker = AiliaTracker()

        try {
            val result = tracker.addTarget(
                category = 0,
                prob = 0.9f,
                x = 0.1f,
                y = 0.1f,
                w = 0.2f,
                h = 0.3f
            )

            if (result == 0) {
                Log.d(TAG, "Successfully added target")
            } else {
                Log.e(TAG, "Failed to add target: $result")
                Log.e(TAG, "Error detail: ${tracker.getErrorDetail()}")
            }

            val computeResult = tracker.compute(640, 480)
            if (computeResult == 0) {
                Log.d(TAG, "Successfully computed tracking")

                val objectCount = tracker.getObjectCount()
                Log.d(TAG, "Object count: $objectCount")

                for (i in 0 until objectCount) {
                    val obj = tracker.getObject(i)
                    obj?.let {
                        Log.d(TAG, "Object $i: id=${it.id}, category=${it.category}, prob=${it.prob}, x=${it.x}, y=${it.y}, w=${it.w}, h=${it.h}")
                    }
                }
            } else {
                Log.e(TAG, "Failed to compute tracking: $computeResult")
            }

        } finally {
            tracker.close()
            Log.d(TAG, "Tracker closed")
        }
        return true;
    }
}
