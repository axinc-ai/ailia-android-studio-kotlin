package jp.axinc.ailia_kotlin

import android.util.Log

import axip.ailia_tracker.AiliaTracker

class AiliaTrackerSample {
    companion object {
        private const val TAG = "AILIA_Main"
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