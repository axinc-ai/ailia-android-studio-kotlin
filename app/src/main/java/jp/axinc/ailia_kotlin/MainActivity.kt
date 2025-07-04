package jp.axinc.ailia_kotlin

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.Options
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import axip.ailia.*
import axip.ailia_tflite.*
import java.io.*
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageView: ImageView
    private lateinit var cameraPreviewView: PreviewView
    private lateinit var modeRadioGroup: RadioGroup
    private lateinit var processingTimeTextView: TextView
    
    private var poseEstimatorSample = AiliaPoseEstimatorSample()
    private var objectDetectionSample = AiliaTFLiteObjectDetectionSample()
    private var selectedEnv: AiliaEnvironment? = null
    private var isInitialized = false
    
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        
        init {
            System.loadLibrary("ailia")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupModeSelection()
        
        if (allPermissionsGranted()) {
            initializeAilia()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        
        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    
    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        cameraPreviewView = findViewById(R.id.cameraPreviewView)
        modeRadioGroup = findViewById(R.id.modeRadioGroup)
        processingTimeTextView = findViewById(R.id.processingTimeTextView)
    }
    
    private fun setupModeSelection() {
        modeRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.imageRadioButton -> {
                    switchToImageMode()
                }
                R.id.cameraRadioButton -> {
                    switchToCameraMode()
                }
            }
        }
        
        switchToImageMode()
    }
    
    private fun switchToImageMode() {
        imageView.visibility = View.VISIBLE
        cameraPreviewView.visibility = View.GONE
        stopCamera()
        processImageMode()
    }
    
    private fun switchToCameraMode() {
        if (allPermissionsGranted()) {
            imageView.visibility = View.GONE
            cameraPreviewView.visibility = View.VISIBLE
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            modeRadioGroup.check(R.id.imageRadioButton)
        }
    }
    
    private fun initializeAilia() {
        try {
            selectedEnv = poseEstimatorSample.ailia_environment(cacheDir.absolutePath)
            
            val proto: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_proto)
            val model: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_weight)
            val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_tiny)
            
            if (poseEstimatorSample.initializePoseEstimator(selectedEnv!!.id, proto, model) &&
                objectDetectionSample.initializeObjectDetection(yoloxModel)) {
                isInitialized = true
                Log.i("AILIA_Main", "Ailia initialized successfully")
            } else {
                Log.e("AILIA_Error", "Failed to initialize Ailia")
            }
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error initializing Ailia: ${e.message}")
        }
    }
    
    private fun processImageMode() {
        if (!isInitialized) {
            initializeAilia()
        }
        
        try {
            val options = Options()
            options.inScaled = false
            val personBmp = BitmapFactory.decodeResource(this.resources, R.raw.person, options)
            
            val img = loadRawImage(personBmp)
            val w = personBmp.width
            val h = personBmp.height
            
            val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(img))
            
            val canvas = Canvas(bitmap)
            val paint = Paint().apply {
                color = Color.WHITE
            }
            
            val paint2 = Paint().apply {
                style = Paint.Style.STROKE
                color = Color.RED
                strokeWidth = 5f
            }
            
            val textPaint = Paint().apply {
                color = Color.BLACK
                textSize = 50f
                isAntiAlias = true
            }
            
            val startTime = System.nanoTime()
            
            val poseTime = poseEstimatorSample.processPoseEstimation(img, canvas, paint, w, h)
            val detectionTime = objectDetectionSample.processObjectDetection(personBmp, canvas, paint2, textPaint, w, h)
            
            val totalTime = (System.nanoTime() - startTime) / 1000000
            
            runOnUiThread {
                imageView.setImageBitmap(bitmap)
                processingTimeTextView.text = "Processing Time: ${totalTime}ms (Pose: ${poseTime}ms, Detection: ${detectionTime}ms)"
            }
            
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error in image mode: ${e.message}")
            runOnUiThread {
                processingTimeTextView.text = "Processing Error: ${e.message}"
            }
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(cameraPreviewView.surfaceProvider)
            }
            
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, CameraFrameAnalyzer())
                }
            
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("AILIA_Error", "Use case binding failed", exc)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun stopCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()
        }, ContextCompat.getMainExecutor(this))
    }
    
    private inner class CameraFrameAnalyzer : ImageAnalysis.Analyzer {
        private var lastAnalyzedTimestamp = 0L
        private val frameRateLimit = 100L
        
        override fun analyze(image: ImageProxy) {
            val currentTimestamp = System.currentTimeMillis()
            if (currentTimestamp - lastAnalyzedTimestamp >= frameRateLimit) {
                if (isInitialized) {
                    processCameraFrame(image)
                }
                lastAnalyzedTimestamp = currentTimestamp
            }
            image.close()
        }
        
        private fun processCameraFrame(image: ImageProxy) {
            try {
                val bitmap = imageProxyToBitmap(image)
                val img = loadRawImage(bitmap)
                val w = bitmap.width
                val h = bitmap.height
                
                val resultBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(resultBitmap)
                canvas.drawBitmap(bitmap, 0f, 0f, null)
                
                val paint = Paint().apply {
                    color = Color.WHITE
                }
                
                val paint2 = Paint().apply {
                    style = Paint.Style.STROKE
                    color = Color.RED
                    strokeWidth = 3f
                }
                
                val textPaint = Paint().apply {
                    color = Color.BLACK
                    textSize = 30f
                    isAntiAlias = true
                }
                
                val startTime = System.nanoTime()
                val poseTime = poseEstimatorSample.processPoseEstimation(img, canvas, paint, w, h)
                val detectionTime = objectDetectionSample.processObjectDetection(bitmap, canvas, paint2, textPaint, w, h)
                val totalTime = (System.nanoTime() - startTime) / 1000000
                
                runOnUiThread {
                    processingTimeTextView.text = "Processing Time: ${totalTime}ms (Pose: ${poseTime}ms, Detection: ${detectionTime}ms) - FPS: ${1000/totalTime.coerceAtLeast(1)}"
                }
                
            } catch (e: Exception) {
                Log.e("AILIA_Error", "Error processing camera frame: ${e.message}")
            }
        }
    }
    
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initializeAilia()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        poseEstimatorSample.releasePoseEstimator()
        objectDetectionSample.releaseObjectDetection()
        cameraExecutor.shutdown()
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
            out.writeByte(i shr 16 and 0xff)
            out.writeByte(i shr 8 and 0xff)
            out.writeByte(i shr 0 and 0xff)
            out.writeByte(i shr 24 and 0xff)
        }
        return bout.toByteArray()
    }
}
