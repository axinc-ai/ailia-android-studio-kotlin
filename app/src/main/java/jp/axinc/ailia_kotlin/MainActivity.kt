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
import android.widget.*
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
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageView: ImageView
    private lateinit var cameraPreviewView: PreviewView
    private lateinit var modeRadioGroup: RadioGroup
    private lateinit var algorithmSpinner: Spinner
    private lateinit var processingTimeTextView: TextView
    private lateinit var resultScrollView: ScrollView
    private lateinit var classificationResultTextView: TextView
    private lateinit var tokenizerInputEditText: EditText
    private lateinit var tokenizerOutputTextView: TextView
    private lateinit var trackingResultTextView: TextView
    
    private var poseEstimatorSample = AiliaPoseEstimatorSample()
    private var objectDetectionSample = AiliaTFLiteObjectDetectionSample()
    private var classificationSample = AiliaTFLiteClassificationSample()
    private var tokenizerSample = AiliaTokenizerSample()
    private var trackerSample = AiliaTrackerSample()
    
    private var selectedEnv: AiliaEnvironment? = null
    private var isInitialized = false
    private var currentAlgorithm = AlgorithmType.POSE_ESTIMATION
    private var pendingAlgorithmSwitch: AlgorithmType? = null
    private var isProcessing = AtomicBoolean(false)
    
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null

    enum class AlgorithmType {
        POSE_ESTIMATION,
        OBJECT_DETECTION,
        TRACKING,
        TOKENIZE,
        CLASSIFICATION
    }
    
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
        updateUIVisibility()
        
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
        algorithmSpinner = findViewById(R.id.algorithmSpinner)
        processingTimeTextView = findViewById(R.id.processingTimeTextView)
        resultScrollView = findViewById(R.id.resultScrollView)
        classificationResultTextView = findViewById(R.id.classificationResultTextView)
        tokenizerInputEditText = findViewById(R.id.tokenizerInputEditText)
        tokenizerOutputTextView = findViewById(R.id.tokenizerOutputTextView)
        trackingResultTextView = findViewById(R.id.trackingResultTextView)
    }
    
    private fun setupModeSelection() {
        val algorithms = arrayOf(
            "PoseEstimation",
            "ObjectDetection", 
            "Tracking",
            "Tokenize",
            "Classification"
        )
        
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, algorithms)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        algorithmSpinner.adapter = adapter
        
        algorithmSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val newAlgorithm = AlgorithmType.values()[position]
                if (newAlgorithm != currentAlgorithm) {
                    switchAlgorithm(newAlgorithm)
                }
            }
            
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
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
    
    private fun processAlgorithm(img: ByteArray, bitmap: Bitmap, canvas: Canvas, w: Int, h: Int): Long {
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
        
        return when (currentAlgorithm) {
            AlgorithmType.POSE_ESTIMATION -> {
                poseEstimatorSample.processPoseEstimation(img, canvas, paint, w, h)
            }
            AlgorithmType.OBJECT_DETECTION -> {
                objectDetectionSample.processObjectDetection(bitmap, canvas, paint2, textPaint, w, h)
            }
            AlgorithmType.CLASSIFICATION -> {
                val time = classificationSample.processClassification(bitmap)
                val result = classificationSample.getLastClassificationResult()
                runOnUiThread {
                    classificationResultTextView.text = "Classification Result: $result"
                }
                time
            }
            AlgorithmType.TOKENIZE -> {
                val inputText = tokenizerInputEditText.text.toString().ifEmpty { "Hello world from ailia!" }
                val time = tokenizerSample.processTokenization(inputText)
                val tokens = tokenizerSample.getLastTokenizationResult()
                runOnUiThread {
                    tokenizerOutputTextView.text = "Tokens: $tokens"
                }
                time
            }
            AlgorithmType.TRACKING -> {
                // First run object detection to get detection results without drawing
                val detectionTime = objectDetectionSample.processObjectDetectionWithoutDrawing(bitmap, w, h)
                val detectionResults = objectDetectionSample.getDetectionResults(bitmap)
                // Then run tracking with the detection results and draw the tracking results
                val trackingTime = trackerSample.processTrackingWithDetections(canvas, paint2, w, h, detectionResults)
                val trackingInfo = trackerSample.getLastTrackingResult()
                runOnUiThread {
                    trackingResultTextView.text = "Tracking Results: $trackingInfo"
                }
                detectionTime + trackingTime
            }
        }
    }
    
    private fun updateUIVisibility() {
        val isImageMode = modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton
        val isCameraMode = modeRadioGroup.checkedRadioButtonId == R.id.cameraRadioButton
        
        when (currentAlgorithm) {
            AlgorithmType.TOKENIZE -> {
                imageView.visibility = View.GONE
                cameraPreviewView.visibility = View.GONE
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.VISIBLE
                tokenizerOutputTextView.visibility = View.VISIBLE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.VISIBLE
            }
            AlgorithmType.CLASSIFICATION -> {
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.GONE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.VISIBLE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
            }
            AlgorithmType.TRACKING -> {
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.GONE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.VISIBLE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.VISIBLE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
            }
            else -> {
                if (isImageMode) {
                    imageView.visibility = View.VISIBLE
                    cameraPreviewView.visibility = View.GONE
                } else {
                    imageView.visibility = View.GONE
                    cameraPreviewView.visibility = View.VISIBLE
                }
                resultScrollView.visibility = View.GONE
                classificationResultTextView.visibility = View.GONE
                tokenizerInputEditText.visibility = View.GONE
                tokenizerOutputTextView.visibility = View.GONE
                trackingResultTextView.visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerInputLabel).visibility = View.GONE
                findViewById<TextView>(R.id.tokenizerOutputLabel).visibility = View.GONE
            }
        }
    }
    
    private fun switchAlgorithm(newAlgorithm: AlgorithmType) {
        if (isProcessing.get()) {
            Log.i("AILIA_Main", "Processing active, queuing algorithm switch to ${newAlgorithm.name}")
            pendingAlgorithmSwitch = newAlgorithm
            return
        }
        
        executeAlgorithmSwitch(newAlgorithm)
    }
    
    private fun executeAlgorithmSwitch(newAlgorithm: AlgorithmType) {
        releaseCurrentAlgorithm()
        currentAlgorithm = newAlgorithm
        isInitialized = false
        updateUIVisibility()
        
        if (modeRadioGroup.checkedRadioButtonId == R.id.imageRadioButton) {
            processImageMode()
        }
    }
    
    private fun releaseCurrentAlgorithm() {
        try {
            poseEstimatorSample.releasePoseEstimator()
            objectDetectionSample.releaseObjectDetection()
            classificationSample.releaseClassification()
            tokenizerSample.releaseTokenizer()
            trackerSample.releaseTracker()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing algorithms: ${e.message}")
        }
    }
    
    private fun switchToImageMode() {
        if (isProcessing.get()) {
            Log.w("AILIA_Main", "Cannot switch mode while processing")
            return
        }
        
        updateUIVisibility()
        stopCamera()
        processImageMode()
    }
    
    private fun switchToCameraMode() {
        if (isProcessing.get()) {
            Log.w("AILIA_Main", "Cannot switch mode while processing")
            return
        }
        
        if (allPermissionsGranted()) {
            updateUIVisibility()
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            modeRadioGroup.check(R.id.imageRadioButton)
        }
    }
    
    private fun initializeAilia() {
        try {
            selectedEnv = poseEstimatorSample.ailia_environment(cacheDir.absolutePath)
            
            when (currentAlgorithm) {
                AlgorithmType.POSE_ESTIMATION -> {
                    val proto: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_proto)
                    val model: ByteArray? = loadRawFile(R.raw.lightweight_human_pose_weight)
                    isInitialized = poseEstimatorSample.initializePoseEstimator(selectedEnv!!.id, proto, model)
                }
                AlgorithmType.OBJECT_DETECTION -> {
                    val yoloxModel: ByteArray? = loadRawFile(R.raw.yolox_tiny)
                    isInitialized = objectDetectionSample.initializeObjectDetection(yoloxModel)
                }
                AlgorithmType.CLASSIFICATION -> {
                    val classificationModel: ByteArray? = loadRawFile(R.raw.mobilenetv2)
                    isInitialized = classificationSample.initializeClassification(classificationModel)
                }
                AlgorithmType.TOKENIZE -> {
                    isInitialized = tokenizerSample.initializeTokenizer()
                }
                AlgorithmType.TRACKING -> {
                    isInitialized = trackerSample.initializeTracker()
                }
            }
            
            if (isInitialized) {
                Log.i("AILIA_Main", "Algorithm ${currentAlgorithm.name} initialized successfully")
            } else {
                Log.e("AILIA_Error", "Failed to initialize algorithm ${currentAlgorithm.name}")
            }
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error initializing algorithm ${currentAlgorithm.name}: ${e.message}")
        }
    }
    
    private fun processImageMode() {
        if (isProcessing.get()) {
            return
        }
        
        isProcessing.set(true)
        
        try {
            if (!isInitialized) {
                initializeAilia()
            }
            
            if (!isInitialized) {
                runOnUiThread {
                    processingTimeTextView.text = "Failed to initialize ${currentAlgorithm.name}"
                }
                return
            }
            
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
            
            val processingTime = processAlgorithm(img, personBmp, canvas, w, h)
            
            runOnUiThread {
                if (currentAlgorithm != AlgorithmType.TOKENIZE) {
                    imageView.setImageBitmap(bitmap)
                }
                processingTimeTextView.text = "Processing Time: ${processingTime}ms (${currentAlgorithm.name})"
            }
            
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error in image mode: ${e.message}")
            runOnUiThread {
                processingTimeTextView.text = "Processing Error: ${e.message}"
            }
        } finally {
            isProcessing.set(false)
            
            pendingAlgorithmSwitch?.let { pendingAlgorithm ->
                pendingAlgorithmSwitch = null
                executeAlgorithmSwitch(pendingAlgorithm)
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
        try {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
            cameraProviderFuture.addListener({
                val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
                camera = null
                imageAnalyzer = null
            }, ContextCompat.getMainExecutor(this))
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error stopping camera: ${e.message}")
        }
    }
    
    private inner class CameraFrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            if (isInitialized) {
                processCameraFrame(image)
            }
            image.close()
        }
        
        private fun processCameraFrame(image: ImageProxy) {
            if (isProcessing.get()) {
                return
            }
            
            isProcessing.set(true)
            
            try {
                val bitmap = imageProxyToBitmap(image)
                val img = loadRawImage(bitmap)
                val w = bitmap.width
                val h = bitmap.height
                
                val resultBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(resultBitmap)
                canvas.drawBitmap(bitmap, 0f, 0f, null)
                
                val processingTime = processAlgorithm(img, bitmap, canvas, w, h)
                
                runOnUiThread {
                    val fps = if (processingTime > 0) 1000 / processingTime else 0
                    processingTimeTextView.text = "Processing Time: ${processingTime}ms (${currentAlgorithm.name}) - FPS: $fps"
                }
                
            } catch (e: Exception) {
                Log.e("AILIA_Error", "Error processing camera frame: ${e.message}")
            } finally {
                isProcessing.set(false)
                
                pendingAlgorithmSwitch?.let { pendingAlgorithm ->
                    pendingAlgorithmSwitch = null
                    runOnUiThread {
                        executeAlgorithmSwitch(pendingAlgorithm)
                    }
                }
            }
        }
    }
    
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, image.width, image.height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
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
        releaseCurrentAlgorithm()
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
