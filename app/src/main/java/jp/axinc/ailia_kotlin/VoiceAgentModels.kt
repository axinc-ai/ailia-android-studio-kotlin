package jp.axinc.ailia_kotlin

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Base64
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import axip.ailia_llm.AiliaLLM
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * Owns downloads, inference and conversation history. Call only on the session's model executor:
 * QNN model creation and destruction must stay on that thread. Only one model is resident at once.
 */
internal class VoiceAgentModels(private val context: Context) {
    private val speech = AiliaSpeechSample(ModelDownloader.modelDirectory(context))
    private val voice = AiliaVoiceSample(ModelDownloader.modelDirectory(context))
    private var llm: AiliaLLM? = null
    private var speechTurn: VoiceAgentSpeechTurn? = null
    private lateinit var config: AiliaVoiceAgentSample.Config
    private lateinit var llmModelFile: File
    private lateinit var projectorFile: File
    // Each entry is one complete voice turn, including tool calls/results and captured images.
    private val history = mutableListOf<MutableList<JSONObject>>()
    private val tools = VoiceAgentTools()

    fun prepare(config: AiliaVoiceAgentSample.Config, checkActive: () -> Unit,
                onDownload: (String, Long, Long) -> Unit) {
        this.config = config
        val progress = object : ModelDownloadListener {
            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                checkActive()
                onDownload(fileName, bytesDownloaded, totalBytes)
            }
            override fun onComplete() = Unit
            override fun onError(error: String) { checkActive(); error(error) }
        }
        checkActive()
        speech.currentModelType = config.speechModelType
        speech.useQnn = config.speechQnn
        speech.language = "ja"
        check(speech.downloadModel(listener = progress)) { "Whisperモデルのダウンロードに失敗しました" }
        checkActive()

        val modelType = config.llmModelType
        val qnnFile = if (config.llmBackend == LLMBackend.QNN) {
            checkNotNull(QnnSupport.llmQnnFileName(modelType)) { "この端末ではGemmaのQNNを使用できません" }
        } else null
        val fileName = qnnFile ?: modelType.fileName
        val llmProgress = object : ModelDownloader.DownloadListener {
            override fun onProgress(bytesDownloaded: Long, totalBytes: Long) = progress.onProgress(fileName, bytesDownloaded, totalBytes)
            override fun onComplete(file: File) = Unit
            override fun onError(error: String) = progress.onError(error)
        }
        val modelFile = if (qnnFile != null) ModelDownloader.downloadQnnLLMModel(context, fileName, llmProgress)
            else ModelDownloader.downloadLLMModel(context, fileName, llmProgress)
        checkActive()
        llmModelFile = checkNotNull(modelFile) { "Gemmaモデルのダウンロードに失敗しました" }

        val projectorName = checkNotNull(SharedLLM.projectorFileName(modelType, config.llmBackend)) {
            "このモデルでは画像入力を使用できません"
        }
        val projectorProgress = object : ModelDownloader.DownloadListener {
            override fun onProgress(bytesDownloaded: Long, totalBytes: Long) = progress.onProgress(projectorName, bytesDownloaded, totalBytes)
            override fun onComplete(file: File) = Unit
            override fun onError(error: String) = progress.onError(error)
        }
        projectorFile = checkNotNull(SharedLLM.download(context, projectorName, config.llmBackend, projectorProgress)) {
            "画像エンコーダのダウンロードに失敗しました"
        }
        checkActive()

        voice.modelType = VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE
        if (config.speechOutputEnabled || config.inputMode == AiliaVoiceAgentSample.InputMode.SYNTHESIZED_QUERIES) {
            check(voice.downloadModel(progress)) { "GPT-SoVITSモデルのダウンロードに失敗しました" }
        }
        checkActive()
        history.clear()
        tools.reset()
    }

    /** Roll back the complete voice turn if any part, including playback, is cancelled. */
    fun withConversationTurn(isCancelled: () -> Boolean, action: () -> Unit) {
        val historyBeforeTurn = history.toList()
        try {
            action()
        } finally {
            if (isCancelled()) {
                history.clear()
                history.addAll(historyBeforeTurn)
            }
        }
    }

    fun startListening(checkActive: () -> Unit) {
        checkActive()
        check(speech.initializeSpeech(config.speechEnvId, liveMode = false)) { "Whisperの初期化に失敗しました" }
        checkActive()
        speechTurn = speech.createBufferedTurn()
    }

    fun pushSpeech(audio: FloatArray, channels: Int, sampleRate: Int): Boolean =
        checkNotNull(speechTurn) { "Speech input not initialized" }.push(audio, channels, sampleRate)

    fun finishRecognition(checkActive: () -> Unit): String {
        val input = try {
            checkActive()
            voiceAgentTranscript(checkNotNull(speechTurn).finish(checkActive))
        } finally {
            speechTurn = null
            speech.releaseSpeech()
        }
        checkActive()
        return input
    }

    fun generateResponse(input: String, checkActive: () -> Unit, onTemperature: (Double) -> Unit,
                         takePhoto: () -> File, onPhoto: (File) -> Unit, onOutput: (String) -> Unit): String {
        val model = AiliaLLM()
        llm = model
        try {
            model.openModelFile(llmModelFile.absolutePath, if (config.llmBackend == LLMBackend.QNN) 0 else 4096)
            model.openMultimodalProjectorFile(projectorFile.absolutePath)
            checkActive()
            // Keep whole user turns: assistant tool calls and their results must stay together.
            while (history.size >= 4) removeOldestTurn()
            val turn = mutableListOf(JSONObject().put("role", "user").put("content", input))
            history += turn
            val contextSize = model.getContextSize()
            val definitions = tools.definitions()
            val fixedTokens = model.getTokenCount(definitions) + model.getTokenCount(SYSTEM_PROMPT) + 64
            model.setSamplingParams(40, 0.9f, 0.0f, 1234)
            model.setTools(definitions)
            model.setThinking(false)
            repeat(8) {
                checkActive()
                while (fixedTokens + history.flatten().sumOf { messageTokens(model, it) } > contextSize - 768) {
                    if (!removeOldestTurn()) break
                }
                model.setPromptJson(JSONArray().put(JSONObject().put("role", "system").put("content", SYSTEM_PROMPT)).apply {
                    history.flatten().forEach { put(it) }
                }.toString())
                Log.d("VoiceAgent", "Tool prompt tokens: ${model.getPromptTokenCount()}")
                val generationLimit = minOf(512, contextSize - model.getPromptTokenCount() - 1)
                check(generationLimit > 0) { "音声入力が長すぎます。対話を開始し直してください" }
                var complete = false
                for (step in 0 until generationLimit) {
                    checkActive()
                    complete = model.generate()
                    if (complete) break
                }
                checkActive()
                check(complete) { "応答の生成が上限に達しました。対話を開始し直してください" }
                val response = JSONObject(model.getResponseJson())
                Log.d("VoiceAgent", "Response JSON: $response")
                turn += response
                val calls = response.optJSONArray("tool_calls")
                if (calls == null || calls.length() == 0) {
                    val text = response.optString("content").trim()
                    check(text.isNotEmpty()) { "応答が空でした。対話を開始し直してください" }
                    onOutput(text)
                    return text
                }
                val photos = mutableListOf<JSONObject>()
                for (index in 0 until calls.length()) {
                    checkActive()
                    val call = calls.getJSONObject(index)
                    val function = call.getJSONObject("function")
                    val name = function.getString("name")
                    val result = tools.execute(name, function.getString("arguments"), onTemperature) {
                        val file = takePhoto()
                        try {
                            checkActive()
                            photos += photoPart(file)
                            checkActive()
                            onPhoto(file)
                            file
                        } catch (error: Exception) {
                            file.delete()
                            throw error
                        }
                    }
                    Log.i("VoiceAgent", "Tool call: $name -> $result")
                    turn += JSONObject().put("role", "tool")
                        .put("tool_call_id", call.getString("id")).put("content", result)
                }
                // The SDK accepts media in user messages; finish all matched tool results first.
                if (photos.isNotEmpty()) {
                    turn += JSONObject().put("role", "user").put("content", JSONArray()
                        .put(JSONObject().put("type", "text").put("text", "撮影した写真です。元の依頼に、この写真を踏まえて答えてください。"))
                        .apply { photos.forEach { put(it) } })
                }
            }
            error("ツールの呼び出しが上限に達しました。もう一度短く話しかけてください")
        } finally {
            model.destroy()
            llm = null
        }
    }

    private fun removeOldestTurn(): Boolean {
        if (history.size <= 1) return false
        history.removeAt(0)
        return true
    }

    /** Reserve image tokens without tokenizing the much larger Base64 payload as text. */
    private fun messageTokens(model: AiliaLLM, message: JSONObject): Int {
        val content = message.optJSONArray("content") ?: return model.getTokenCount(message.toString()) + 32
        var count = 32
        for (index in 0 until content.length()) {
            val part = content.getJSONObject(index)
            count += if (part.optString("type") == "image") 512 else model.getTokenCount(part.optString("text"))
        }
        return count
    }

    /** Store bounded, upright JPEG data in history independently of the preview file lifetime. */
    private fun photoPart(file: File): JSONObject {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true; inSampleSize = 1 }
        BitmapFactory.decodeFile(file.absolutePath, options)
        require(options.outWidth > 0 && options.outHeight > 0) { "写真を読み込めませんでした" }
        while (maxOf(options.outWidth, options.outHeight) / options.inSampleSize > 512) options.inSampleSize *= 2
        options.inJustDecodeBounds = false
        val bitmap = checkNotNull(BitmapFactory.decodeFile(file.absolutePath, options)) { "写真を読み込めませんでした" }
        var upright: Bitmap? = null
        try {
            val exif = ExifInterface(file.absolutePath)
            val matrix = Matrix().apply {
                postRotate(exif.rotationDegrees.toFloat())
                if (exif.isFlipped) postScale(-1f, 1f)
            }
            upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            val bytes = ByteArrayOutputStream().use { output ->
                check(upright.compress(Bitmap.CompressFormat.JPEG, 85, output)) { "写真の変換に失敗しました" }
                output.toByteArray()
            }
            return JSONObject().put("type", "image").put("data", Base64.encodeToString(bytes, Base64.NO_WRAP))
        } finally {
            if (upright !== bitmap) upright?.recycle()
            bitmap.recycle()
        }
    }

    companion object {
        private const val SYSTEM_PROMPT = "You are a helpful assistant. Use tools to perform requested actions. Reply briefly in Japanese after receiving tool results."
    }

    fun synthesize(text: String, checkActive: () -> Unit): AudioUtil.WavFileData =
        withVoice(checkActive) { synthesize -> synthesize(text) }

    /** Reuse one voice model/reference throughout a response; never overlap native model calls. */
    fun <T> withVoice(checkActive: () -> Unit, action: (synthesize: (String) -> AudioUtil.WavFileData) -> T): T {
        checkActive()
        val reference = AudioUtil().loadRawAudio(context.resources.openRawResource(R.raw.reference_audio_girl))
        try {
            check(voice.initializeVoice(config.voiceEnvId)) { "GPT-SoVITSの初期化に失敗しました" }
            checkActive()
            return action { text -> synthesizeInitialized(text, reference, checkActive) }
        } finally {
            voice.releaseVoice()
        }
    }

    private fun synthesizeInitialized(text: String, reference: AudioUtil.WavFileData,
                                      checkActive: () -> Unit): AudioUtil.WavFileData {
        checkActive()
        check(voice.textToSpeech(reference.audioData, reference.channels, reference.sampleRate,
            "水をマレーシアから買わなくてはならない。", "ja", text, "ja", autoPlay = false) >= 0) {
            "音声合成に失敗しました"
        }
        checkActive()
        return AudioUtil.WavFileData(voice.lastAudioSampleRate, voice.lastAudioChannels, checkNotNull(voice.lastAudioData))
    }

    fun release(clearHistory: Boolean = true) {
        speechTurn = null
        speech.releaseSpeech()
        voice.releaseVoice()
        runCatching { llm?.destroy() }.onFailure { Log.w("VoiceAgent", "Failed to release LLM", it) }
        llm = null
        if (clearHistory) history.clear()
    }
}
