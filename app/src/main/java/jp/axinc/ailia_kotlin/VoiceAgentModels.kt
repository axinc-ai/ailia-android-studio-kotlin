package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Log
import axip.ailia_llm.AiliaLLM
import axip.ailia_llm.AiliaLLMChatMessage
import java.io.File

/**
 * Owns downloads, inference and conversation history. Call only on the session's model executor:
 * QNN model creation and destruction must stay on that thread. Only one model is resident at once.
 */
internal class VoiceAgentModels(private val context: Context) {
    private val speech = AiliaSpeechSample(ModelDownloader.modelDirectory(context))
    private val voice = AiliaVoiceSample(ModelDownloader.modelDirectory(context))
    private var llm: AiliaLLM? = null
    private lateinit var config: AiliaVoiceAgentSample.Config
    private lateinit var llmModelFile: File
    private val history = mutableListOf<AiliaLLMChatMessage>()

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
        speech.currentModelType = SpeechModelType.WHISPER_LARGE_V3_TURBO
        speech.useQnn = config.speechQnn
        speech.language = "ja"
        check(speech.downloadModel(listener = progress)) { "Whisperモデルのダウンロードに失敗しました" }
        checkActive()

        val modelType = LLMModelType.GEMMA_4_E2B
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

        voice.modelType = VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE
        check(voice.downloadModel(progress)) { "GPT-SoVITSモデルのダウンロードに失敗しました" }
        checkActive()
        history.clear()
        history += AiliaLLMChatMessage("system", "あなたは音声対話アシスタントです。日本語で、自然な話し言葉の1〜3文で簡潔に答えてください。マークダウン、箇条書き、絵文字は使わないでください。")
    }

    /** Roll back both messages if any part of a turn, including playback, is cancelled. */
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

    fun recognize(audio: AudioUtil.WavFileData, checkActive: () -> Unit): String {
        val input = try {
            check(speech.initializeSpeech(config.speechEnvId)) { "Whisperの初期化に失敗しました" }
            checkActive()
            voiceAgentTranscript(speech.process(audio.audioData, audio.channels, audio.sampleRate))
        } finally {
            speech.releaseSpeech()
        }
        checkActive()
        return input
    }

    fun generateResponse(input: String, checkActive: () -> Unit, onOutput: (String) -> Unit): String {
        val model = AiliaLLM()
        llm = model
        try {
            model.openModelFile(llmModelFile.absolutePath, if (config.llmBackend == LLMBackend.QNN) 0 else 4096)
            checkActive()
            // Bound conversation memory; no transcript history is rendered on screen.
            while (history.size > 9) { history.removeAt(1); history.removeAt(1) }
            history += AiliaLLMChatMessage("user", input)
            val contextSize = model.getContextSize()
            // Leave room for generation and the model's chat template, including on smaller QNN contexts.
            while (history.size > 2 && history.sumOf { model.getTokenCount(it.content) + 32 } > contextSize - 640) {
                history.removeAt(1)
                history.removeAt(1)
            }
            model.setTools(null)
            model.setThinking(false)
            model.setSamplingParams(40, 0.9f, 0.4f, 1234)
            model.setPrompt(history.toTypedArray())
            val generationLimit = minOf(512, contextSize - model.getPromptTokenCount() - 1)
            check(generationLimit > 0) { "音声入力が長すぎます。対話を開始し直してください" }
            val output = StringBuilder()
            var complete = false
            for (step in 0 until generationLimit) {
                checkActive()
                complete = model.generate()
                output.append(model.getDeltaText())
                onOutput(output.toString())
                if (complete) break
            }
            checkActive()
            check(complete) { "応答の生成が上限に達しました。対話を開始し直してください" }
            val response = output.toString().trim()
            check(response.isNotEmpty()) { "応答が空でした。対話を開始し直してください" }
            history += AiliaLLMChatMessage("assistant", response)
            return response
        } finally {
            model.destroy()
            llm = null
        }
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
        speech.releaseSpeech()
        voice.releaseVoice()
        runCatching { llm?.destroy() }.onFailure { Log.w("VoiceAgent", "Failed to release LLM", it) }
        llm = null
        if (clearHistory) history.clear()
    }
}
