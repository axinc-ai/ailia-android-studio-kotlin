package jp.axinc.ailia_kotlin

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Handler
import android.os.Looper
import android.util.Log
import axip.ailia_llm.AiliaLLM
import axip.ailia_llm.AiliaLLMChatMessage
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToInt

/**
 * Whisper → Gemma → GPT-SoVITS, with automatic microphone turn taking.
 * All native model creation, inference and destruction run on the supplied model executor.
 * The executor must be the same one used for the application's other LLMs (QNN thread affinity).
 * Microphone capture is stopped during recognition, generation and playback to avoid feedback.
 * Only one model is resident at a time so the sample can run on memory-constrained phones.
 */
class AiliaVoiceAgentSample(context: Context, private val executor: ExecutorService) {
    enum class State {
        PREPARING, LISTENING, SYNTHESIZING_QUERY, PLAYING_QUERY,
        SUBMITTING, RECOGNIZING, THINKING, SYNTHESIZING, SPEAKING, CANCELLING, TEST_COMPLETE, STOPPING
    }
    enum class InputMode { MICROPHONE, SYNTHESIZED_QUERIES }
    data class Config(
        val speechEnvId: Int,
        val speechQnn: Boolean,
        val llmBackend: LLMBackend,
        val voiceEnvId: Int,
        val inputMode: InputMode = InputMode.MICROPHONE,
        val testQueries: List<String> = emptyList(),
    ) {
        init {
            require(inputMode != InputMode.SYNTHESIZED_QUERIES ||
                (testQueries.isNotEmpty() && testQueries.all { it.isNotBlank() })) {
                "テストクエリを1行に1件ずつ入力してください"
            }
        }
    }
    interface Listener {
        fun onState(state: State)
        fun onInput(text: String)
        fun onOutput(text: String)
        fun onWaveform(samples: FloatArray)
        fun onDownload(fileName: String, downloaded: Long, total: Long)
        fun onStopped(error: String?)
    }

    private val context = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val generation = AtomicLong()
    private val speech = AiliaSpeechSample(ModelDownloader.modelDirectory(context))
    private val voice = AiliaVoiceSample(ModelDownloader.modelDirectory(context))
    private var llm: AiliaLLM? = null
    private lateinit var config: Config
    private lateinit var llmModelFile: File
    private val history = mutableListOf<AiliaLLMChatMessage>()
    private var listener: Listener? = null
    private var currentState = State.STOPPING
    @Volatile private var inputMode = InputMode.MICROPHONE
    private val playbackLock = Any()
    private var playbackTrack: AudioTrack? = null
    // Accessed on the UI thread, except for token-gated recorder callbacks.
    private class MicInput(val recorder: WavRecorder) {
        val hasSamples = AtomicBoolean(false)
        val manualSubmit = AtomicBoolean(false)
    }
    @Volatile private var micInput: MicInput? = null
    var isActive = false
        private set
    private var stopping = false

    fun start(config: Config, listener: Listener) {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (isActive || stopping) return
        this.listener = listener
        inputMode = config.inputMode
        isActive = true
        val token = generation.incrementAndGet()
        notifyState(State.PREPARING)
        executor.execute {
            try {
                initialize(token, config)
                when (config.inputMode) {
                    InputMode.MICROPHONE -> emit(token) { listen(token) }
                    InputMode.SYNTHESIZED_QUERIES -> runTestQueries(token)
                }
            } catch (_: CancellationException) {
                // stop() already queued resource release on this executor.
            } catch (e: Exception) {
                fail(token, e)
            }
        }
    }

    private fun initialize(token: Long, config: Config) {
        this.config = config
        val progress = object : ModelDownloadListener {
            override fun onProgress(fileName: String, bytesDownloaded: Long, totalBytes: Long) {
                ensureActive(token)
                emit(token) { listener?.onDownload(fileName, bytesDownloaded, totalBytes) }
            }
            override fun onComplete() = Unit
            override fun onError(error: String) { ensureActive(token); error(error) }
        }
        ensureActive(token)
        speech.currentModelType = SpeechModelType.WHISPER_LARGE_V3_TURBO
        speech.useQnn = config.speechQnn
        speech.language = "ja"
        check(speech.downloadModel(listener = progress)) { "Whisperモデルのダウンロードに失敗しました" }
        ensureActive(token)

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
        ensureActive(token)
        llmModelFile = checkNotNull(modelFile) { "Gemmaモデルのダウンロードに失敗しました" }

        voice.modelType = VoiceModelType.GPT_SOVITS_V2_PRO_DISTILL_BASE
        check(voice.downloadModel(progress)) { "GPT-SoVITSモデルのダウンロードに失敗しました" }
        ensureActive(token)
        history.clear()
        history += AiliaLLMChatMessage("system", "あなたは音声対話アシスタントです。日本語で、自然な話し言葉の1〜3文で簡潔に答えてください。マークダウン、箇条書き、絵文字は使わないでください。")
    }

    /** UI thread only. A fresh recorder prevents a cancelled capture from touching the next turn. */
    private fun listen(token: Long) {
        if (generation.get() != token) return
        notifyState(State.LISTENING)
        val endpoint = VoiceAgentEndpoint()
        val capture = WavRecorder()
        val turn = MicInput(capture)
        micInput = turn
        val file = File(context.cacheDir, "voice_agent_$token.wav")
        capture.startRecording(file, object : WavRecorder.RecordingListener {
            override fun onWaveform(chunk: FloatArray, sampleRate: Int) {
                if (generation.get() != token) return
                turn.hasSamples.set(true)
                emit(token) { listener?.onWaveform(chunk) }
                if (endpoint.accept(chunk)) capture.stopRecording()
            }
            override fun onCompleted(file: File, audio: FloatArray, sampleRate: Int) {
                file.delete()
                if (generation.get() != token) return
                if (!turn.manualSubmit.get() && !endpoint.hasSpeech) {
                    emit(token) { listen(token) }
                    return
                }
                // Recognition may be queued behind a cancelled native synthesis call.
                emit(token) { notifyState(State.SUBMITTING) }
                executor.execute {
                    try {
                        respond(token, endpoint.trim(audio), sampleRate)
                        emit(token) { listen(token) }
                    } catch (_: CancellationException) {
                    } catch (e: Exception) {
                        fail(token, e)
                    }
                }
            }
            override fun onError(error: String) = fail(token, IllegalStateException(error))
        })
    }

    /** Finish the current recording immediately, including speech too short/quiet for endpointing. */
    fun submitMicrophoneInput(): Boolean {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (!isActive || inputMode != InputMode.MICROPHONE) return false
        val turn = micInput ?: return false
        if (!turn.hasSamples.get() || !turn.recorder.isRecording || !turn.manualSubmit.compareAndSet(false, true)) return false
        notifyState(State.SUBMITTING)
        turn.recorder.stopRecording()
        return true
    }

    /** Cancel this turn without ending the session. Native calls finish on their owning worker. */
    fun interruptResponse(resumeMicrophone: Boolean = true): Boolean {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (!isActive || currentState !in INTERRUPTIBLE_STATES) return false
        val token = generation.incrementAndGet()
        inputMode = InputMode.MICROPHONE
        micInput?.recorder?.cancelRecording()
        micInput = null
        silencePlayback()
        notifyState(State.CANCELLING)
        listener?.onOutput("")
        // This runs after the cancelled turn's finally blocks and before any new inference.
        executor.execute { releaseModels(clearHistory = false) }
        if (resumeMicrophone) emit(token) { listen(token) }
        return true
    }

    /** Continue after the UI obtains microphone permission for a cancelled test session. */
    fun resumeMicrophoneInput() {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (isActive && currentState == State.CANCELLING) {
            val token = generation.get()
            emit(token) { listen(token) }
        }
    }

    companion object {
        val INTERRUPTIBLE_STATES = setOf(State.SUBMITTING, State.RECOGNIZING, State.THINKING,
            State.SYNTHESIZING, State.SPEAKING, State.SYNTHESIZING_QUERY, State.PLAYING_QUERY)
    }

    /** Generated PCM is passed directly to Whisper; neither the microphone nor query text feeds the LLM. */
    private fun runTestQueries(token: Long) {
        for (query in config.testQueries) {
            state(token, State.SYNTHESIZING_QUERY)
            val audio = synthesize(token, query)
            state(token, State.PLAYING_QUERY)
            play(token, audio.audioData, audio.channels, audio.sampleRate)
            respond(token, audio.audioData, audio.sampleRate, audio.channels)
        }
        // Keep the latest recognized query and answer visible until the user closes the session.
        state(token, State.TEST_COMPLETE)
    }

    private fun respond(token: Long, audio: FloatArray, sampleRate: Int, channels: Int = 1) {
        val historyBeforeTurn = history.toList()
        try {
            respondToAudio(token, audio, sampleRate, channels)
        } finally {
            if (generation.get() != token) {
                history.clear()
                history.addAll(historyBeforeTurn)
            }
        }
    }

    private fun respondToAudio(token: Long, audio: FloatArray, sampleRate: Int, channels: Int) {
        state(token, State.RECOGNIZING)
        val input = try {
            check(speech.initializeSpeech(config.speechEnvId)) { "Whisperの初期化に失敗しました" }
            ensureActive(token)
            voiceAgentTranscript(speech.process(audio, channels, sampleRate))
        } finally {
            speech.releaseSpeech()
        }
        ensureActive(token)
        if (input.isBlank()) {
            check(inputMode != InputMode.SYNTHESIZED_QUERIES) { "テスト音声を認識できませんでした" }
            return
        }
        emit(token) { listener?.onInput(input); listener?.onOutput("") }
        state(token, State.THINKING)
        val response = generateResponse(token, input)
        ensureActive(token)
        emit(token) { listener?.onOutput(response) }
        state(token, State.SYNTHESIZING)
        val segments = voiceAgentSpeechSegments(response)
        check(segments.isNotEmpty()) { "読み上げる応答がありません" }
        withVoice(token) { reference ->
            var firstPlayback = true
            var synthesized = 0
            var played = 0
            voiceAgentQueueSpeech(segments, { ensureActive(token) }, { segment ->
                Log.d("VoiceAgent", "Synthesizing segment ${++synthesized}/${segments.size}: $segment")
                synthesizeInitialized(token, segment, reference)
            }, { result, checkActive ->
                if (firstPlayback) {
                    state(token, State.SPEAKING)
                    firstPlayback = false
                }
                Log.d("VoiceAgent", "Playing segment ${++played}/${segments.size}")
                play(token, result.audioData, result.channels, result.sampleRate, checkActive)
            })
        }
    }

    private fun synthesize(token: Long, text: String): AudioUtil.WavFileData =
        withVoice(token) { reference -> synthesizeInitialized(token, text, reference) }

    /** Reuse one voice model/reference throughout a response; never overlap native model calls. */
    private fun <T> withVoice(token: Long, action: (AudioUtil.WavFileData) -> T): T {
        ensureActive(token)
        val reference = AudioUtil().loadRawAudio(context.resources.openRawResource(R.raw.reference_audio_girl))
        try {
            check(voice.initializeVoice(config.voiceEnvId)) { "GPT-SoVITSの初期化に失敗しました" }
            ensureActive(token)
            return action(reference)
        } finally {
            voice.releaseVoice()
        }
    }

    private fun synthesizeInitialized(token: Long, text: String, reference: AudioUtil.WavFileData): AudioUtil.WavFileData {
        ensureActive(token)
        check(voice.textToSpeech(reference.audioData, reference.channels, reference.sampleRate,
            "水をマレーシアから買わなくてはならない。", "ja", text, "ja", autoPlay = false) >= 0) {
            "音声合成に失敗しました"
        }
        ensureActive(token)
        return AudioUtil.WavFileData(voice.lastAudioSampleRate, voice.lastAudioChannels, checkNotNull(voice.lastAudioData))
    }

    private fun generateResponse(token: Long, input: String): String {
        val model = AiliaLLM()
        llm = model
        try {
            model.openModelFile(llmModelFile.absolutePath, if (config.llmBackend == LLMBackend.QNN) 0 else 4096)
            ensureActive(token)
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
                ensureActive(token)
                complete = model.generate()
                output.append(model.getDeltaText())
                val text = output.toString()
                emit(token) { listener?.onOutput(text) }
                if (complete) break
            }
            ensureActive(token)
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

    /** Stream small buffers so stop() takes effect promptly even during long answers. */
    private fun play(token: Long, audio: FloatArray, channels: Int, rate: Int,
                     checkActive: () -> Unit = { ensureActive(token) }) {
        checkActive()
        require(audio.isNotEmpty() && channels in 1..2 && rate > 0) { "合成音声の形式が不正です" }
        val mask = if (channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO
        val minimum = AudioTrack.getMinBufferSize(rate, mask, AudioFormat.ENCODING_PCM_16BIT)
        check(minimum > 0) { "音声出力を初期化できません" }
        val track = AudioTrack(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build(),
            AudioFormat.Builder().setSampleRate(rate).setChannelMask(mask).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build(),
            maxOf(minimum, rate / 5 * channels * 2), AudioTrack.MODE_STREAM, android.media.AudioManager.AUDIO_SESSION_ID_GENERATE)
        try {
            check(track.state == AudioTrack.STATE_INITIALIZED) { "音声出力を初期化できません" }
            val pcm = ShortArray(audio.size) { (audio[it].coerceIn(-1f, 1f) * 32767).roundToInt().toShort() }
            synchronized(playbackLock) {
                checkActive()
                playbackTrack = track
                track.play()
            }
            var offset = 0
            while (offset < pcm.size) {
                checkActive()
                val count = minOf(rate / 20 * channels, pcm.size - offset)
                val written = track.write(pcm, offset, count)
                check(written > 0) { "音声出力に失敗しました ($written)" }
                offset += written
                playbackWaveform(token, audio, track.playbackHeadPosition * channels, rate, channels)
            }
            val deadline = android.os.SystemClock.elapsedRealtime() + 5000
            while (track.playbackHeadPosition < pcm.size / channels) {
                checkActive()
                check(android.os.SystemClock.elapsedRealtime() < deadline) { "音声出力が停止しました" }
                playbackWaveform(token, audio, track.playbackHeadPosition * channels, rate, channels)
                Thread.sleep(40)
            }
        } finally {
            synchronized(playbackLock) {
                if (playbackTrack === track) playbackTrack = null
                try { track.stop() } finally { track.release() }
            }
        }
    }

    private fun silencePlayback() = synchronized(playbackLock) {
        playbackTrack?.let { track ->
            runCatching { track.pause(); track.flush() }
                .onFailure { Log.w("VoiceAgent", "Failed to pause playback", it) }
        }
    }

    private fun playbackWaveform(token: Long, audio: FloatArray, offset: Int, rate: Int, channels: Int) {
        val start = offset.coerceIn(0, audio.size)
        val chunk = audio.copyOfRange(start, minOf(start + rate / 20 * channels, audio.size))
        emit(token) { listener?.onWaveform(chunk) }
    }

    fun stop(error: String? = null) {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (!isActive || stopping) return
        isActive = false
        stopping = true
        val stopToken = generation.incrementAndGet()
        micInput?.recorder?.cancelRecording()
        micInput = null
        silencePlayback()
        notifyState(State.STOPPING)
        executor.execute {
            releaseModels()
            main.post {
                if (generation.get() == stopToken) {
                    stopping = false
                    listener?.onStopped(error)
                    listener = null
                }
            }
        }
    }

    private fun releaseModels(clearHistory: Boolean = true) {
        speech.releaseSpeech()
        voice.releaseVoice()
        runCatching { llm?.destroy() }.onFailure { Log.w("VoiceAgent", "Failed to release LLM", it) }
        llm = null
        if (clearHistory) history.clear()
    }

    private fun fail(token: Long, error: Exception) {
        Log.e("VoiceAgent", "Voice session failed", error)
        emit(token) { stop(error.message ?: "音声対話に失敗しました") }
    }
    private fun ensureActive(token: Long) { if (generation.get() != token) throw CancellationException() }
    private fun notifyState(state: State) { currentState = state; listener?.onState(state) }
    private fun state(token: Long, state: State) { ensureActive(token); emit(token) { notifyState(state) } }
    private fun emit(token: Long, action: () -> Unit) { main.post { if (generation.get() == token) action() } }
}
