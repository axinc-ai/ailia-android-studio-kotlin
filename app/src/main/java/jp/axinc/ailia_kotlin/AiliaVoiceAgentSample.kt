package jp.axinc.ailia_kotlin

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Orchestrates microphone turns and UI callbacks for Whisper → Gemma → GPT-SoVITS.
 * VoiceAgentModels owns inference/history; VoiceAgentPlayer owns PCM output.
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
    private val models = VoiceAgentModels(this.context)
    private val player = VoiceAgentPlayer()
    private var listener: Listener? = null
    private var currentState = State.STOPPING
    @Volatile private var inputMode = InputMode.MICROPHONE
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
                models.prepare(config, { ensureActive(token) }) { fileName, downloaded, total ->
                    emit(token) { listener?.onDownload(fileName, downloaded, total) }
                }
                when (config.inputMode) {
                    InputMode.MICROPHONE -> emit(token) { listen(token) }
                    InputMode.SYNTHESIZED_QUERIES -> runTestQueries(token, config.testQueries)
                }
            } catch (_: CancellationException) {
                // stop() already queued resource release on this executor.
            } catch (e: Exception) {
                fail(token, e)
            }
        }
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
                        respond(token, AudioUtil.WavFileData(sampleRate, 1, endpoint.trim(audio)))
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
        player.silence()
        notifyState(State.CANCELLING)
        listener?.onOutput("")
        // This runs after the cancelled turn's finally blocks and before any new inference.
        executor.execute { models.release(clearHistory = false) }
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
    private fun runTestQueries(token: Long, queries: List<String>) {
        for (query in queries) {
            state(token, State.SYNTHESIZING_QUERY)
            val audio = models.synthesize(query) { ensureActive(token) }
            state(token, State.PLAYING_QUERY)
            play(token, audio)
            respond(token, audio)
        }
        // Keep the latest recognized query and answer visible until the user closes the session.
        state(token, State.TEST_COMPLETE)
    }

    private fun respond(token: Long, audio: AudioUtil.WavFileData) {
        models.withConversationTurn(isCancelled = { generation.get() != token }) {
            respondToAudio(token, audio)
        }
    }

    private fun respondToAudio(token: Long, audio: AudioUtil.WavFileData) {
        state(token, State.RECOGNIZING)
        val input = models.recognize(audio) { ensureActive(token) }
        if (input.isBlank()) {
            check(inputMode != InputMode.SYNTHESIZED_QUERIES) { "テスト音声を認識できませんでした" }
            return
        }
        emit(token) { listener?.onInput(input); listener?.onOutput("") }
        state(token, State.THINKING)
        val response = models.generateResponse(input, { ensureActive(token) }) { text ->
            emit(token) { listener?.onOutput(text) }
        }
        ensureActive(token)
        emit(token) { listener?.onOutput(response) }
        speakResponse(token, response)
    }

    private fun speakResponse(token: Long, response: String) {
        state(token, State.SYNTHESIZING)
        val segments = voiceAgentSpeechSegments(response)
        check(segments.isNotEmpty()) { "読み上げる応答がありません" }
        models.withVoice({ ensureActive(token) }) { synthesize ->
            var firstPlayback = true
            var synthesized = 0
            var played = 0
            voiceAgentQueueSpeech(segments, { ensureActive(token) }, { segment ->
                Log.d("VoiceAgent", "Synthesizing segment ${++synthesized}/${segments.size}: $segment")
                synthesize(segment)
            }, { result, checkActive ->
                if (firstPlayback) {
                    state(token, State.SPEAKING)
                    firstPlayback = false
                }
                Log.d("VoiceAgent", "Playing segment ${++played}/${segments.size}")
                play(token, result, checkActive)
            })
        }
    }

    private fun play(token: Long, audio: AudioUtil.WavFileData,
                     checkActive: () -> Unit = { ensureActive(token) }) {
        player.play(audio, checkActive) { chunk -> emit(token) { listener?.onWaveform(chunk) } }
    }

    fun stop(error: String? = null) {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (!isActive || stopping) return
        isActive = false
        stopping = true
        val stopToken = generation.incrementAndGet()
        micInput?.recorder?.cancelRecording()
        micInput = null
        player.silence()
        notifyState(State.STOPPING)
        executor.execute {
            models.release()
            main.post {
                if (generation.get() == stopToken) {
                    stopping = false
                    listener?.onStopped(error)
                    listener = null
                }
            }
        }
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
