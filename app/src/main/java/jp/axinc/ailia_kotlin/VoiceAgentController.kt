package jp.axinc.ailia_kotlin

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.Drawable
import android.media.AudioManager
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import com.google.android.material.button.MaterialButton
import axip.ailia.Ailia
import axip.ailia.AiliaModel
import java.util.concurrent.ExecutorService

/** Owns the setup and minimal in-call UI; model work stays in AiliaVoiceAgentSample. */
class VoiceAgentController(
    private val activity: AppCompatActivity,
    executor: ExecutorService,
    private val requestMicrophone: () -> Unit,
    private val showSession: (Boolean) -> Unit,
) {
    private data class Environment(val name: String, val id: Int, val qnn: Boolean = false)
    private val sample = AiliaVoiceAgentSample(activity, executor)
    private val speechBackend: Spinner = activity.findViewById(R.id.voiceAgentSpeechBackend)
    private val llmBackend: Spinner = activity.findViewById(R.id.voiceAgentLlmBackend)
    private val voiceBackend: Spinner = activity.findViewById(R.id.voiceAgentVoiceBackend)
    private val inputMode: RadioGroup = activity.findViewById(R.id.voiceAgentInputMode)
    private val queryPanel: View = activity.findViewById(R.id.voiceAgentQueryPanel)
    private val testQueries: EditText = activity.findViewById(R.id.voiceAgentTestQueries)
    private val start: Button = activity.findViewById(R.id.voiceAgentStart)
    private val end: Button = activity.findViewById(R.id.voiceAgentEnd)
    private val send: MaterialButton = activity.findViewById(R.id.voiceAgentSend)
    private val error: TextView = activity.findViewById(R.id.voiceAgentError)
    private val status: TextView = activity.findViewById(R.id.voiceAgentStatus)
    private val progress: TextView = activity.findViewById(R.id.voiceAgentProgress)
    private val input: TextView = activity.findViewById(R.id.voiceAgentInput)
    private val output: TextView = activity.findViewById(R.id.voiceAgentOutput)
    private val orb: VoiceAgentOrbView = activity.findViewById(R.id.voiceAgentOrb)
    private var speechEnvironments = emptyList<Environment>()
    private var voiceEnvironments = emptyList<Environment>()
    private var llmBackends = emptyList<LLMBackend>()
    private var disposed = false
    private enum class PermissionAction { START, RESUME }
    private var permissionAction: PermissionAction? = null
    private val root: View = activity.findViewById(R.id.rootLayout)
    private var originalBackground: Drawable? = null
    private var lightStatusBars = false
    private var lightNavigationBars = false
    private var sessionShown = false
    private var actionBarWasVisible = false
    private var currentState: AiliaVoiceAgentSample.State? = null
    private val audioManager = activity.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val focusListener = AudioManager.OnAudioFocusChangeListener { focus -> if (focus < 0) activity.runOnUiThread { stop() } }
    private val back = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() = stop()
    }

    init {
        activity.onBackPressedDispatcher.addCallback(activity, back)
        start.setOnClickListener { start() }
        end.setOnClickListener { stop() }
        send.setOnClickListener {
            if (currentState == AiliaVoiceAgentSample.State.LISTENING) {
                sample.submitMicrophoneInput()
            } else if (currentState in AiliaVoiceAgentSample.INTERRUPTIBLE_STATES) {
                val granted = ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
                if (sample.interruptResponse(resumeMicrophone = granted) && !granted) {
                    permissionAction = PermissionAction.RESUME
                    requestMicrophone()
                }
            }
        }
        inputMode.setOnCheckedChangeListener { _, checkedId ->
            queryPanel.visibility = if (checkedId == R.id.voiceAgentTestInput) View.VISIBLE else View.GONE
        }
        try {
            Ailia.SetTemporaryCachePath(activity.cacheDir.absolutePath)
            val environments = AiliaModel.getEnvironments().filter { isSelectableAiliaEnvironment(it.name) }
            val cpu = environments.filter { !QnnSupport.isQnnEnvironmentName(it.name) }
            check(cpu.isNotEmpty()) { "音声認識に使用できるバックエンドがありません" }
            val cpuIndex = preferredBlasThenCpuEnvironmentIndex(cpu.map { it.type })
            speechEnvironments = cpu.map { Environment(it.name, it.id) } +
                if (QnnSupport.speechQnnFileName(SpeechModelType.WHISPER_LARGE_V3_TURBO) != null)
                    listOf(Environment("QNN (NPU) ${QnnSupport.socName}", cpu[cpuIndex].id, true)) else emptyList()
            voiceEnvironments = environments.filter {
                !QnnSupport.isQnnEnvironmentName(it.name) || QnnSupport.isFp16Supported()
            }.map { Environment(it.name, it.id) }
            llmBackends = LLMBackend.values().filter {
                it != LLMBackend.QNN || QnnSupport.isLLMQnnAvailable(LLMModelType.GEMMA_4_E2B)
            }
            configure(speechBackend, speechEnvironments.map { it.name },
                speechEnvironments.indexOfFirst { it.qnn }.takeIf { it >= 0 } ?: cpuIndex)
            configure(llmBackend, llmBackends.map { it.displayName }, llmBackends.indexOf(LLMBackend.QNN).coerceAtLeast(0))
            val voiceDefault = voiceEnvironments.indexOfFirst { it.id == cpu[cpuIndex].id }.coerceAtLeast(0)
            configure(voiceBackend, voiceEnvironments.map { it.name }, voiceDefault)
        } catch (e: Exception) {
            showError(e.message ?: "バックエンドの取得に失敗しました")
            start.isEnabled = false
        }
    }

    private fun configure(spinner: Spinner, names: List<String>, selection: Int) {
        spinner.adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_item, names).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinner.setSelection(selection)
    }

    @Suppress("DEPRECATION")
    fun start() {
        if (disposed || !start.isEnabled || sample.isActive) return
        val mode = if (inputMode.checkedRadioButtonId == R.id.voiceAgentTestInput)
            AiliaVoiceAgentSample.InputMode.SYNTHESIZED_QUERIES else AiliaVoiceAgentSample.InputMode.MICROPHONE
        val queries = testQueries.text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.toList()
        if (mode == AiliaVoiceAgentSample.InputMode.SYNTHESIZED_QUERIES && queries.isEmpty()) {
            showError(activity.getString(R.string.voice_agent_query_required))
            return
        }
        if (mode == AiliaVoiceAgentSample.InputMode.MICROPHONE &&
            ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            permissionAction = PermissionAction.START
            requestMicrophone()
            return
        }
        if (audioManager.requestAudioFocus(focusListener, AudioManager.STREAM_MUSIC, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            showError("音声入出力を開始できません。他の音声アプリの終了後にお試しください")
            return
        }
        val speech = speechEnvironments[speechBackend.selectedItemPosition]
        val config = AiliaVoiceAgentSample.Config(speech.id, speech.qnn,
            llmBackends[llmBackend.selectedItemPosition], voiceEnvironments[voiceBackend.selectedItemPosition].id,
            inputMode = mode, testQueries = queries)
        error.visibility = View.GONE
        input.text = ""
        output.text = ""
        end.contentDescription = activity.getString(R.string.voice_agent_end)
        send.visibility = if (mode == AiliaVoiceAgentSample.InputMode.MICROPHONE) View.VISIBLE else View.GONE
        send.isEnabled = false
        send.setIconResource(R.drawable.ic_send)
        send.contentDescription = activity.getString(R.string.voice_agent_send)
        progress.text = "モデルを準備しています…"
        progress.visibility = View.VISIBLE
        actionBarWasVisible = activity.supportActionBar?.isShowing == true
        originalBackground = root.background
        root.setBackgroundColor(Color.BLACK)
        val bars = WindowCompat.getInsetsController(activity.window, root)
        lightStatusBars = bars.isAppearanceLightStatusBars
        lightNavigationBars = bars.isAppearanceLightNavigationBars
        bars.isAppearanceLightStatusBars = false
        bars.isAppearanceLightNavigationBars = false
        sessionShown = true
        activity.supportActionBar?.hide()
        activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        start.isEnabled = false
        back.isEnabled = true
        testQueries.clearFocus()
        (activity.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
            .hideSoftInputFromWindow(testQueries.windowToken, 0)
        showSession(true)
        sample.start(config, object : AiliaVoiceAgentSample.Listener {
            override fun onState(state: AiliaVoiceAgentSample.State) {
                if (disposed) return
                currentState = state
                status.setText(when (state) {
                    AiliaVoiceAgentSample.State.PREPARING -> R.string.voice_agent_status_preparing
                    AiliaVoiceAgentSample.State.LISTENING,
                    AiliaVoiceAgentSample.State.WAITING_QUERY -> R.string.voice_agent_status_listening
                    AiliaVoiceAgentSample.State.SUBMITTING,
                    AiliaVoiceAgentSample.State.RECOGNIZING -> R.string.voice_agent_status_recognizing
                    AiliaVoiceAgentSample.State.THINKING -> R.string.voice_agent_status_thinking
                    AiliaVoiceAgentSample.State.SYNTHESIZING_QUERY,
                    AiliaVoiceAgentSample.State.SYNTHESIZING -> R.string.voice_agent_status_synthesizing
                    AiliaVoiceAgentSample.State.PLAYING_QUERY,
                    AiliaVoiceAgentSample.State.SPEAKING -> R.string.voice_agent_status_speaking
                    AiliaVoiceAgentSample.State.CANCELLING -> R.string.voice_agent_status_cancelling
                    AiliaVoiceAgentSample.State.TEST_COMPLETE -> R.string.voice_agent_status_complete
                    AiliaVoiceAgentSample.State.STOPPING -> R.string.voice_agent_status_stopping
                })
                send.isEnabled = false
                send.alpha = .35f
                when {
                    state == AiliaVoiceAgentSample.State.LISTENING -> {
                        send.visibility = View.VISIBLE
                        send.setIconResource(R.drawable.ic_send)
                        send.contentDescription = activity.getString(R.string.voice_agent_send)
                    }
                    state in AiliaVoiceAgentSample.INTERRUPTIBLE_STATES -> {
                        send.visibility = View.VISIBLE
                        send.setIconResource(R.drawable.ic_stop)
                        send.contentDescription = activity.getString(R.string.voice_agent_interrupt)
                        send.isEnabled = true
                        send.alpha = 1f
                    }
                    state == AiliaVoiceAgentSample.State.TEST_COMPLETE -> send.visibility = View.GONE
                }
                orb.setState(state)
                if (state == AiliaVoiceAgentSample.State.TEST_COMPLETE) {
                    end.contentDescription = activity.getString(R.string.voice_agent_test_complete_end)
                }
                progress.visibility = if (state == AiliaVoiceAgentSample.State.PREPARING) View.VISIBLE else View.GONE
            }
            override fun onInput(text: String) {
                if (!disposed) { input.text = text; input.contentDescription = "音声入力: $text" }
            }
            override fun onOutput(text: String) {
                if (!disposed) { output.text = text; output.contentDescription = "音声出力: $text" }
            }
            override fun onWaveform(samples: FloatArray) {
                if (!disposed) {
                    orb.setAudio(samples)
                    if (currentState == AiliaVoiceAgentSample.State.LISTENING) {
                        send.isEnabled = true
                        send.alpha = 1f
                    }
                }
            }
            override fun onDownload(fileName: String, downloaded: Long, total: Long) {
                if (!disposed) progress.text = formatDownloadProgress(fileName, downloaded, total)
            }
            override fun onStopped(error: String?) {
                restoreSetup()
                if (disposed) return
                start.isEnabled = true
                start.text = "開始"
                if (error != null) showError(error)
            }
        })
    }

    fun onMicrophonePermission(granted: Boolean) {
        val action = permissionAction ?: return
        permissionAction = null
        if (granted) {
            when (action) {
                PermissionAction.START -> start()
                PermissionAction.RESUME -> sample.resumeMicrophoneInput()
            }
        } else if (action == PermissionAction.RESUME) {
            stop()
            showError("マイクを許可すると音声入力を再開できます")
        } else showError("マイクを許可すると音声対話を開始できます")
    }

    fun stop() {
        permissionAction = null
        if (!sample.isActive) return
        sample.stop()
        restoreSetup()
        if (!disposed) start.text = "終了処理中…"
    }

    @Suppress("DEPRECATION")
    private fun restoreSetup() {
        audioManager.abandonAudioFocus(focusListener)
        activity.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        back.isEnabled = false
        if (!disposed) {
            if (sessionShown) {
                root.background = originalBackground
                val bars = WindowCompat.getInsetsController(activity.window, root)
                bars.isAppearanceLightStatusBars = lightStatusBars
                bars.isAppearanceLightNavigationBars = lightNavigationBars
                sessionShown = false
            }
            if (actionBarWasVisible) activity.supportActionBar?.show()
            showSession(false)
        }
    }

    fun close() { stop(); disposed = true }
    private fun showError(text: String) { error.text = text; error.visibility = View.VISIBLE }
}
