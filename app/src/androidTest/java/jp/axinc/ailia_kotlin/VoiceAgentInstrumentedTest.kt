package jp.axinc.ailia_kotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.sin

@RunWith(AndroidJUnit4::class)
class VoiceAgentInstrumentedTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    @Test fun setupAndSessionAreSeparateAndStopRestoresSettings() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                val menu = activity.findViewById<Spinner>(R.id.algorithmSpinner)
                assertEquals("ToolUse", menu.getItemAtPosition(MainActivity.AlgorithmType.VOICE_AGENT.ordinal - 1))
                menu.setSelection(MainActivity.AlgorithmType.VOICE_AGENT.ordinal)
            }
            instrumentation.waitForIdleSync()
            screenshot("voice-agent-setup.png")
            // Exercise the real UI transition while keeping heavyweight model work off the UI thread.
            instrumentation.uiAutomation.executeShellCommand("pm grant ${instrumentation.targetContext.packageName} android.permission.RECORD_AUDIO").use {
                FileInputStream(it.fileDescriptor).use { stream -> stream.readBytes() }
            }
            scenario.onActivity { activity ->
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
                assertFalse(activity.findViewById<View>(R.id.modelSpinner).isShown)
                val mode = activity.findViewById<RadioGroup>(R.id.voiceAgentInputMode)
                val queries = activity.findViewById<EditText>(R.id.voiceAgentTestQueries)
                assertEquals(R.id.voiceAgentMicInput, mode.checkedRadioButtonId)
                assertEquals(View.GONE, activity.findViewById<View>(R.id.voiceAgentQueryPanel).visibility)
                mode.check(R.id.voiceAgentTestInput)
                assertEquals(View.VISIBLE, activity.findViewById<View>(R.id.voiceAgentQueryPanel).visibility)
                queries.setText(" \n ")
                activity.findViewById<Button>(R.id.voiceAgentStart).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentError).isShown)
                assertFalse(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                queries.setText(R.string.voice_agent_default_queries)
                // Starting and stopping in the same UI callback also exercises cancellation of test synthesis.
                activity.findViewById<Button>(R.id.voiceAgentStart).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                assertEquals(View.GONE, activity.findViewById<View>(R.id.voiceAgentSend).visibility)
                assertFalse(activity.findViewById<View>(R.id.rootScrollView).isShown)
                assertFalse(activity.findViewById<View>(R.id.llmChatContainer).isShown)
                assertFalse(activity.findViewById<View>(R.id.processingTimeTextView).isShown)
                activity.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
            }
            waitForStartEnabled(scenario)
            scenario.onActivity { activity ->
                activity.findViewById<RadioGroup>(R.id.voiceAgentInputMode).check(R.id.voiceAgentMicInput)
                assertEquals(View.GONE, activity.findViewById<View>(R.id.voiceAgentQueryPanel).visibility)
                activity.findViewById<Button>(R.id.voiceAgentStart).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSend).isShown)
                assertFalse(activity.findViewById<Button>(R.id.voiceAgentSend).isEnabled)
                activity.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                assertFalse(activity.findViewById<View>(R.id.voiceAgentSession).isShown)
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
            }
        }
    }

    /** Capture a short real microphone buffer and submit it before the silence endpoint can fire. */
    @Test(timeout = 60_000) fun microphoneSendButtonSubmitsWithoutWaitingForVad() {
        assumeTrue("Enable explicitly on a device with cached models", InstrumentationRegistry.getArguments().getString("voiceAgentNative") == "true")
        val context = instrumentation.targetContext
        System.loadLibrary("ailia")
        val speechFile = QnnSupport.speechQnnFileName(SpeechModelType.WHISPER_LARGE_V3_TURBO)
        val llmFile = QnnSupport.llmQnnFileName(LLMModelType.GEMMA_4_E2B)
        assumeTrue(speechFile != null && llmFile != null)
        val directory = ModelDownloader.modelDirectory(context)
        assumeTrue(File(directory, speechFile!!).exists() && File(directory, llmFile!!).exists())
        instrumentation.uiAutomation.executeShellCommand("pm grant ${context.packageName} android.permission.RECORD_AUDIO").use {
            FileInputStream(it.fileDescriptor).use { stream -> stream.readBytes() }
        }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { it.findViewById<Spinner>(R.id.algorithmSpinner).setSelection(MainActivity.AlgorithmType.VOICE_AGENT.ordinal) }
            instrumentation.waitForIdleSync()
            scenario.onActivity { it.findViewById<Button>(R.id.voiceAgentStart).performClick() }
            val deadline = SystemClock.elapsedRealtime() + 15_000
            var ready = false
            while (!ready && SystemClock.elapsedRealtime() < deadline) {
                scenario.onActivity { ready = it.findViewById<Button>(R.id.voiceAgentSend).isEnabled }
                if (!ready) SystemClock.sleep(20)
            }
            assertTrue("Microphone did not become ready", ready)
            screenshot("voice-agent-microphone-send.png")
            scenario.onActivity { activity ->
                val send = activity.findViewById<Button>(R.id.voiceAgentSend)
                assertTrue(send.isShown)
                send.performClick()
                assertTrue("Switch immediately from send to cancel", send.isEnabled)
                assertEquals(activity.getString(R.string.voice_agent_interrupt), send.contentDescription)
                assertEquals("音声を送信しています", activity.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
            }
            var recognizing = false
            val recognitionDeadline = SystemClock.elapsedRealtime() + 5000
            while (!recognizing && SystemClock.elapsedRealtime() < recognitionDeadline) {
                scenario.onActivity {
                    recognizing = it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription == "音声を認識しています"
                }
                if (!recognizing) SystemClock.sleep(20)
            }
            assertTrue("Manual send did not forward the recorded audio to recognition", recognizing)
            scenario.onActivity { activity ->
                assertTrue(activity.findViewById<Button>(R.id.voiceAgentSend).isEnabled)
                activity.findViewById<Button>(R.id.voiceAgentSend).performClick()
            }
            waitForOrbState(scenario, "音声入力を待っています", 5000)
            waitForSendEnabled(scenario)
            scenario.onActivity { activity ->
                assertEquals(activity.getString(R.string.voice_agent_send), activity.findViewById<Button>(R.id.voiceAgentSend).contentDescription)
                activity.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                assertTrue(activity.findViewById<View>(R.id.voiceAgentSetup).isShown)
            }
            waitForStartEnabled(scenario, 15_000)
        }
    }

    @Test fun slimeMovesAtRestAndReactsToMicrophoneAndPlaybackAudio() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            fun frame(state: AiliaVoiceAgentSample.State?, audio: FloatArray?): Bitmap {
                val bitmap = Bitmap.createBitmap(400, 400, Bitmap.Config.ARGB_8888)
                scenario.onActivity {
                    val orb = it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb)
                    orb.layout(0, 0, 400, 400)
                    if (state != null) orb.setState(state)
                    if (audio != null) orb.setAudio(audio)
                    orb.draw(Canvas(bitmap))
                }
                return bitmap
            }
            val idle = frame(AiliaVoiceAgentSample.State.LISTENING, null)
            SystemClock.sleep(300)
            val movingIdle = frame(null, null)
            assertFalse("The character should keep moving without audio", idle.sameAs(movingIdle))
            assertEquals(Color.BLACK, idle.getPixel(0, 0))
            val audio = FloatArray(1600) { .45f * sin(it * .13f) }
            var microphone = idle
            repeat(12) {
                if (microphone !== idle) microphone.recycle()
                microphone = frame(null, audio)
                SystemClock.sleep(20)
            }
            val quietArea = whiteArea(idle)
            assertTrue("Microphone amplitude should expand the body", whiteArea(microphone) > quietArea * 1.2)
            var speaking = frame(AiliaVoiceAgentSample.State.SPEAKING, audio)
            repeat(12) {
                speaking.recycle()
                speaking = frame(null, audio)
                SystemClock.sleep(20)
            }
            assertTrue("Playback amplitude should expand the same body", whiteArea(speaking) > quietArea * 1.2)
            val directory = instrumentation.targetContext.getExternalFilesDir(null)
            for ((name, bitmap) in listOf("slime-idle.png" to idle, "slime-speaking.png" to speaking)) {
                File(directory, name).outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
            idle.recycle(); movingIdle.recycle(); microphone.recycle(); speaking.recycle()
        }
    }

    @Test(timeout = 600_000) fun responseButtonCancelsGenerationSynthesisAndPlayback() {
        assumeTrue("Enable explicitly on a device with cached models", InstrumentationRegistry.getArguments().getString("voiceAgentNative") == "true")
        val context = instrumentation.targetContext
        System.loadLibrary("ailia")
        val speech = QnnSupport.speechQnnFileName(SpeechModelType.WHISPER_LARGE_V3_TURBO)
        val llm = QnnSupport.llmQnnFileName(LLMModelType.GEMMA_4_E2B)
        assumeTrue(speech != null && llm != null)
        val directory = ModelDownloader.modelDirectory(context)
        assumeTrue(File(directory, speech!!).exists() && File(directory, llm!!).exists())
        instrumentation.uiAutomation.executeShellCommand("pm grant ${context.packageName} android.permission.RECORD_AUDIO").use {
            FileInputStream(it.fileDescriptor).use { stream -> stream.readBytes() }
        }
        val results = JSONArray()
        val stageFilter = InstrumentationRegistry.getArguments().getString("voiceAgentCancelStage")
        for ((stage, description) in listOf("thinking" to "応答を考えています", "synthesis" to "音声を合成しています", "playback" to "応答を読み上げています")) {
            if (stageFilter != null && stage != stageFilter) continue
            ActivityScenario.launch(MainActivity::class.java).use { scenario ->
                scenario.onActivity { it.findViewById<Spinner>(R.id.algorithmSpinner).setSelection(MainActivity.AlgorithmType.VOICE_AGENT.ordinal) }
                instrumentation.waitForIdleSync()
                scenario.onActivity {
                    it.findViewById<RadioGroup>(R.id.voiceAgentInputMode).check(R.id.voiceAgentTestInput)
                    it.findViewById<EditText>(R.id.voiceAgentTestQueries).setText("こんにちは。自己紹介をしてください。")
                    it.findViewById<Button>(R.id.voiceAgentStart).performClick()
                }
                waitForOrbState(scenario, description, 180_000)
                if (stage == "synthesis") SystemClock.sleep(8000) // Allow model loading to enter the blocking synthesis call.
                screenshot("voice-agent-slime-$stage.png")
                val beforeCancel = SystemClock.elapsedRealtime()
                scenario.onActivity {
                    val button = it.findViewById<Button>(R.id.voiceAgentSend)
                    assertTrue(button.isEnabled)
                    assertEquals(it.getString(R.string.voice_agent_interrupt), button.contentDescription)
                    assertEquals(description, it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
                    button.performClick()
                }
                waitForOrbState(scenario, "音声入力を待っています", 5000)
                waitForSendEnabled(scenario)
                results.put(JSONObject().put("stage", stage).put("microphone_ready_ms", SystemClock.elapsedRealtime() - beforeCancel))
                if (stage == "synthesis") {
                    // Submit the next recording while the old blocking call is finishing, then cancel that queued turn too.
                    scenario.onActivity { it.findViewById<Button>(R.id.voiceAgentSend).performClick() }
                    SystemClock.sleep(250)
                    scenario.onActivity {
                        assertEquals("音声を送信しています", it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
                        assertEquals(it.getString(R.string.voice_agent_interrupt), it.findViewById<Button>(R.id.voiceAgentSend).contentDescription)
                        it.findViewById<Button>(R.id.voiceAgentSend).performClick()
                    }
                    waitForOrbState(scenario, "音声入力を待っています", 5000)
                    waitForSendEnabled(scenario)
                }
                SystemClock.sleep(300)
                scenario.onActivity {
                    assertEquals("音声入力を待っています", it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription)
                    assertEquals("", it.findViewById<TextView>(R.id.voiceAgentOutput).text.toString())
                    assertEquals(it.getString(R.string.voice_agent_send), it.findViewById<Button>(R.id.voiceAgentSend).contentDescription)
                    it.findViewById<Button>(R.id.voiceAgentEnd).performClick()
                }
                waitForStartEnabled(scenario, 60_000)
            }
        }
        File(context.getExternalFilesDir(null), "voice-agent-cancellation-test.json").writeText(results.toString(2))
    }

    private fun whiteArea(bitmap: Bitmap): Int {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return pixels.count { Color.red(it) > 245 && Color.green(it) > 245 && Color.blue(it) > 245 }
    }

    private fun waitForSendEnabled(scenario: ActivityScenario<MainActivity>) {
        val deadline = SystemClock.elapsedRealtime() + 5000
        while (SystemClock.elapsedRealtime() < deadline) {
            var ready = false
            scenario.onActivity { ready = it.findViewById<Button>(R.id.voiceAgentSend).isEnabled }
            if (ready) return
            SystemClock.sleep(20)
        }
        fail("Microphone send did not become ready")
    }

    private fun waitForOrbState(scenario: ActivityScenario<MainActivity>, state: String, timeoutMs: Long) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        var actual = ""
        while (SystemClock.elapsedRealtime() < deadline) {
            scenario.onActivity { actual = it.findViewById<VoiceAgentOrbView>(R.id.voiceAgentOrb).contentDescription.toString() }
            if (actual == state) return
            SystemClock.sleep(20)
        }
        fail("Expected $state but was $actual")
    }

    /** Uses the public start path and real GPT-SoVITS query audio; no microphone or private hooks. */
    @Test(timeout = 600_000) fun synthesizedQueriesCompleteTwoVoiceTurns() {
        assumeTrue("Enable explicitly on a device with cached models", InstrumentationRegistry.getArguments().getString("voiceAgentNative") == "true")
        val context = instrumentation.targetContext
        System.loadLibrary("ailia")
        axip.ailia.Ailia.SetTemporaryCachePath(context.cacheDir.absolutePath)
        val qnnSpeech = QnnSupport.speechQnnFileName(SpeechModelType.WHISPER_LARGE_V3_TURBO)
        val qnnLlm = QnnSupport.llmQnnFileName(LLMModelType.GEMMA_4_E2B)
        assumeTrue(qnnSpeech != null && qnnLlm != null)
        val directory = ModelDownloader.modelDirectory(context)
        assumeTrue(File(directory, qnnSpeech!!).exists() && File(directory, qnnLlm!!).exists())
        val cpu = axip.ailia.AiliaModel.getEnvironments().first {
            it.type == axip.ailia.AiliaEnvironment.TYPE_CPU && !QnnSupport.isQnnEnvironmentName(it.name)
        }
        val executor = Executors.newSingleThreadExecutor()
        val sample = AiliaVoiceAgentSample(context, executor)
        val states = CopyOnWriteArrayList<AiliaVoiceAgentSample.State>()
        val inputs = CopyOnWriteArrayList<String>()
        val answers = CopyOnWriteArrayList<String>()
        val completed = CountDownLatch(1)
        val stopped = CountDownLatch(1)
        val queries = context.getString(R.string.voice_agent_default_queries).lines()
        var latestOutput = ""
        var failure: String? = null
        try {
            instrumentation.runOnMainSync {
                sample.start(AiliaVoiceAgentSample.Config(cpu.id, true, LLMBackend.QNN, cpu.id,
                    AiliaVoiceAgentSample.InputMode.SYNTHESIZED_QUERIES, queries),
                    object : AiliaVoiceAgentSample.Listener {
                        override fun onState(state: AiliaVoiceAgentSample.State) {
                            states += state
                            if (state == AiliaVoiceAgentSample.State.SPEAKING) answers += latestOutput
                            if (state == AiliaVoiceAgentSample.State.TEST_COMPLETE) completed.countDown()
                        }
                        override fun onInput(text: String) { inputs += text }
                        override fun onOutput(text: String) { latestOutput = text }
                        override fun onWaveform(samples: FloatArray) = Unit
                        override fun onDownload(fileName: String, downloaded: Long, total: Long) = Unit
                        override fun onStopped(error: String?) {
                            failure = error
                            completed.countDown()
                            stopped.countDown()
                        }
                    })
            }
            assertTrue("Synthesized query test timed out", completed.await(540, TimeUnit.SECONDS))
            instrumentation.waitForIdleSync()
            File(context.getExternalFilesDir(null), "voice-agent-synthesized-test.json").writeText(
                JSONObject().put("queries", JSONArray(queries)).put("recognized", JSONArray(inputs))
                    .put("answers", JSONArray(answers)).put("states", JSONArray(states.map { it.name }))
                    .put("error", failure ?: JSONObject.NULL).toString(2))
            assertNull("Voice agent failed", failure)
            assertEquals(2, inputs.size)
            assertTrue(inputs.first().contains("自己紹介"))
            assertEquals(2, answers.size)
            assertTrue(answers.all { it.isNotBlank() })
            assertTrue("Expected an answer of two: ${answers.last()}", answers.last().contains("2") || answers.last().contains("二"))
            assertFalse(states.contains(AiliaVoiceAgentSample.State.LISTENING))
            val turn = listOf(AiliaVoiceAgentSample.State.SYNTHESIZING_QUERY, AiliaVoiceAgentSample.State.PLAYING_QUERY,
                AiliaVoiceAgentSample.State.RECOGNIZING, AiliaVoiceAgentSample.State.THINKING,
                AiliaVoiceAgentSample.State.SYNTHESIZING, AiliaVoiceAgentSample.State.SPEAKING)
            assertEquals(listOf(AiliaVoiceAgentSample.State.PREPARING) + turn + turn +
                AiliaVoiceAgentSample.State.TEST_COMPLETE, states.toList())
        } finally {
            instrumentation.runOnMainSync { sample.stop() }
            assertTrue("Models were not released", stopped.await(30, TimeUnit.SECONDS))
            executor.shutdown()
        }
    }

    private fun waitForStartEnabled(scenario: ActivityScenario<MainActivity>, timeoutMs: Long = 5000) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            var enabled = false
            scenario.onActivity { enabled = it.findViewById<Button>(R.id.voiceAgentStart).isEnabled }
            if (enabled) return
            SystemClock.sleep(50)
        }
        fail("Start button did not become enabled after cancellation")
    }

    private fun screenshot(name: String) {
        instrumentation.waitForIdleSync()
        SystemClock.sleep(350) // Let ActionBar hide/show and the next layout frame finish.
        val bitmap = instrumentation.uiAutomation.takeScreenshot() ?: return
        File(instrumentation.targetContext.getExternalFilesDir(null), name).outputStream().use {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        bitmap.recycle()
    }
}
