package jp.axinc.ailia_kotlin

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CancellationException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/** Keep punctuation and closing quotes with the preceding phrase; do not split decimal numbers. */
internal fun voiceAgentSpeechSegments(text: String): List<String> {
    val segments = mutableListOf<String>()
    val pending = StringBuilder()
    val punctuation = "。、！？!?.,，．"
    val closings = "」』）)]】〉》”’\"'"
    var boundary = false
    fun flush() {
        val phrase = pending.toString().trim()
        if (phrase.any { it.isLetterOrDigit() }) segments += phrase
        else if (phrase.isNotEmpty() && segments.isNotEmpty()) segments[segments.lastIndex] += phrase
        pending.setLength(0)
        boundary = false
    }
    for ((index, char) in text.withIndex()) {
        val numericSeparator = char in ".,，．" && index > 0 && index + 1 < text.length &&
            text[index - 1].isDigit() && text[index + 1].isDigit()
        val separator = char in punctuation && !numericSeparator
        if (char == '\n' || char == '\r') {
            flush()
            continue
        }
        if (boundary && !char.isWhitespace() && !separator && char !in closings) flush()
        pending.append(char)
        if (separator) boundary = true
    }
    flush()
    return segments
}

/**
 * Synthesis stays on the caller's model thread; only PCM playback runs on a second thread.
 * At most two completed clips wait ahead of playback. Both sides observe failures/cancellation,
 * and this call always joins playback before another turn can start.
 */
internal fun <T : Any> voiceAgentQueueSpeech(
    segments: List<String>,
    checkActive: () -> Unit,
    synthesize: (String) -> T,
    play: (T, checkActive: () -> Unit) -> Unit,
) {
    val queue = ArrayBlockingQueue<T>(2)
    val producerDone = AtomicBoolean(false)
    val stopped = AtomicBoolean(false)
    val failure = AtomicReference<Throwable?>(null)
    fun checkRunning() {
        checkActive()
        failure.get()?.let { throw it }
        if (stopped.get()) throw CancellationException()
    }
    val playback = Thread({
        try {
            while (true) {
                checkRunning()
                val audio = queue.poll(50, TimeUnit.MILLISECONDS)
                if (audio != null) {
                    checkRunning()
                    play(audio, ::checkRunning)
                } else if (producerDone.get() && queue.isEmpty()) break
            }
        } catch (error: Throwable) {
            if (!stopped.get()) failure.compareAndSet(null, error)
        }
    }, "VoiceAgentPlayback")
    playback.start()
    try {
        for (segment in segments) {
            checkRunning()
            val audio = synthesize(segment)
            checkRunning()
            while (!queue.offer(audio, 50, TimeUnit.MILLISECONDS)) checkRunning()
        }
        producerDone.set(true)
        while (playback.isAlive) {
            checkRunning()
            playback.join(50)
        }
        checkRunning()
    } finally {
        stopped.set(true)
        playback.interrupt()
        // Join even if the caller was interrupted: never let stale PCM escape into the next turn.
        var interrupted = false
        while (playback.isAlive) {
            try { playback.join() } catch (_: InterruptedException) { interrupted = true }
        }
        queue.clear()
        if (interrupted) Thread.currentThread().interrupt()
    }
}
