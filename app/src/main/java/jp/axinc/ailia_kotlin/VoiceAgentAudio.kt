package jp.axinc.ailia_kotlin

import kotlin.math.sqrt

/** Lightweight microphone endpointing; Whisper also applies its Silero VAD before recognition. */
internal class VoiceAgentEndpoint {
    private var samples = 0
    private var voicedSamples = 0
    private var silentSamples = 0
    private var firstVoiceSample = -1
    val hasSpeech: Boolean get() = voicedSamples >= SAMPLE_RATE / 4

    fun accept(chunk: FloatArray): Boolean {
        val voiced = rms(chunk) >= 0.012f
        if (voiced) {
            if (firstVoiceSample < 0) firstVoiceSample = samples
            voicedSamples += chunk.size
            silentSamples = 0
        } else {
            silentSamples += chunk.size
        }
        samples += chunk.size
        return hasSpeech && silentSamples >= SAMPLE_RATE
    }

    fun trim(audio: FloatArray): FloatArray =
        audio.copyOfRange((firstVoiceSample - SAMPLE_RATE / 3).coerceIn(0, audio.size), audio.size)

    companion object {
        const val SAMPLE_RATE = 16_000
        fun rms(samples: FloatArray): Float =
            if (samples.isEmpty()) 0f else sqrt(samples.sumOf { (it * it).toDouble() } / samples.size).toFloat()
    }
}

/** Only remove the timestamp supplied by AiliaSpeechSample; preserve numbers in the utterance. */
internal fun voiceAgentTranscript(lines: List<String>): String = lines.joinToString(" ") {
    it.replaceFirst(Regex("^\\[\\d+:\\d+ - \\d+:\\d+\\]\\s*"), "").trim()
}.trim()
