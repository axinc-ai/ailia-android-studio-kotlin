package jp.axinc.ailia_kotlin

import org.junit.Assert.*
import org.junit.Test

class VoiceAgentAudioTest {
    private val silence = FloatArray(1600)
    private val speech = FloatArray(1600) { if (it % 2 == 0) .1f else -.1f }

    @Test fun silenceDoesNotSubmitEvenAfterThirtySeconds() {
        val endpoint = VoiceAgentEndpoint()
        repeat(300) { assertFalse(endpoint.accept(silence)) }
        assertFalse(endpoint.hasSpeech)
    }

    @Test fun isolatedNoiseDoesNotSubmit() {
        val endpoint = VoiceAgentEndpoint()
        endpoint.accept(speech)
        repeat(20) { assertFalse(endpoint.accept(silence)) }
        assertFalse(endpoint.hasSpeech)
    }

    @Test fun submitsAfterOneSecondOfSilenceAndKeepsShortPausesInsideTurn() {
        val endpoint = VoiceAgentEndpoint()
        repeat(3) { assertFalse(endpoint.accept(speech)) }
        repeat(9) { assertFalse(endpoint.accept(silence)) }
        assertFalse(endpoint.accept(speech))
        repeat(9) { assertFalse(endpoint.accept(silence)) }
        assertTrue(endpoint.accept(silence))
    }

    @Test fun trimsLongWaitButKeepsPreroll() {
        val endpoint = VoiceAgentEndpoint()
        repeat(100) { endpoint.accept(silence) }
        repeat(3) { endpoint.accept(speech) }
        val audio = FloatArray(164800) { it.toFloat() }
        val trimmed = endpoint.trim(audio)
        assertEquals((160000 - 16000 / 3).toFloat(), trimmed.first())
        assertEquals(audio.last(), trimmed.last())
    }

    @Test fun removesOnlySpeechTimestamps() {
        assertEquals("10時に会いましょう。 [1, 2]を足して。", voiceAgentTranscript(listOf(
            "[00:00 - 00:03] 10時に会いましょう。", "[00:03 - 00:08] [1, 2]を足して。")))
        assertEquals("", voiceAgentTranscript(emptyList()))
    }
}
