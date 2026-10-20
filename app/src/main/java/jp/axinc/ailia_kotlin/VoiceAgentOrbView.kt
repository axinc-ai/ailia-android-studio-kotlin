package jp.axinc.ailia_kotlin

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** An abstract white orb. Microphone and playback PCM drive the same soft shape animation. */
class VoiceAgentOrbView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val body = Path()
    private val pointsX = FloatArray(48)
    private val pointsY = FloatArray(48)
    private val levels = FloatArray(12)
    private var state = AiliaVoiceAgentSample.State.PREPARING
    private var targetEnergy = 0f
    private var energy = 0f
    private var lastAudioMs = 0L
    private var lastFrameMs = 0L
    private val startedMs = SystemClock.uptimeMillis()

    init { importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_YES }

    fun setState(value: AiliaVoiceAgentSample.State) {
        state = value
        targetEnergy = 0f
        levels.fill(0f)
        contentDescription = when (value) {
            AiliaVoiceAgentSample.State.PREPARING -> "モデルを準備しています"
            AiliaVoiceAgentSample.State.LISTENING -> "音声入力を待っています"
            AiliaVoiceAgentSample.State.SYNTHESIZING_QUERY -> "テストクエリの音声を合成しています"
            AiliaVoiceAgentSample.State.PLAYING_QUERY -> "テストクエリの音声を再生しています"
            AiliaVoiceAgentSample.State.SUBMITTING -> "音声を送信しています"
            AiliaVoiceAgentSample.State.RECOGNIZING -> "音声を認識しています"
            AiliaVoiceAgentSample.State.THINKING -> "応答を考えています"
            AiliaVoiceAgentSample.State.SYNTHESIZING -> "音声を合成しています"
            AiliaVoiceAgentSample.State.SPEAKING -> "応答を読み上げています"
            AiliaVoiceAgentSample.State.CANCELLING -> "応答を中止しています"
            AiliaVoiceAgentSample.State.TEST_COMPLETE -> "テスト完了"
            AiliaVoiceAgentSample.State.STOPPING -> "対話を終了しています"
        }
        invalidate()
    }

    fun setAudio(samples: FloatArray) {
        val rms = VoiceAgentEndpoint.rms(samples)
        targetEnergy = (sqrt((rms - .008f).coerceAtLeast(0f)) * 2.2f).coerceAtMost(1f)
        for (i in levels.indices) {
            val start = i * samples.size / levels.size
            val end = (i + 1) * samples.size / levels.size
            var sum = 0f
            for (j in start until end) sum += samples[j] * samples[j]
            val value = if (end > start) sqrt(sum / (end - start)) else 0f
            levels[i] = levels[i] * .4f + (value * 6f).coerceAtMost(1f) * .6f
        }
        lastAudioMs = SystemClock.uptimeMillis()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK)
        val now = SystemClock.uptimeMillis()
        val dt = if (lastFrameMs == 0L) .016f else ((now - lastFrameMs) / 1000f).coerceIn(.001f, .05f)
        lastFrameMs = now
        if (now - lastAudioMs > 180) {
            targetEnergy = 0f
            val decay = exp(-dt / .24f)
            for (i in levels.indices) levels[i] *= decay
        }
        val smoothing = 1f - exp(-dt / if (targetEnergy > energy) .07f else .24f)
        energy += (targetEnergy - energy) * smoothing
        val time = (now - startedMs) / 1000f
        val busy = when (state) {
            AiliaVoiceAgentSample.State.PREPARING, AiliaVoiceAgentSample.State.RECOGNIZING,
            AiliaVoiceAgentSample.State.THINKING, AiliaVoiceAgentSample.State.SYNTHESIZING,
            AiliaVoiceAgentSample.State.SYNTHESIZING_QUERY, AiliaVoiceAgentSample.State.CANCELLING -> true
            else -> false
        }
        val radius = min(width, height) * .30f * (1f + .22f * energy + .018f * sin(time * 1.4f))
        val cx = width / 2f + radius * .022f * sin(time * .9f)
        val cy = height / 2f + radius * .035f * sin(time * 1.1f)
        val stretchX = 1f + .035f * sin(time * 1.2f) + .06f * energy * sin(time * 4.1f)
        val stretchY = 1f - .035f * sin(time * 1.2f) + .05f * energy * cos(time * 3.7f)
        val speed = if (busy) 1.8 else 1.0
        for (i in pointsX.indices) {
            val angle = i * PI * 2 / pointsX.size
            val bandPosition = i.toFloat() * levels.size / pointsX.size
            val bandIndex = bandPosition.toInt()
            val fraction = bandPosition - bandIndex
            val band = levels[bandIndex] * (1f - fraction) + levels[(bandIndex + 1) % levels.size] * fraction
            val wobble = .028 * sin(angle * 3 + time * speed) + .018 * cos(angle * 5 - time * .85) +
                (if (busy) .022 else .009) * sin(angle * 2 - time * 1.7) +
                energy * (.055 * sin(angle * 4 + time * 4.3) + .035 * band)
            val r = radius * (1 + wobble)
            pointsX[i] = (cx + cos(angle) * r * stretchX).toFloat()
            pointsY[i] = (cy + sin(angle) * r * stretchY).toFloat()
        }
        drawBody(canvas)

        if (isShown && windowVisibility == VISIBLE) postInvalidateOnAnimation()
    }

    /** Closed Catmull-Rom spline keeps the outline soft, without allocating objects per frame. */
    private fun drawBody(canvas: Canvas) {
        body.reset()
        body.moveTo(pointsX[0], pointsY[0])
        val count = pointsX.size
        for (i in 0 until count) {
            val previous = (i + count - 1) % count
            val next = (i + 1) % count
            val following = (i + 2) % count
            body.cubicTo(
                pointsX[i] + (pointsX[next] - pointsX[previous]) / 6f,
                pointsY[i] + (pointsY[next] - pointsY[previous]) / 6f,
                pointsX[next] - (pointsX[following] - pointsX[i]) / 6f,
                pointsY[next] - (pointsY[following] - pointsY[i]) / 6f,
                pointsX[next], pointsY[next],
            )
        }
        body.close()
        paint.color = Color.WHITE
        canvas.drawPath(body, paint)
    }

    override fun onVisibilityChanged(changedView: View, visibility: Int) {
        super.onVisibilityChanged(changedView, visibility)
        if (visibility == VISIBLE) invalidate()
    }
}
