package jp.axinc.ailia_kotlin

import android.util.Log
import axip.ailia_tokenizer.AiliaTokenizer

class AiliaTokenizerSample {
    private var tokenizer: AiliaTokenizer? = null
    private var isInitialized = false
    private var lastTokenizationResult: String = ""

    fun initializeTokenizer(tokenizerType: Int = AiliaTokenizer.AILIA_TOKENIZER_TYPE_WHISPER): Boolean {
        return try {
            if (isInitialized) {
                releaseTokenizer()
            }
            
            tokenizer = AiliaTokenizer(tokenizerType)
            isInitialized = true
            Log.i("AILIA_Main", "Tokenizer initialized successfully")
            true
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to initialize tokenizer: ${e.javaClass.name}: ${e.message}")
            releaseTokenizer()
            false
        }
    }

    fun processTokenization(text: String): Long {
        if (!isInitialized || tokenizer == null) {
            Log.e("AILIA_Error", "Tokenizer not initialized")
            return -1
        }

        return try {
            val startTime = System.nanoTime()
            
            val tokens = tokenizer!!.encode(text)
            var tokensText = ""
            for (i in tokens.indices) {
                tokensText += "${tokens[i]}"
                if (i < tokens.size - 1) tokensText += ", "
            }
            lastTokenizationResult = tokensText
            Log.i("AILIA_Main", "Tokens: $tokensText")
            
            val endTime = System.nanoTime()
            (endTime - startTime) / 1000000
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Failed to process tokenization: ${e.javaClass.name}: ${e.message}")
            -1
        }
    }

    fun releaseTokenizer() {
        try {
            tokenizer?.close()
        } catch (e: Exception) {
            Log.e("AILIA_Error", "Error releasing tokenizer: ${e.javaClass.name}: ${e.message}")
        } finally {
            tokenizer = null
            isInitialized = false
            Log.i("AILIA_Main", "Tokenizer released")
        }
    }
    
    fun getLastTokenizationResult(): String {
        return lastTokenizationResult
    }
}
