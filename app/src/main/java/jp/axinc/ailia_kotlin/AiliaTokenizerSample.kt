package jp.axinc.ailia_kotlin

import android.util.Log

import axip.ailia_tokenizer.AiliaTokenizer

class AiliaTokenizerSample {
    fun ailia_tokenize(): Boolean {
        val tokenizer: axip.ailia_tokenizer.AiliaTokenizer = axip.ailia_tokenizer.AiliaTokenizer(tokenizerType = axip.ailia_tokenizer.AiliaTokenizer.AILIA_TOKENIZER_TYPE_WHISPER)
        //tokenizer.loadFiles(dictionaryPath = dictionaryPath, vocabPath = vocabPath)
        var tokens = tokenizer.encode("Hello world.")
        var tokens_text = "Tokens : "
        for (i in 0 until tokens.indices.count()){
            tokens_text += tokens[i].toString() + " , "
        }
        Log.i("AILIA_Main", tokens_text)
        tokenizer.close()
        return true;
    }
}