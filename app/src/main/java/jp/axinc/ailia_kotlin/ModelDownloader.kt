package jp.axinc.ailia_kotlin

import android.content.Context
import android.util.Base64
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.Locale
import java.util.Properties

/**
 * A model file and the information required to validate it.
 *
 * [expectedSize] and [sha256] are optional because not all hosted models expose
 * immutable release metadata. Files downloaded by this class still receive a
 * sidecar containing the server length and computed SHA-256, so subsequent
 * launches can detect local corruption without accessing the network.
 */
data class ModelFileSpec(
    val url: String,
    val fileName: String,
    val expectedSize: Long? = null,
    val sha256: String? = null,
)

/** Robust, shared downloader used by every sample in this application. */
object ModelDownloader {
    private const val TAG = "ModelDownloader"
    private const val BASE_URL = "https://storage.googleapis.com/ailia-models"
    private const val BUFFER_SIZE = 64 * 1024

    /** 進捗通知の間隔。数GBのモデルで通知がログ/UIを圧迫しないように間引く。 */
    private const val PROGRESS_STEP_BYTES = 4L * 1024 * 1024

    const val GEMMA_2_MODEL_URL = "$BASE_URL/gemma/gemma-2-2b-it-Q4_K_M.gguf"
    const val SAMPLE_IMAGE_URL = "$BASE_URL/misc/sample_image.jpg"

    interface DownloadListener {
        fun onProgress(bytesDownloaded: Long, totalBytes: Long)
        fun onComplete(file: File)
        fun onError(error: String)
    }

    /** Durable app-specific storage. Android does not evict it as a cache entry. */
    fun modelDirectory(context: Context): File = (context.getExternalFilesDir(null) ?: context.filesDir).apply {
        check(exists() || mkdirs()) { "Failed to create model directory: $absolutePath" }
    }

    fun downloadLLMModel(
        context: Context,
        fileName: String,
        listener: DownloadListener? = null,
    ): File? {
        val directory = modelDirectory(context)
        migrateLegacyCache(context, directory, fileName)
        return downloadFile(
            directory,
            ModelFileSpec("$BASE_URL/gemma/$fileName", fileName),
            listener,
        )
    }

    /** listenerへ中継しつつ、ダウンロード失敗の理由を[onError]でも受け取れるようにする。 */
    fun recordingErrors(listener: DownloadListener?, onError: (String) -> Unit): DownloadListener =
        object : DownloadListener {
            override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
                listener?.onProgress(bytesDownloaded, totalBytes)
            }

            override fun onComplete(file: File) {
                listener?.onComplete(file)
            }

            override fun onError(error: String) {
                onError.invoke(error)
                listener?.onError(error)
            }
        }

    fun downloadFile(
        directory: File,
        spec: ModelFileSpec,
        listener: DownloadListener? = null,
    ): File? {
        val file = File(directory, spec.fileName)
        return try {
            check(directory.exists() || directory.mkdirs()) {
                "Failed to create model directory: ${directory.absolutePath}"
            }
            val parent = file.parentFile ?: directory
            check(parent.exists() || parent.mkdirs()) {
                "Failed to create model subdirectory: ${parent.absolutePath}"
            }

            if (isValid(file, spec)) {
                Log.i(TAG, "Using validated model: ${file.absolutePath}")
                listener?.onComplete(file)
                return file
            }
            deleteInvalidArtifacts(file)

            // 途中のファイルから再開すると、その間にサーバー側のファイルが差し替わった場合や
            // 配信キャッシュに古い内容が残っている場合に新旧が混ざるため、常に最初から取り直す
            val tempFile = File(directory, "${spec.fileName}.download")
            val connection = (URL(spec.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 120_000
                requestMethod = "GET"
                instanceFollowRedirects = true
                connect()
            }
            try {
                if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                    error("HTTP ${connection.responseCode} for ${spec.url}")
                }
                val responseSize = connection.getHeaderField("Content-Length")?.toLongOrNull()
                    ?.takeIf { it >= 0 }
                val requiredSize = spec.expectedSize ?: responseSize
                // Google Cloud Storageはファイル全体のMD5を x-goog-hash で返す
                val expectedMd5 = googMd5(connection.headerFields)

                val sha256 = MessageDigest.getInstance("SHA-256")
                val md5 = MessageDigest.getInstance("MD5")
                var downloaded = 0L
                var lastReported = -1L
                connection.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            if (count == 0) continue
                            output.write(buffer, 0, count)
                            sha256.update(buffer, 0, count)
                            md5.update(buffer, 0, count)
                            downloaded += count
                            if (downloaded - lastReported >= PROGRESS_STEP_BYTES) {
                                listener?.onProgress(downloaded, requiredSize ?: -1L)
                                lastReported = downloaded
                            }
                        }
                        output.fd.sync()
                    }
                }
                listener?.onProgress(downloaded, requiredSize ?: -1L)

                check(downloaded > 0) { "Downloaded file is empty: ${spec.fileName}" }
                check(requiredSize == null || downloaded == requiredSize) {
                    "Size mismatch for ${spec.fileName}: expected=$requiredSize actual=$downloaded"
                }
                val actualMd5 = md5.digest().toHex()
                check(expectedMd5 == null || actualMd5.equals(expectedMd5, ignoreCase = true)) {
                    "MD5 mismatch for ${spec.fileName}: expected=$expectedMd5 actual=$actualMd5"
                }
                if (expectedMd5 == null) Log.w(TAG, "No MD5 from server, skipped MD5 check: ${spec.fileName}")
                val actualSha256 = sha256.digest().toHex()
                check(spec.sha256 == null || actualSha256.equals(spec.sha256, ignoreCase = true)) {
                    "SHA-256 mismatch for ${spec.fileName}"
                }

                moveChecked(tempFile, file)
                writeMetadata(file, spec.url, downloaded, actualSha256)
                check(isValid(file, spec)) { "Downloaded model validation failed: ${spec.fileName}" }
                listener?.onComplete(file)
                file
            } finally {
                connection.disconnect()
            }
        } catch (e: Exception) {
            // 次回は最初から取り直すため、一時ファイルも残さない
            deleteInvalidArtifacts(file)
            val error = "Download failed for ${spec.fileName}: ${e.message}"
            Log.e(TAG, error, e)
            listener?.onError(error)
            null
        }
    }

    /** x-goog-hash ヘッダ("crc32c=...,md5=..."、複数行の場合あり)からMD5を16進文字列で取り出す。 */
    private fun googMd5(headers: Map<String?, List<String>>): String? {
        val values = headers.entries
            .filter { it.key.equals("x-goog-hash", ignoreCase = true) }
            .flatMap { it.value }
            .flatMap { it.split(",") }
        val encoded = values.map(String::trim).firstOrNull { it.startsWith("md5=") }?.removePrefix("md5=")
            ?: return null
        return try {
            Base64.decode(encoded, Base64.NO_WRAP).toHex()
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "Invalid MD5 in x-goog-hash: $encoded")
            null
        }
    }

    fun downloadFile(
        directory: File,
        spec: ModelFileSpec,
        listener: ModelDownloadListener?,
    ): File? = downloadFile(directory, spec, object : DownloadListener {
        override fun onProgress(bytesDownloaded: Long, totalBytes: Long) {
            listener?.onProgress(spec.fileName, bytesDownloaded, totalBytes)
        }

        override fun onComplete(file: File) = Unit

        override fun onError(error: String) {
            listener?.onError(error)
        }
    })

    fun isDownloaded(directory: File, spec: ModelFileSpec): Boolean = isValid(File(directory, spec.fileName), spec)

    fun isLLMModelDownloaded(context: Context, fileName: String): Boolean = isDownloaded(
        modelDirectory(context),
        ModelFileSpec("$BASE_URL/gemma/$fileName", fileName),
    )

    fun downloadGemma2Model(context: Context, listener: DownloadListener? = null): File? =
        downloadLLMModel(context, "gemma-2-2b-it-Q4_K_M.gguf", listener)

    fun downloadSampleImage(context: Context, listener: DownloadListener? = null): File? = downloadFile(
        modelDirectory(context),
        ModelFileSpec(SAMPLE_IMAGE_URL, "sample_image.jpg"),
        listener,
    )

    fun isGemma2ModelDownloaded(context: Context): Boolean =
        isLLMModelDownloaded(context, "gemma-2-2b-it-Q4_K_M.gguf")

    fun isSampleImageDownloaded(context: Context): Boolean =
        isDownloaded(modelDirectory(context), ModelFileSpec(SAMPLE_IMAGE_URL, "sample_image.jpg"))

    fun getGemma2ModelPath(context: Context): String =
        File(modelDirectory(context), "gemma-2-2b-it-Q4_K_M.gguf").absolutePath

    fun getSampleImagePath(context: Context): String =
        File(modelDirectory(context), "sample_image.jpg").absolutePath

    private fun isValid(file: File, spec: ModelFileSpec): Boolean {
        if (!file.isFile || !file.canRead() || file.length() <= 0) return false
        if (spec.expectedSize != null && file.length() != spec.expectedSize) return false

        val metadataFile = metadataFile(file)
        if (!metadataFile.isFile) {
            // Preserve valid models downloaded by older application versions. New downloads
            // always have metadata and receive full hash validation below.
            return spec.sha256 == null || sha256(file).equals(spec.sha256, ignoreCase = true)
        }
        val metadata = Properties().apply {
            FileInputStream(metadataFile).use(::load)
        }
        val recordedLength = metadata.getProperty("length")?.toLongOrNull() ?: return false
        val recordedSha256 = metadata.getProperty("sha256") ?: return false
        if (metadata.getProperty("url") != spec.url || recordedLength != file.length()) return false
        if (spec.sha256 != null && !recordedSha256.equals(spec.sha256, ignoreCase = true)) return false
        return sha256(file).equals(recordedSha256, ignoreCase = true)
    }

    private fun writeMetadata(file: File, url: String, length: Long, sha256: String) {
        val temp = File(file.parentFile, "${file.name}.metadata.download")
        val properties = Properties().apply {
            setProperty("url", url)
            setProperty("length", length.toString())
            setProperty("sha256", sha256)
        }
        FileOutputStream(temp).use { output ->
            properties.store(output, "ailia model download metadata")
            output.fd.sync()
        }
        moveChecked(temp, metadataFile(file))
    }

    private fun moveChecked(source: File, destination: File) {
        if (destination.exists() && !destination.delete()) {
            error("Failed to replace ${destination.absolutePath}")
        }
        check(source.renameTo(destination)) {
            "Failed to move ${source.absolutePath} to ${destination.absolutePath}"
        }
    }

    private fun deleteInvalidArtifacts(file: File) {
        listOf(
            file,
            metadataFile(file),
            File(file.parentFile, "${file.name}.metadata.download"),
            File(file.parentFile, "${file.name}.download"),
        )
            .filter(File::exists)
            .forEach { artifact ->
                if (!artifact.delete()) Log.w(TAG, "Failed to delete ${artifact.absolutePath}")
            }
    }

    private fun metadataFile(file: File): File = File(file.parentFile, "${file.name}.metadata")

    private fun migrateLegacyCache(context: Context, directory: File, fileName: String) {
        val destination = File(directory, fileName)
        val legacy = File(context.cacheDir, fileName)
        if (destination.exists() || !legacy.isFile || legacy.length() <= 0) return
        try {
            if (!legacy.renameTo(destination)) {
                legacy.copyTo(destination, overwrite = false)
                check(legacy.delete()) { "Failed to remove migrated cache file" }
            }
            Log.i(TAG, "Migrated model from cache: ${destination.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "Could not migrate legacy cache model $fileName", e)
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                if (count > 0) digest.update(buffer, 0, count)
            }
        }
        return digest.digest().toHex()
    }

    private fun ByteArray.toHex(): String = joinToString("") {
        String.format(Locale.ROOT, "%02x", it.toInt() and 0xFF)
    }
}
