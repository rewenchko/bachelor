package com.example.bachelor

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class RecognitionResult(
    val label: String,
    val score: Float
)

class MLViewModel : ViewModel() {

    private val _batchProgress = MutableStateFlow(0)
    val batchProgress: StateFlow<Int> = _batchProgress

    private val _batchTotal = MutableStateFlow(0)
    val batchTotal: StateFlow<Int> = _batchTotal

    private val _tfLiteResults = MutableStateFlow<List<RecognitionResult>>(emptyList())
    val tfLiteResults: StateFlow<List<RecognitionResult>> = _tfLiteResults

    private val _pyTorchResults = MutableStateFlow<List<RecognitionResult>>(emptyList())
    val pyTorchResults: StateFlow<List<RecognitionResult>> = _pyTorchResults

    private val _tfLiteProcessingTime = MutableStateFlow(0L)
    val tfLiteProcessingTime: StateFlow<Long> = _tfLiteProcessingTime

    private val _pyTorchProcessingTime = MutableStateFlow(0L)
    val pyTorchProcessingTime: StateFlow<Long> = _pyTorchProcessingTime

    private lateinit var tfLiteLabels: List<String>
    private lateinit var pyTorchLabels: List<String>

    fun clearResults() {
        _tfLiteResults.value = emptyList()
        _pyTorchResults.value = emptyList()
        _tfLiteProcessingTime.value = 0
        _pyTorchProcessingTime.value = 0
    }

    fun processTFLiteImage(context: Context, uri: Uri, onComplete: () -> Unit) {
        viewModelScope.launch {
            try {
                if (!::tfLiteLabels.isInitialized) {
                    tfLiteLabels = loadTFLiteLabels(context).drop(1)

                }

                val bitmap = getBitmapFromUri(context, uri)
                val startTime = System.currentTimeMillis()

                val results = withContext(Dispatchers.Default) {
                    runTFLiteInference(context, bitmap)
                }

                val endTime = System.currentTimeMillis()
                _tfLiteProcessingTime.value = endTime - startTime
                _tfLiteResults.value = results
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                onComplete()
            }
        }
    }

    fun processPyTorchImage(context: Context, uri: Uri, onComplete: () -> Unit) {
        viewModelScope.launch {
            try {
                if (!::pyTorchLabels.isInitialized) {
                    pyTorchLabels = loadPyTorchLabels(context)
                }

                val bitmap = getBitmapFromUri(context, uri)
                val startTime = System.currentTimeMillis()

                val results = withContext(Dispatchers.Default) {
                    runPyTorchInference(context, bitmap)
                }

                val endTime = System.currentTimeMillis()
                _pyTorchProcessingTime.value = endTime - startTime
                _pyTorchResults.value = results
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                onComplete()
            }
        }
    }

    private suspend fun runTFLiteInference(context: Context, bitmap: Bitmap): List<RecognitionResult> {
        return withContext(Dispatchers.IO) {
            try {
                val usedMemoryBefore = Debug.getNativeHeapAllocatedSize()

                val interpreter = loadEfficientNetTFLiteModel(context)
                android.util.Log.d("BONUS", "EfficientNet geladen: $interpreter")

                val imageProcessor = ImageProcessor.Builder()
                    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(NormalizeOp(127.5f, 127.5f))
                    .build()

                val tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(bitmap)
                val processedImage = imageProcessor.process(tensorImage)

                val probabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1000), DataType.FLOAT32)

                val startTime = System.nanoTime()
                interpreter.run(processedImage.buffer, probabilityBuffer.buffer)
                val endTime = System.nanoTime()
                val durationMs = (endTime - startTime) / 1_000_000

                val usedMemoryAfter = Debug.getNativeHeapAllocatedSize()
                val memoryUsedKB = (usedMemoryAfter - usedMemoryBefore) / 1024

                val outputProcessor = TensorProcessor.Builder().build()
                val labeledProbability = TensorLabel(tfLiteLabels, outputProcessor.process(probabilityBuffer))

                val results = labeledProbability.mapWithFloatValue.entries
                    .sortedByDescending { it.value }
                    .take(5)
                    .map { RecognitionResult(it.key, it.value) }

                interpreter.close()

                results.firstOrNull()?.let {
                }

                results
            } catch (e: Exception) {
                e.printStackTrace()
                emptyList()
            }
        }
    }

    private suspend fun runPyTorchInference(context: Context, bitmap: Bitmap): List<RecognitionResult> {
        return withContext(Dispatchers.IO) {
            try {
                val assetFilePath = copyAssetToCache(context, "mobilenet_v2_scripted.pt")
                val module = Module.load(assetFilePath)

                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB
                )

                val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
                val logits = outputTensor.dataAsFloatArray

                // ✅ Softmax korrekt berechnen:
                val expScores = logits.map { Math.exp(it.toDouble()) }
                val sumExp = expScores.sum()
                val softmax = expScores.map { (it / sumExp).toFloat() }

                // 🔝 Top-5 Ergebnisse mit Labels
                logits.indices.sortedByDescending { softmax[it] }
                    .take(5)
                    .map {
                        RecognitionResult(
                            pyTorchLabels.getOrElse(it) { "Unknown" },
                            softmax[it]
                        )
                    }
            } catch (e: Exception) {
                e.printStackTrace()
                emptyList()
            }
        }
    }

    fun getBitmapFromUri(context: Context, uri: Uri): Bitmap {
        val original = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }

        return ThumbnailUtils.extractThumbnail(original, 224, 224)
    }

    private fun loadTFLiteLabels(context: Context): List<String> {
        return context.assets.open("imagenet_labels.txt").bufferedReader().readLines()
    }

    private fun loadPyTorchLabels(context: Context): List<String> {
        // Drop die erste Zeile ("background") – wie bei TFLite
        return context.assets.open("imagenet_labels.txt").bufferedReader().readLines().drop(1)
    }

    private fun copyAssetToCache(context: Context, assetName: String): String {
        val file = File(context.cacheDir, assetName)
        if (!file.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    private fun loadEfficientNetTFLiteModel(context: Context): Interpreter {
        val tfliteModel = FileUtil.loadMappedFile(context, "efficientnet_lite0.tflite")
        return Interpreter(tfliteModel)
    }

    private fun loadEfficientNetTFLiteInt8Model(context: Context): Interpreter {
        val tfliteModel = FileUtil.loadMappedFile(context, "efficientnet_lite0_int8.tflite")
        val options = Interpreter.Options()
        return Interpreter(tfliteModel, options)
    }



    fun runBatchInference(context: Context, onComplete: (File) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                if (!::tfLiteLabels.isInitialized) {
                    tfLiteLabels = loadTFLiteLabels(context).drop(1)
                }
                if (!::pyTorchLabels.isInitialized) {
                    pyTorchLabels = loadPyTorchLabels(context)
                }

                val groundTruth = loadGroundTruth(context)

                var tfTop1 = 0
                var tfTop5 = 0
                var tfInt8Top1 = 0
                var tfInt8Top5 = 0
                var ptTop1 = 0
                var ptTop5 = 0
                var totalImages = 0

                val startBattery = getBatteryLevel(context)

                // Ladezeiten & Modellgrößen
                val tfStartLoad = System.nanoTime()
                val tfInterpreter = loadEfficientNetTFLiteModel(context)
                val tfEndLoad = System.nanoTime()
                val tfLoadTimeMs = (tfEndLoad - tfStartLoad) / 1_000_000
                val tfModelSizeKb = context.assets.openFd("efficientnet_lite0.tflite").length / 1024

                val tfInt8StartLoad = System.nanoTime()
                val tfInt8Interpreter = loadEfficientNetTFLiteInt8Model(context)
                val tfInt8EndLoad = System.nanoTime()
                val tfInt8LoadTimeMs = (tfInt8EndLoad - tfInt8StartLoad) / 1_000_000
                val tfInt8ModelSizeKb = context.assets.openFd("efficientnet_lite0_int8.tflite").length / 1024

                val ptStartLoad = System.nanoTime()
                val ptFile = File(copyAssetToCache(context, "mobilenet_v2_scripted.pt"))
                val ptModule = Module.load(ptFile.absolutePath)
                val ptEndLoad = System.nanoTime()
                val ptLoadTimeMs = (ptEndLoad - ptStartLoad) / 1_000_000
                val ptModelSizeKb = ptFile.length() / 1024

                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                val fileName = "benchmark_$timestamp.csv"
                val file = File(context.getExternalFilesDir(null), fileName)

                val imageNames = context.assets.list("test_images")
                if (imageNames.isNullOrEmpty()) {
                    Log.e("BATCH", "Keine Bilder im assets/test_images Verzeichnis gefunden!")
                    return@launch
                }

                _batchProgress.value = 0
                _batchTotal.value = imageNames.size

                FileWriter(file).use { writer ->
                    writer.append("timestamp,device,framework,model,image,label,score,durationMs,memoryKB,batteryDrop,rank\n")

                    for ((index, imageName) in imageNames.withIndex()) {
                        _batchProgress.value = index + 1

                        try {
                            val inputStream = context.assets.open("test_images/$imageName")
                            val bitmap = BitmapFactory.decodeStream(inputStream)

                            val batteryBefore = getBatteryLevel(context)

                            val (tfResults, tfTime, tfMemory) = runTFLiteInferenceWithMeta(context, bitmap)
                            val (tfInt8Results, tfInt8Time, tfInt8Memory) = runTFLiteInt8InferenceWithMeta(context, bitmap)
                            val (ptResults, ptTime, ptMemory) = runPyTorchInferenceWithMeta(context, bitmap)

                            val batteryAfter = getBatteryLevel(context)
                            val batteryDrop = (batteryBefore - batteryAfter).coerceAtLeast(0)

                            val groundLabel = groundTruth[imageName]?.lowercase()?.trim()

                            val tfRank = tfResults.indexOfFirst { it.label.lowercase().trim() == groundLabel } + 1
                            val tfInt8Rank = tfInt8Results.indexOfFirst { it.label.lowercase().trim() == groundLabel } + 1
                            val ptRank = ptResults.indexOfFirst { it.label.lowercase().trim() == groundLabel } + 1

                            tfResults.firstOrNull()?.let {
                                logResultToCSV(writer, "TFLite", "efficientnet_lite0", it.label, it.score, tfTime, tfMemory, imageName, batteryDrop, tfRank)
                                if (tfRank == 1) tfTop1++
                                if (tfRank in 1..5) tfTop5++
                            }

                            tfInt8Results.firstOrNull()?.let {
                                logResultToCSV(writer, "TFLite", "efficientnet_lite0_int8", it.label, it.score, tfInt8Time, tfInt8Memory, imageName, batteryDrop, tfInt8Rank)
                                if (tfInt8Rank == 1) tfInt8Top1++
                                if (tfInt8Rank in 1..5) tfInt8Top5++
                            }

                            ptResults.firstOrNull()?.let {
                                logResultToCSV(writer, "PyTorch", "mobilenet_v2", it.label, it.score, ptTime, ptMemory, imageName, batteryDrop, ptRank)
                                if (ptRank == 1) ptTop1++
                                if (ptRank in 1..5) ptTop5++
                            }

                            totalImages++
                        } catch (e: Exception) {
                            Log.e("BATCH", "Fehler beim Verarbeiten von $imageName: ${e.message}")
                            e.printStackTrace()
                        }
                    }

                    val tfAcc1 = tfTop1.toFloat() / totalImages * 100
                    val tfAcc5 = tfTop5.toFloat() / totalImages * 100
                    val tfInt8Acc1 = tfInt8Top1.toFloat() / totalImages * 100
                    val tfInt8Acc5 = tfInt8Top5.toFloat() / totalImages * 100
                    val ptAcc1 = ptTop1.toFloat() / totalImages * 100
                    val ptAcc5 = ptTop5.toFloat() / totalImages * 100

                    val endBattery = getBatteryLevel(context)
                    val batteryUsed = startBattery - endBattery
                    val avgBatteryDrop = if (totalImages > 0) batteryUsed.toFloat() / totalImages else 0f

                    writer.append("\n")
                    writer.append("# Summary\n")
                    writer.append("Top-1 Accuracy TFLite,$tfAcc1%\n")
                    writer.append("Top-5 Accuracy TFLite,$tfAcc5%\n")
                    writer.append("Top-1 Accuracy TFLite INT8,$tfInt8Acc1%\n")
                    writer.append("Top-5 Accuracy TFLite INT8,$tfInt8Acc5%\n")
                    writer.append("Top-1 Accuracy PyTorch,$ptAcc1%\n")
                    writer.append("Top-5 Accuracy PyTorch,$ptAcc5%\n")
                    writer.append("Battery Start,$startBattery%\n")
                    writer.append("Battery End,$endBattery%\n")
                    writer.append("Battery Used,$batteryUsed%\n")
                    writer.append("Average battery per image,$avgBatteryDrop%\n")
                    writer.append("Model Load Time TFLite,$tfLoadTimeMs ms\n")
                    writer.append("Model Load Time TFLite INT8,$tfInt8LoadTimeMs ms\n")
                    writer.append("Model Load Time PyTorch,$ptLoadTimeMs ms\n")
                    writer.append("Model Size TFLite,$tfModelSizeKb KB\n")
                    writer.append("Model Size TFLite INT8,$tfInt8ModelSizeKb KB\n")
                    writer.append("Model Size PyTorch,$ptModelSizeKb KB\n")
                    writer.flush()

                    Log.d("BATCH_FINAL", "Accuracy & Battery + Model-Info gespeichert")
                }

                Log.d("CSV_OUTPUT", "CSV gespeichert unter: ${file.absolutePath} (${file.length()} Bytes)")

                withContext(Dispatchers.Main) {
                    onComplete(file)
                }

            } catch (e: Exception) {
                Log.e("CSV_ERROR", "Fehler beim CSV schreiben: ${e.message}")
                e.printStackTrace()
            }
        }
    }


    private fun logResultToCSV(
        writer: FileWriter,
        framework: String,
        model: String,
        label: String,
        score: Float,
        durationMs: Long,
        memoryKB: Long,
        imageName: String,
        batteryDrop: Int,
        rank: Int // 🆕 Rank des korrekten Labels (1 = Top-1, 2 = Top-2, ...; 0 = nicht in Top-5)
    ) {
        val timestamp = System.currentTimeMillis()
        val device = "${Build.MANUFACTURER} ${Build.MODEL}"
        val confidencePercent = "%.2f".format(score * 100)

        val line = "$timestamp,$device,$framework,$model,$imageName,$label,$confidencePercent,$durationMs,$memoryKB,$batteryDrop,$rank\n"
        writer.append(line)
        Log.d("CSV_WRITE", line.trim())
    }

    private suspend fun runTFLiteInt8InferenceWithMeta(context: Context, bitmap: Bitmap): Triple<List<RecognitionResult>, Long, Long> {
        return withContext(Dispatchers.IO) {
            val runtime = Runtime.getRuntime()
            val usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory()

            val interpreter = loadEfficientNetTFLiteInt8Model(context)

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .build()

            val tensorImage = TensorImage(DataType.UINT8)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            val probabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1000), DataType.UINT8)

            val startTime = System.nanoTime()
            interpreter.run(processedImage.buffer, probabilityBuffer.buffer)
            val endTime = System.nanoTime()

            val usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory()
            val memoryUsedKB = (usedMemoryAfter - usedMemoryBefore).coerceAtLeast(0) / 1024
            val durationMs = (endTime - startTime) / 1_000_000

            val rawArray = probabilityBuffer.buffer.array()
            val intValues = rawArray.map { it.toInt() }

            val topIndices = intValues.indices.sortedByDescending { intValues[it] }.take(5)

            val results = topIndices.map { index ->
                val label = tfLiteLabels.getOrElse(index) { "Unknown" }
                val score = (intValues[index] + 128) / 255f

                RecognitionResult(label, score)
            }

            Log.d("INT8_DEBUG", "Top-5: " + results.joinToString { "${it.label}: ${it.score}" })

            interpreter.close()
            Triple(results, durationMs, memoryUsedKB)
        }
    }


    private suspend fun runTFLiteInferenceWithMeta(context: Context, bitmap: Bitmap): Triple<List<RecognitionResult>, Long, Long> {
        return withContext(Dispatchers.IO) {
            val runtime = Runtime.getRuntime()
            val usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory()

            val interpreter = loadEfficientNetTFLiteModel(context)

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(NormalizeOp(127.5f, 127.5f))
                .build()

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            val probabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1000), DataType.FLOAT32)

            val startTime = System.nanoTime()
            interpreter.run(processedImage.buffer, probabilityBuffer.buffer)
            val endTime = System.nanoTime()

            val usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory()
            val memoryUsedKB = (usedMemoryAfter - usedMemoryBefore).coerceAtLeast(0) / 1024
            val durationMs = (endTime - startTime) / 1_000_000

            val outputProcessor = TensorProcessor.Builder().build()
            val labeledProbability = TensorLabel(tfLiteLabels, outputProcessor.process(probabilityBuffer))

            val map = labeledProbability.mapWithFloatValue

            // 🔍 Log Top-10 Scores
            val top10 = map.entries
                .sortedByDescending { it.value }
                .take(10)
                .joinToString(", ") { "${it.key}: ${"%.4f".format(it.value)}" }

            Log.d("SCORES_TFLITE", "Top-10 Scores: $top10")

            val results = map.entries
                .sortedByDescending { it.value }
                .take(5)
                .map { RecognitionResult(it.key, it.value) }

            interpreter.close()
            Triple(results, durationMs, memoryUsedKB)
        }
    }

    private suspend fun runPyTorchInferenceWithMeta(context: Context, bitmap: Bitmap): Triple<List<RecognitionResult>, Long, Long> {
        return withContext(Dispatchers.IO) {
            val runtime = Runtime.getRuntime()
            val usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory()

            val assetFilePath = copyAssetToCache(context, "mobilenet_v2_scripted.pt")
            val module = Module.load(assetFilePath)

            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            val startTime = System.nanoTime()
            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val endTime = System.nanoTime()

            val usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory()
            val memoryUsedKB = (usedMemoryAfter - usedMemoryBefore).coerceAtLeast(0) / 1024
            val durationMs = (endTime - startTime) / 1_000_000

            val scores = outputTensor.dataAsFloatArray
            val normalizedScores = softmax(scores)

            // Debug: Top-10 Scores loggen
            val top10 = scores.indices
                .sortedByDescending { normalizedScores[it] }
                .take(10)
                .map { index -> "${pyTorchLabels.getOrElse(index) { "Unknown" }}: ${"%.4f".format(normalizedScores[index])}" }
                .joinToString(", ")

            Log.d("SCORES_PT", "Top-10 Scores: $top10")


            val results = scores.indices.sortedByDescending { normalizedScores[it] }
                .take(5)
                .map { RecognitionResult(pyTorchLabels.getOrElse(it) { "Unknown" }, normalizedScores[it]) }

            Triple(results, durationMs, memoryUsedKB)
        }
    }

    fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { Math.exp((it - maxLogit).toDouble()) }
        val sumExps = exps.sum()
        return exps.map { (it / sumExps).toFloat() }.toFloatArray()
    }

    private fun resizeBitmap(input: Bitmap, width: Int, height: Int): Bitmap {
        return Bitmap.createScaledBitmap(input, width, height, true)
    }

    fun getBatteryLevel(context: Context): Int {
        val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
        return batteryManager.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
    }

    private suspend fun loadGroundTruth(context: Context): Map<String, String> {
        return withContext(Dispatchers.IO) {
            val inputStream = context.assets.open("ground_truth.csv")
            val lines = inputStream.bufferedReader().readLines().drop(1)
            lines.mapNotNull {
                val parts = it.split(",")
                if (parts.size == 2) parts[0] to parts[1] else null
            }.toMap()
        }
    }
}
