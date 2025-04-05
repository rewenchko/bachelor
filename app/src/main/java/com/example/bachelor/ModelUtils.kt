package com.example.mlcomparison

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Utility class for model-related operations
 */
class ModelUtils {
    companion object {
        private const val TF_LITE_MODEL = "mobilenet_v2.tflite"
        private const val PYTORCH_MODEL = "mobilenet_v2.pt"
        private const val LABELS_FILE = "imagenet_labels.txt"

        private const val TF_LITE_IMAGE_SIZE = 224
        private const val PYTORCH_IMAGE_SIZE = 224

        /**
         * Initialize TensorFlow Lite interpreter with GPU delegate if available
         */
        fun createTFLiteInterpreter(context: Context): Interpreter {
            val tfliteModel = FileUtil.loadMappedFile(context, TF_LITE_MODEL)
            val tfliteOptions = Interpreter.Options()

            // Use GPU delegate if available
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
               // tfliteOptions.addDelegate(GpuDelegate(delegateOptions))
            } else {
                // If GPU is not available, try to use multiple threads
                tfliteOptions.setNumThreads(4)
            }

            return Interpreter(tfliteModel, tfliteOptions)
        }

        /**
         * Initialize PyTorch Module
         */
        fun createPyTorchModule(context: Context): Module {
            val modelPath = assetFilePath(context, PYTORCH_MODEL)
            return Module.load(modelPath)
        }

        /**
         * Process image for TensorFlow Lite model
         */
        fun processBitmapForTFLite(bitmap: Bitmap): TensorImage {
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(TF_LITE_IMAGE_SIZE, TF_LITE_IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(127.5f, 127.5f))
                .build()

            val tensorImage = TensorImage.fromBitmap(bitmap)
            return imageProcessor.process(tensorImage)
        }

        /**
         * Process image for PyTorch model
         */
        fun processBitmapForPyTorch(bitmap: Bitmap): IValue {
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            return IValue.from(inputTensor)
        }

        /**
         * Load labels from assets
         */
        fun loadLabels(context: Context): List<String> {
            return context.assets.open(LABELS_FILE).bufferedReader().readLines()
        }

        /**
         * Copy asset file to cache directory to make it accessible
         */
        private fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.cacheDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }

            try {
                context.assets.open(assetName).use { inputStream ->
                    FileOutputStream(file).use { outputStream ->
                        val buffer = ByteArray(4 * 1024)
                        var read: Int
                        while (inputStream.read(buffer).also { read = it } != -1) {
                            outputStream.write(buffer, 0, read)
                        }
                        outputStream.flush()
                    }
                }
                return file.absolutePath
            } catch (e: IOException) {
                throw RuntimeException("Error copying asset file: $assetName", e)
            }
        }

        /**
         * Run a benchmarking test to compare TensorFlow Lite and PyTorch performance
         */
        fun runBenchmark(
            context: Context,
            bitmap: Bitmap,
            iterations: Int = 10
        ): Pair<BenchmarkResult, BenchmarkResult> {
            // Initialize models
            val tfliteInterpreter = createTFLiteInterpreter(context)
            val pyTorchModule = createPyTorchModule(context)

            // Prepare images
            val tfliteImage = processBitmapForTFLite(bitmap)
            val pyTorchInput = processBitmapForPyTorch(bitmap)

            // TensorFlow Lite benchmark
            val tfliteResults = mutableListOf<Long>()
            for (i in 0 until iterations) {
                val startTime = SystemClock.elapsedRealtime()

                // Create output buffer
                val outputBuffer = ByteBuffer.allocateDirect(1001 * 4).apply {
                    order(ByteOrder.nativeOrder())
                }

                // Run inference
                tfliteInterpreter.run(tfliteImage.buffer, outputBuffer)

                val endTime = SystemClock.elapsedRealtime()
                tfliteResults.add(endTime - startTime)
            }

            // PyTorch benchmark
            val pyTorchResults = mutableListOf<Long>()
            for (i in 0 until iterations) {
                val startTime = SystemClock.elapsedRealtime()

                // Run inference
                val output = pyTorchModule.forward(pyTorchInput)

                val endTime = SystemClock.elapsedRealtime()
                pyTorchResults.add(endTime - startTime)
            }

            // Clean up
            tfliteInterpreter.close()

            // Calculate statistics
            val tfliteAvg = tfliteResults.average()
            val pyTorchAvg = pyTorchResults.average()

            return Pair(
                BenchmarkResult("TensorFlow Lite", tfliteAvg, tfliteResults.min(), tfliteResults.max()),
                BenchmarkResult("PyTorch Mobile", pyTorchAvg, pyTorchResults.min(), pyTorchResults.max())
            )
        }
    }

    data class BenchmarkResult(
        val framework: String,
        val averageMs: Double,
        val minMs: Long,
        val maxMs: Long
    )
}