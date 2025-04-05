package com.example.mlcomparison

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun BenchmarkScreen() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    var isRunning by remember { mutableStateOf(false) }
    var tfLiteResult by remember { mutableStateOf<ModelUtils.BenchmarkResult?>(null) }
    var pyTorchResult by remember { mutableStateOf<ModelUtils.BenchmarkResult?>(null) }

    var iterations by remember { mutableStateOf(10) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(scrollState),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Model Benchmarking",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        // Iterations slider
        Text("Number of iterations: $iterations")
        Slider(
            value = iterations.toFloat(),
            onValueChange = { iterations = it.toInt() },
            valueRange = 1f..50f,
            steps = 49,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                coroutineScope.launch {
                    isRunning = true
                    val benchmarkResults = runBenchmark(context, iterations)
                    tfLiteResult = benchmarkResults.first
                    pyTorchResult = benchmarkResults.second
                    isRunning = false
                }
            },
            enabled = !isRunning
        ) {
            Text(if (isRunning) "Running Benchmark..." else "Run Benchmark")
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Results
        if (isRunning) {
            CircularProgressIndicator(modifier = Modifier.padding(16.dp))
        } else {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Benchmark Results",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )

                    if (tfLiteResult != null && pyTorchResult != null) {
                        ResultRow("Framework", "Avg (ms)", "Min (ms)", "Max (ms)")
                        Divider(modifier = Modifier.padding(vertical = 8.dp))

                        tfLiteResult?.let {
                            ResultRow(
                                it.framework,
                                String.format("%.2f", it.averageMs),
                                it.minMs.toString(),
                                it.maxMs.toString()
                            )
                        }

                        pyTorchResult?.let {
                            ResultRow(
                                it.framework,
                                String.format("%.2f", it.averageMs),
                                it.minMs.toString(),
                                it.maxMs.toString()
                            )
                        }

                        Spacer(modifier = Modifier.height(16.dp))
                        Divider(modifier = Modifier.padding(bottom = 16.dp))

                        // Comparison
                        val tfAvg = tfLiteResult?.averageMs ?: 0.0
                        val pyAvg = pyTorchResult?.averageMs ?: 0.0

                        val fasterFramework = if (tfAvg < pyAvg) "TensorFlow Lite" else "PyTorch Mobile"
                        val speedupRatio = kotlin.math.max(tfAvg, pyAvg) / kotlin.math.max(1.0, kotlin.math.min(tfAvg, pyAvg))

                        Text(
                            text = "Conclusion: $fasterFramework is ${String.format("%.2fx", speedupRatio)} faster",
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "Based on $iterations iterations with a test image",
                            fontSize = 12.sp,
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    } else {
                        Text("Run the benchmark to see results")
                    }
                }
            }
        }
    }
}

@Composable
private fun ResultRow(col1: String, col2: String, col3: String, col4: String) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = col1,
            modifier = Modifier.weight(1.5f),
            fontWeight = FontWeight.Medium
        )
        Text(
            text = col2,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = col3,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = col4,
            modifier = Modifier.weight(1f)
        )
    }
}

private suspend fun runBenchmark(
    context: Context,
    iterations: Int
): Pair<ModelUtils.BenchmarkResult, ModelUtils.BenchmarkResult> {
    return withContext(Dispatchers.Default) {
        // Load a sample image from assets for benchmarking
        val inputStream = context.assets.open("sample_image.jpg")
        val bitmap = BitmapFactory.decodeStream(inputStream)

        // Run benchmark
        ModelUtils.runBenchmark(context, bitmap, iterations)
    }
}