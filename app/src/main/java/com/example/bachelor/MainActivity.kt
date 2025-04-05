package com.example.bachelor

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import coil.request.ImageRequest
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MLComparisonApp()
                }
            }
        }
    }
}

@Composable
fun MLComparisonApp(viewModel: MLViewModel = viewModel()) {
    val context = LocalContext.current
    val scrollState = rememberScrollState()

    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var processingTfLite by remember { mutableStateOf(false) }
    var processingPyTorch by remember { mutableStateOf(false) }
    var processingBatch by remember { mutableStateOf(false) }


    val tfLiteResults by viewModel.tfLiteResults.collectAsState()
    val pyTorchResults by viewModel.pyTorchResults.collectAsState()
    val tfLiteTimeMs by viewModel.tfLiteProcessingTime.collectAsState()
    val pyTorchTimeMs by viewModel.pyTorchProcessingTime.collectAsState()

    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let {
            selectedImageUri = it
            viewModel.clearResults()
        }
    }

    // Camera permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Toast.makeText(context, "Camera permission granted", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(context, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(scrollState),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "ML Model Comparison",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = {
                // Check camera permission first
                when {
                    ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.CAMERA
                    ) == PackageManager.PERMISSION_GRANTED -> {
                        // Camera permission is already granted, launch camera
                        Toast.makeText(context, "Camera opened", Toast.LENGTH_SHORT).show()
                        // Here you would launch camera activity if needed
                    }
                    else -> {
                        // Request camera permission
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            }) {
                Text("Take Photo")
            }

            Button(onClick = {
                imagePickerLauncher.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }) {
                Text("Select Image")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Selected image preview
        selectedImageUri?.let { uri ->
            AsyncImage(
                model = ImageRequest.Builder(LocalContext.current)
                    .data(uri)
                    .crossfade(true)
                    .build(),
                contentDescription = "Selected Image",
                modifier = Modifier
                    .size(300.dp)
                    .padding(16.dp)
            )

            // Process image buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(
                    onClick = {
                        processingTfLite = true
                        viewModel.processTFLiteImage(context, uri) {
                            processingTfLite = false
                        }


                    },
                    enabled = !processingTfLite && !processingPyTorch
                ) {
                    Text(if (processingTfLite) "Processing..." else "TensorFlow Lite")
                }

                Button(
                    onClick = {
                        processingPyTorch = true
                        viewModel.processPyTorchImage(context, uri) {
                            processingPyTorch = false
                        }
                    },
                    enabled = !processingPyTorch && !processingTfLite
                ) {
                    Text(if (processingPyTorch) "Processing..." else "PyTorch Mobile")
                }
            }

            // Results comparison
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = {
                try {
                    val model = Interpreter(FileUtil.loadMappedFile(context, "efficientnet_lite0.tflite"))
                    Log.d("BONUS", "EfficientNet geladen: $model")
                } catch (e: Exception) {
                    Log.e("BONUS", "Fehler beim Laden des Modells: ${e.message}")
                    e.printStackTrace()
                }
            }) {
                Text("Testmodell laden")
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {

                    Spacer(modifier = Modifier.height(24.dp))

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        Button(
                            onClick = {
                                processingBatch = true
                                viewModel.runBatchInference(context) { file ->
                                    processingBatch = false
                                    Toast.makeText(context, "CSV gespeichert:\n${file.name}", Toast.LENGTH_LONG).show()
                                    Log.d("CSV_UI", "CSV-Datei: ${file.absolutePath} (${file.length()} Bytes)")
                                }
                            },
                            enabled = !processingBatch
                        ) {
                            Text(if (processingBatch) "Läuft..." else "Batch-Test (alle Bilder)")
                        }

                        if (processingBatch) {
                            val batchProgress by viewModel.batchProgress.collectAsState()
                            val batchTotal by viewModel.batchTotal.collectAsState()

                            Spacer(modifier = Modifier.height(12.dp))
                            LinearProgressIndicator(
                                progress = if (batchTotal > 0) batchProgress / batchTotal.toFloat() else 0f,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(8.dp)
                            )
                            Text(
                                text = "$batchProgress von $batchTotal Bildern...",
                                fontSize = 14.sp,
                                modifier = Modifier.padding(top = 8.dp)
                            )
                        }



                        Button(onClick = {
                            val dir = context.getExternalFilesDir(null)
                            val latestFile = dir?.listFiles()
                                ?.filter { it.name.endsWith(".csv") }
                                ?.maxByOrNull { it.lastModified() }

                            if (latestFile != null && latestFile.exists()) {
                                val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", latestFile)
                                val intent = Intent(Intent.ACTION_SEND).apply {
                                    type = "text/csv"
                                    putExtra(Intent.EXTRA_STREAM, uri)
                                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                }
                                context.startActivity(Intent.createChooser(intent, "CSV teilen via"))
                            } else {
                                Toast.makeText(context, "Keine CSV-Datei gefunden", Toast.LENGTH_SHORT).show()
                            }
                        }) {
                            Text("CSV teilen")
                        }

                    }



                    Text(
                        text = "Results Comparison",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )

                    // TFLite results
                    if (tfLiteResults.isNotEmpty()) {
                        Text(
                            text = "TensorFlow Lite (${tfLiteTimeMs}ms):",
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                        tfLiteResults.take(3).forEach { result ->
                            Text(
                                text = "${result.label}: ${String.format("%.2f", result.score * 100)}%",
                                modifier = Modifier.padding(start = 8.dp, top = 4.dp)
                            )
                        }
                    }


                    // PyTorch results
                    if (pyTorchResults.isNotEmpty()) {
                        Text(
                            text = "PyTorch Mobile (${pyTorchTimeMs}ms):",
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(top = 16.dp)
                        )
                        pyTorchResults.take(3).forEach { result ->
                            Text(
                                text = "${result.label}: ${String.format("%.2f", result.score * 100)}%",
                                modifier = Modifier.padding(start = 8.dp, top = 4.dp)
                            )
                        }
                    }

                    // Performance comparison
                    if (tfLiteTimeMs > 0 && pyTorchTimeMs > 0) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Divider()
                        Spacer(modifier = Modifier.height(16.dp))

                        val fasterFramework = if (tfLiteTimeMs < pyTorchTimeMs) "TensorFlow Lite" else "PyTorch Mobile"
                        val timeDiff = kotlin.math.abs(tfLiteTimeMs - pyTorchTimeMs)
                        val percentFaster = (timeDiff / kotlin.math.max(tfLiteTimeMs, pyTorchTimeMs).toFloat()) * 100

                        Text(
                            text = "Performance: $fasterFramework was faster by ${String.format("%.2f", percentFaster)}% (${timeDiff}ms)",
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}