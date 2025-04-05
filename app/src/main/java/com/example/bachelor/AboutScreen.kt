package com.example.mlcomparison

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun AboutScreen() {
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(scrollState),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "About This App",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "ML Comparison App",
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    "This app demonstrates and compares two popular machine learning frameworks for Android:",
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Divider(modifier = Modifier.padding(vertical = 8.dp))

                FrameworkInfo(
                    "TensorFlow Lite",
                    "Google's lightweight solution for mobile and edge devices.",
                    listOf(
                        "Optimized for Android devices",
                        "Supports hardware acceleration",
                        "Excellent quantization support",
                        "Smaller model sizes"
                    )
                )

                Spacer(modifier = Modifier.height(16.dp))

                FrameworkInfo(
                    "PyTorch Mobile",
                    "Mobile version of the popular PyTorch framework.",
                    listOf(
                        "Dynamic computation graphs",
                        "Smooth research-to-production workflow",
                        "Intuitive API for PyTorch developers",
                        "Better support for newer model architectures"
                    )
                )

                Divider(modifier = Modifier.padding(vertical = 16.dp))

                Text(
                    text = "App Features",
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp,
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                BulletPoint("Image recognition using pre-trained models")
                BulletPoint("Side-by-side performance comparison")
                BulletPoint("Detailed benchmarking with multiple iterations")
                BulletPoint("Modern UI with Jetpack Compose")

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "This app was created as a demonstration and comparison tool. The models used are MobileNetV2 variants pre-trained on the ImageNet dataset."
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Getting Started",
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    "1. In the Image Recognition tab, select or take a photo to analyze",
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    "2. Use the buttons to process the image with either TensorFlow Lite or PyTorch",
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    "3. View the classification results and performance metrics",
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                Text(
                    "4. Use the Benchmarking tab to run multiple iterations for a detailed comparison",
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }
        }
    }
}

@Composable
private fun FrameworkInfo(title: String, description: String, features: List<String>) {
    Text(
        text = title,
        fontWeight = FontWeight.Bold,
        fontSize = 16.sp,
        modifier = Modifier.padding(bottom = 4.dp)
    )

    Text(
        text = description,
        modifier = Modifier.padding(bottom = 8.dp)
    )

    features.forEach { feature ->
        BulletPoint(feature)
    }
}

@Composable
private fun BulletPoint(text: String) {
    Row(
        modifier = Modifier.padding(vertical = 2.dp)
    ) {
        Text(
            text = "• ",
            fontWeight = FontWeight.Bold
        )
        Text(text = text)
    }
}