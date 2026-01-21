# Face Recognition System v1.0.0

![System Overview](https://img.shields.io/badge/Version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Ready-red)
![License](https://img.shields.io/badge/License-MIT-orange)

A modular, high-performance solution for real-time face recognition and mask violation detection designed for multi-source video processing with intelligent optimization capabilities.

## 📋 Project Overview

The **Face Recognition System v1.0.0** is a comprehensive, modular face recognition system engineered for **real-time processing** across multiple video sources simultaneously. The system delivers robust face recognition, progressive mask detection, and context-aware performance optimization for deployment in security, surveillance, and compliance monitoring scenarios.

### Core Capabilities

- **Real-time Multi-Source Processing**: Handle multiple RTSP streams, CCTV feeds, local cameras, and video files concurrently
- **Advanced Face Recognition**: High-accuracy identity matching with temporal consistency
- **Progressive Mask Detection**: Temporal analysis for reliable mask compliance monitoring
- **Intelligent Resource Optimization**: Dynamic scaling based on scene complexity and hardware capabilities

## ✨ Key Features

### 🎥 Multi-Source Streaming

- **Thread-safe Stream Manager**: Simultaneous processing of multiple RTSP, CCTV, local cameras, and video files
- **Automatic Stream Recovery**: Health monitoring with automatic reconnection attempts every 30 seconds
- **CCTV Naming System**: Automatic identifier extraction from RTSP URLs with fallback to source IDs

### 🎭 Progressive Mask Detection

- **Weighted Temporal Analysis**: Avoids hard resets by using weighted label progressive calculations
- **Duration-Aware Verification**: Mask status verification over time for reliable compliance monitoring
- **Dynamic Threshold Adjustment**: Context-sensitive violation detection

### ⚡ Intelligent Scaling

- **SceneContextAnalyzer**: Adjusts resolution based on face density, scene complexity, lighting conditions, and motion levels
- **Adaptive Processing**: Automatically optimizes processing parameters for current scene requirements
- **Resource-Aware Optimization**: Balances accuracy and performance based on available hardware

### 🔍 High-Speed Recognition Engine

- **Voyager Approximate Nearest Neighbor**: Rapid identity matching with high-dimensional embeddings
- **GPU Acceleration**: Optimized for GTX 1650 Ti GPU with PyTorch tensor operations
- **Multiple Backend Support**: Facenet512 or ArcFace embeddings with JSON database storage

### 🔊 Duration-Aware Alerting

- **Threshold-Based Notification**: Triggers synchronized audio alerts and server pushes only after violation meets specific duration and frame thresholds
- **Multi-Channel Output**: Supports voice interfaces, visual indicators, and server notifications
- **Configurable Sensitivity**: Adjustable parameters for different operational requirements

### ⚙️ Unified Configuration

- **ConfigManager**: Hierarchical configuration management with validation against min/max ranges and type checks
- **Predefined Profiles**: Optimized settings for different operational modes
- **Runtime Adjustment**: Dynamic configuration changes without system restart

## 🏗️ System Architecture

### Modular Package Structure

```
face-recognition-system/
├── recognition/           # Face recognition engines
│   ├── base_engine.py    # Base recognition functionality
│   ├── robust_engine.py  # Enhanced recognition with fault tolerance
│   └── voyager_search.py # Voyager-based approximate nearest neighbor
├── streaming/            # Video stream management
│   ├── realtime_processor.py    # Real-time processing loops
│   ├── stream_manager.py        # Multi-source stream management
│   └── health_monitor.py        # Stream health monitoring
├── processing/           # Image processing modules
│   ├── face_detector.py    # YOLO-based face detection
│   ├── quality_assessor.py # Face quality assessment
│   └── temporal_fusion.py  # Temporal analysis for mask detection
├── tracking/            # Object tracking
│   ├── byte_tracker.py    # ByteTrack integration
│   └── identity_manager.py # Identity fairness control
├── alerting/            # Notification systems
│   ├── voice_interface.py # Audio alert generation
│   └── notification_manager.py # Duration-tracked notifications
├── logging/             # Data logging systems
│   ├── csv_logger.py    # CSV data logging
│   └── image_logger.py  # Base64/JSON server push capabilities
└── utils/               # Utility functions
    ├── config_manager.py # Configuration management
    └── resource_monitor.py # System resource tracking
```

## 📦 Installation & Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration, optional)
- **RAM**: Minimum 8GB, 16GB recommended
- **GPU**: NVIDIA GTX 1650 Ti or equivalent (optional but recommended)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/your-organization/face-recognition-system.git
cd face-recognition-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Required Libraries

```bash
# Core dependencies
pip install opencv-python>=4.8.0
pip install ultralytics>=8.0.0  # YOLO models
pip install deepface>=0.0.79
pip install onnxruntime-gpu>=1.15.0
pip install scikit-learn>=1.3.0
pip install psutil>=5.9.0
pip install requests>=2.31.0

# Optional but recommended
pip install nvidia-ml-py3>=12.0  # GPU monitoring
pip install pyaudio>=0.2.11      # Audio alerts
```

**Note**: The system automatically detects CUDA availability and falls back to CPU processing if necessary. For optimal performance, ensure CUDA-compatible drivers are installed.

## ⚙️ Configuration Profiles

The system uses a hierarchical configuration system managed by `ConfigManager`. Switch between predefined operational modes:

### Profile Comparison Table

| Profile                  | Resolution | Scale Factor | Recognition Threshold | Temporal Fusion | Use Case                         |
| ------------------------ | ---------- | ------------ | --------------------- | --------------- | -------------------------------- |
| **Balanced**             | 1280×720   | 1.0          | 0.6                   | Enabled         | General purpose, default setting |
| **High Accuracy**        | 1920×1080  | 1.5          | 0.75                  | Enabled         | Critical security applications   |
| **High Performance**     | 854×480    | 0.7          | 0.5                   | Disabled        | High FPS requirements            |
| **Small Face Detection** | 1600×900   | 2.0          | 0.65                  | Enabled         | Distant or crowded scenes        |

### Configuration Parameters

```python
# Example configuration structure
config = {
    "streaming": {
        "max_sources": 4,
        "health_check_interval": 30,
        "reconnection_attempts": 3
    },
    "processing": {
        "face_detection_confidence": 0.7,
        "mask_detection_confidence": 0.8,
        "temporal_window": 15
    },
    "recognition": {
        "model_name": "Facenet512",
        "distance_metric": "cosine",
        "threshold": 0.6
    },
    "alerting": {
        "violation_duration_threshold": 30,
        "violation_frame_threshold": 15,
        "enable_voice_alerts": True
    }
}
```

## 🚀 Usage Instructions

### Starting the System

```bash
# Primary entry point with default configuration
python entry_multi.py

# With custom configuration file
python entry_multi.py --config configs/custom_config.json

# Enable multiple sources with specific RTSP streams
python entry_multi.py --multi-source --rtsp rtsp://stream1 --rtsp rtsp://stream2

# Force CPU mode (disable GPU)
python entry_multi.py --no-gpu

# Test server connectivity
python entry_multi.py --test-server
```

### Command Line Arguments

| Argument         | Description                                              | Default                |
| ---------------- | -------------------------------------------------------- | ---------------------- |
| `--config`       | Path to custom configuration file                        | `configs/default.json` |
| `--multi-source` | Enable multi-source processing                           | `False`                |
| `--rtsp`         | RTSP stream URLs (can be multiple)                       | None                   |
| `--cctv-index`   | CCTV camera indices (comma-separated)                    | None                   |
| `--no-gpu`       | Disable GPU acceleration                                 | `False`                |
| `--test-server`  | Test server connectivity on startup                      | `False`                |
| `--profile`      | Use predefined profile (high_acc, high_perf, small_face) | `balanced`             |

### Interactive Controls

During runtime, use these keyboard shortcuts:

| Key     | Function       | Description                                           |
| ------- | -------------- | ----------------------------------------------------- |
| **q**   | Quit           | Gracefully shutdown all processes                     |
| **l**   | Logging Toggle | Enable/disable data logging                           |
| **m**   | Layout Cycle   | Switch between Grid, Horizontal, and Vertical layouts |
| **v**   | Voice Alerts   | Toggle audio notification alerts                      |
| **s**   | Screenshot     | Capture current display to file                       |
| **p**   | Pause/Resume   | Pause processing on all streams                       |
| **1-9** | Source Focus   | Maximize selected source in grid layout               |
| **+/-** | Zoom           | Adjust display zoom level                             |

## 🔧 Technical Highlights for Developers

### GPU Optimization

- **FP16 Precision**: Models loaded with mixed precision for memory efficiency
- **CUDA Provider Optimization**: Specific provider configuration for ONNX runtime
- **Tensor Management**: Efficient PyTorch tensor operations with GPU memory pooling
- **Automatic Fallback**: Seamless transition to CPU when GPU memory is constrained

### Dynamic Identity Policy

- **Transparent Display**: Known identities remain visible during violation verification
- **Fairness Control**: Prevention of identity bias in crowded scenarios
- **Temporal Consistency**: Identity persistence across frames and occlusions

### Resource Management

- **Memory Monitoring**: Automatic tracking with 1GB threshold warnings
- **Thread Pool Management**: Dynamic thread allocation based on source count
- **File Handle Tracking**: Prevention of resource leaks through systematic monitoring
- **GPU Memory Optimization**: Adaptive model loading based on available VRAM

### Validation & Error Handling

- **Configuration Validation**: All parameters validated against min/max ranges and type checks
- **Graceful Degradation**: System maintains partial functionality during component failures
- **Comprehensive Logging**: Structured logs for debugging and performance analysis

## 🖥️ Output Visualization

### Display Layouts

#### Grid Layout

- **Arrangement**: Equal-sized grid for all active sources
- **Overlay**: Source health indicators, FPS counter, violation counts
- **Navigation**: Keyboard shortcuts for source focus

#### Horizontal Layout

- **Arrangement**: Sources displayed horizontally with adjustable widths
- **Features**: Timeline visualization for violation events
- **Metrics**: Real-time performance statistics per stream

#### Vertical Layout

- **Arrangement**: Vertical stacking with detailed per-source analytics
- **Details**: Expanded recognition information and confidence scores
- **Controls**: Individual stream controls (pause, restart, configure)

### Overlay Information

Each display includes:

- **Source Identifier**: CCTV name or stream URL
- **Health Status**: Connection quality indicator (Green/Yellow/Red)
- **FPS Counter**: Current processing rate
- **Violation Counter**: Verified mask violations
- **System Status**: GPU/CPU utilization and memory usage
- **Recognition Status**: Active identities and confidence levels

## 📊 Performance Metrics

| Metric                   | Value Range          | Optimal |
| ------------------------ | -------------------- | ------- |
| **Processing FPS**       | 15-30 FPS per stream | 25+ FPS |
| **Recognition Accuracy** | 92-98%               | 95%+    |
| **Detection Latency**    | 50-150ms             | <100ms  |
| **GPU Memory Usage**     | 2-4GB                | 3GB     |
| **CPU Utilization**      | 30-70%               | 50%     |

## 🔄 API Integration

### Server Push Configuration

```json
{
  "server": {
    "endpoint": "https://your-server.com/api/violations",
    "api_key": "your-api-key",
    "push_interval": 5,
    "max_retries": 3,
    "timeout": 10
  },
  "data_format": {
    "include_image": true,
    "image_format": "base64",
    "metadata_fields": ["timestamp", "camera_id", "identity", "confidence"]
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Low FPS**
   - Check GPU drivers and CUDA installation
   - Reduce stream resolution or number of sources
   - Switch to High Performance profile

2. **Stream Connection Issues**
   - Verify network connectivity and RTSP URLs
   - Check firewall settings for port access
   - Increase `reconnection_attempts` in configuration

3. **High Memory Usage**
   - Enable `resource_monitor` in configuration
   - Reduce `max_sources` parameter
   - Clear identity database cache periodically

4. **Recognition Accuracy Problems**
   - Update face embeddings database
   - Adjust recognition threshold in configuration
   - Ensure proper lighting conditions in video sources

### Logging and Debugging

```bash
# Enable verbose logging
python entry_multi.py --log-level DEBUG

# Generate performance report
python utils/performance_report.py --duration 60

# Check system compatibility
python utils/system_check.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For technical support, feature requests, or bug reports:

- **Issues**: [GitHub Issues](https://github.com/JabirJelek/faceRecog/issues)
- **Email**: faridraihan17@gmail.com

---

**System Version**: 1.0.0  
**Last Updated**: October 2023  
**Compatibility**: Windows 10+
