# DefectVision - 3D Industrial Anomaly Detection

Real-time anomaly detection system for industrial quality control using YOLO and computer vision.

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://gitlab.com/imad.sanoussi.mail-group/defectvision.git
cd defectvision

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Connect your repository**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitLab account
   - Select this repository

2. **Configure deployment**
   - **Repository**: `imad.sanoussi.mail-group/defectvision`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: `3.9+` (recommended 3.10)

3. **Deploy**
   - Click "Deploy new app"
   - Wait for build completion (2-3 minutes)

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- OpenCV 4.8+
- Streamlit 1.28+
- Ultralytics 8.0+

## 🏗️ Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── src/                      # Source code modules
├── models/                   # Model configurations
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
└── .streamlit/              # Streamlit configuration
```

## 🔧 Features

- **Real-time anomaly detection** using YOLO models
- **Multi-class classification** for different defect types
- **Interactive web interface** with Streamlit
- **Camera integration** for live inspection
- **Batch processing** for image uploads
- **Performance metrics** and visualization

## 📊 Model Information

- **Architecture**: YOLOv8-based anomaly detection
- **Classes**: Multiple defect categories
- **Training**: MVTec 3D-AD dataset
- **Input size**: 800x800 pixels

## 🌐 Deployment

### Streamlit Cloud (Recommended)
- Free tier available
- Automatic builds from Git
- Custom domains supported
- Built-in monitoring

### Alternative Platforms
- Heroku (requires Dyno for ML workloads)
- Railway (modern alternative)
- DigitalOcean App Platform
- AWS/GCP/Azure (enterprise)

## 📝 Notes

- Model weights are excluded from git (use `.pt` files in production)
- Large datasets are excluded from version control
- Camera access requires HTTPS in production
- GPU acceleration available on paid tiers

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Merge Request

## 📄 License

This project is licensed under the MIT License.

---

**Deployed on Streamlit Cloud**: [Live Demo URL] (after deployment)
