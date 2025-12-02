# GitHub Repository Setup Guide

## 📋 Quick Start

Follow these steps to create a GitHub repository for your RISC-V EdgeAI project.

## Method 1: Using GitHub CLI (Recommended)

### Prerequisites
1. Install GitHub CLI: https://cli.github.com/
2. Install Git: https://git-scm.com/downloads

### Steps

```powershell
# 1. Navigate to your project directory (you're already here!)
cd C:\Users\vjaligam\projects\riscv-edgeAI

# 2. Initialize Git repository
git init

# 3. Add all files
git add .

# 4. Make your first commit
git commit -m "Initial commit: Children height prediction model for RISC-V EdgeAI"

# 5. Login to GitHub (if not already logged in)
gh auth login

# 6. Create GitHub repository and push
gh repo create riscv-edgeAI --public --source=. --remote=origin --push
```

## Method 2: Using GitHub Web Interface

### Step 1: Create Repository on GitHub

1. Go to https://github.com
2. Click the **+** icon (top right) → **New repository**
3. Fill in the details:
   - **Repository name**: `riscv-edgeAI` (or your preferred name)
   - **Description**: "Children height prediction ML model for RISC-V edge devices"
   - **Visibility**: Public or Private (your choice)
   - ⚠️ **DO NOT** check "Initialize with README" (we already have one)
4. Click **Create repository**

### Step 2: Push Your Local Code

```powershell
# Navigate to your project directory
cd C:\Users\vjaligam\projects\riscv-edgeAI

# Initialize Git repository
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit: Children height prediction model for RISC-V EdgeAI"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/vjaligam/riscv-edgeAI.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Method 3: Using GitHub Desktop

1. Download GitHub Desktop: https://desktop.github.com/
2. Open GitHub Desktop
3. File → Add Local Repository
4. Browse to: `C:\Users\vjaligam\projects\riscv-edgeAI`
5. Click "Publish repository"
6. Choose repository name and visibility
7. Click "Publish repository"

## 📁 What Will Be Included?

Your repository will include:

### Core Files
- ✅ `README.md` - Project documentation
- ✅ `requirements.txt` - Python dependencies
- ✅ `LICENSE` - MIT License
- ✅ `.gitignore` - Ignore unnecessary files

### Python Scripts
- ✅ `children_age.py` - Data visualization
- ✅ `train_height_model.py` - Model training
- ✅ `hw_inference.py` - Hardware inference

### Model Files
- ✅ `height_model_quantized.tflite` - Quantized model (8.69 KB)
- ✅ `height_model.tflite` - Standard TFLite model
- ✅ `height_model.keras` - Full Keras model
- ✅ `height_model.h5` - Legacy H5 format
- ✅ `height_model_saved/` - SavedModel format
- ✅ `scaler.pkl` - Input preprocessor
- ✅ `model_info.json` - Model specifications

### Visualizations
- ✅ `training_history.png` - Training plots
- ✅ `predictions.png` - Prediction accuracy

## 🔒 Should You Include Model Files?

**Option 1: Include models (Recommended for showcase)**
- ✅ Others can immediately test your model
- ✅ Complete reproducibility
- ❌ Larger repository size (~100 KB)

**Option 2: Exclude models (Recommended for active development)**
- Add these lines to `.gitignore`:
```
*.h5
*.keras
*.tflite
*.pkl
height_model_saved/
```
- Users will need to run `train_height_model.py` to generate models

## 📝 Suggested Repository Description

```
Children Height Prediction ML Model for RISC-V EdgeAI 🚀

A lightweight neural network that predicts children's height based on age and gender, 
optimized for deployment on RISC-V edge devices. Includes complete training pipeline 
and TensorFlow Lite models (8.69 KB).

Features:
• 4.74 cm accuracy (MAE)
• 0.23 ms inference time
• WHO growth standards data
• TFLite quantized model
• Ready for edge deployment

Tech: TensorFlow, Python, RISC-V, Edge AI, TFLite
```

## 🏷️ Suggested Topics/Tags

Add these topics to your repository (on GitHub, click the gear icon next to "About"):

```
machine-learning
edge-ai
tensorflow
tflite
riscv
neural-network
embedded-ml
edge-computing
python
height-prediction
```

## 📊 GitHub Repository Settings

After creating the repository, consider:

1. **Add Description** - Use the suggested description above
2. **Add Topics** - Add relevant tags for discoverability
3. **Enable Issues** - For bug reports and feature requests
4. **Add Website** - Link to documentation or demo
5. **Add Releases** - Create v1.0 release with model files

## 🚀 Creating Your First Release

After pushing to GitHub:

```powershell
# Tag your first release
git tag -a v1.0 -m "Release v1.0: Initial model deployment"
git push origin v1.0
```

Then on GitHub:
1. Go to your repository
2. Click "Releases" → "Create a new release"
3. Select tag `v1.0`
4. Title: "v1.0 - Initial Release"
5. Description: Model performance metrics, features
6. Attach: `height_model_quantized.tflite`, `scaler.pkl`
7. Click "Publish release"

## 🔄 Future Updates

When making changes:

```powershell
# Check status
git status

# Add changes
git add .

# Commit with meaningful message
git commit -m "Add feature: batch prediction optimization"

# Push to GitHub
git push
```

## 🤝 Collaboration

To allow others to contribute:

1. Enable Issues on GitHub
2. Create CONTRIBUTING.md (optional)
3. Add branch protection rules (Settings → Branches)
4. Consider adding GitHub Actions for CI/CD

## 📱 Share Your Project

Once published, share your repository URL:
```
https://github.com/YOUR_USERNAME/riscv-edgeAI
```

Perfect for:
- LinkedIn posts
- Portfolio
- Job applications
- Academic papers
- Conference presentations

## ⚠️ Important Notes

1. **Never commit sensitive data** (API keys, passwords, credentials)
2. **Check .gitignore** before first commit
3. **Review files** with `git status` before committing
4. **Write meaningful commit messages**
5. **Update README** as project evolves

## 🎯 Next Steps

After creating your repository:

1. ✅ Star your own repo (shows confidence!)
2. ✅ Add detailed README badges (build status, license, etc.)
3. ✅ Create GitHub Pages site (Settings → Pages)
4. ✅ Add demo GIFs or screenshots
5. ✅ Share on social media
6. ✅ Submit to awesome lists (awesome-ml, awesome-edge-ai)

---

**Need Help?**
- GitHub Docs: https://docs.github.com
- Git Basics: https://git-scm.com/book/en/v2
- GitHub CLI: https://cli.github.com/manual/

Good luck with your repository! 🚀

