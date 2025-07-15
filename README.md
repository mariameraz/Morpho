# Installation

# --------------------------------------------
# COMPLETE TUTORIAL: INSTALL CONDA + SETUP ENVIRONMENT
# --------------------------------------------

To download **Morpho** repository from GitHub as a ZIP file (without using Git), follow these steps:

---

### **Via GitHub's Web Interface**  
1. **Go to your repository**:  
   Open [https://github.com/mariameraz/Morpho](https://github.com/mariameraz/Morpho) in a browser.  

2. **Click the green "Code" button** (top right of the file list).  

3. **Select "Download ZIP"**:  
   ![Image](https://docs.github.com/assets/cb-138303/images/help/repository/download-repo-zip.png)  

4. **Unzip the file** (`Morpho-main.zip`) to your desired folder.  

5. **Set up the Conda environment** from the unzipped folder:  


# === 1. Install Miniconda (OS-specific) ===
# Run the appropriate command for your OS in a terminal:

# Linux/Mac (in terminal):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Windows (in PowerShell as Admin):
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda.exe"
Start-Process -FilePath "miniconda.exe" -ArgumentList "/S /AddToPath=1 /D=$HOME\miniconda" -Wait
Remove-Item miniconda.exe
# Restart PowerShell after installation

# === 2. Verify installation ===
conda --version  # Should show something like "conda 23.11.0"


# === 3. Create and activate environment from your environment.yml ===
# Ensure morpho_env.yml is in your current directory
# (Replace with actual path if needed)

   ```bash
   cd Morpho-main
   conda env create -f morpho_env.yml 
   conda activate morpho_env  # Use the name from your environment.yml
   ```

# === 4. Verify everything works ===
conda list  # Show installed packages

