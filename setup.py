import os
import subprocess

# Define the base path
base_path = '/content/StableDIFF/repository/'

# Change to the base directory
os.chdir(base_path)

# Clone the CodeFormer repository
print("Cloning CodeFormer repository...")
subprocess.run(['git', 'clone', 'https://github.com/notsk11/CodeFormer'])

# Change to the CodeFormer directory
codeformer_path = os.path.join(base_path, 'CodeFormer')
os.chdir(codeformer_path)

# Install python dependencies from requirements.txt
print("Installing python dependencies from requirements.txt...")
subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

# Install basicsr using its setup.py script in develop mode
print("Installing basicsr in develop mode...")
subprocess.run(['python', 'basicsr/setup.py', 'develop'])

# Download the pre-trained models
print("Downloading pre-trained models (facelib and CodeFormer)...")
subprocess.run(['python', 'scripts/download_pretrained_models.py', 'facelib'])
subprocess.run(['python', 'scripts/download_pretrained_models.py', 'CodeFormer'])

print("Installation and setup complete.")
