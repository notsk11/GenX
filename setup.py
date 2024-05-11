import os
import subprocess
import contextlib

if not os.path.exists('/content/GenX/repository/'):
    os.mkdir('/content/GenX/repository/')
else:
    print("Directory already exists.")

os.chdir('/content/GenX/repository/')

with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    result = subprocess.run(['git', 'clone', 'https://github.com/notsk11/CodeFormer', '/content/GenX/repository/CodeFormer/'])

if result.returncode == 0:
    print("Copied CodeFormer Repo")
else:
    print("Failed to copy CodeFormer Repo")

os.chdir('/content/GenX/repository/CodeFormer/')

with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'])
    print("Installed CodeFormer Requirements")

process = subprocess.run(['python', 'basicsr/setup.py', 'develop'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print("Installed BasicSR")

subprocess.run(['python', 'scripts/download_pretrained_models.py', 'facelib'])
subprocess.run(['python', 'scripts/download_pretrained_models.py', 'CodeFormer'])
print("Downloaded pre-trained models (facelib and CodeFormer)...")

os.chdir('/content/GenX/')

with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'])
    print("Installed GenX Requirements")
print("Installation and setup complete.")
