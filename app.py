import shlex
import subprocess

from huggingface_hub import HfApi

api = HfApi()
api.snapshot_download(repo_id="XCLiu/InstaFlow_hidden", repo_type="space", local_dir=".")
subprocess.run(shlex.split("pip install -r requirements.txt"))
subprocess.run(shlex.split("python app.py"))
