import os


from huggingface_hub import login
login(token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))