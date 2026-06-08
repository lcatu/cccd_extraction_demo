"""
Script upload hf_space_demo len Hugging Face Spaces.
Chay: python upload_to_hf.py
"""

import subprocess
import sys
import os
import shutil

# ==========================================
# CẤU HÌNH - Bạn chỉ cần sửa TOKEN ở đây
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN", "PASTE_YOUR_TOKEN_HERE")

HF_SPACE  = "AnhTu03/demo_cccd_extraction"
REPO_URL  = f"https://AnhTu03:{HF_TOKEN}@huggingface.co/spaces/{HF_SPACE}"

# Thư mục clone tạm
CLONE_DIR = os.path.join(os.path.dirname(__file__), "_hf_clone_tmp")

# Các file/folder cần upload từ hf_space_demo
SRC_DIR = os.path.dirname(__file__)
FILES_TO_COPY = [
    "app.py",
    "requirements.txt",
    "packages.txt",
    "README.md",
    "models",
]

def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True)
    return result

def main():
    if HF_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        print("ERROR: Hãy mở file này và dán HF_TOKEN của bạn vào trước khi chạy!")
        sys.exit(1)

    # Xóa thư mục clone cũ nếu có
    if os.path.exists(CLONE_DIR):
        print(f"Đang xóa thư mục clone cũ: {CLONE_DIR}")
        shutil.rmtree(CLONE_DIR)

    # Bước 1: Clone repo HF Space về máy
    print("\n=== Bước 1: Clone HF Space repo ===")
    run(["git", "clone", REPO_URL, CLONE_DIR])

    # Bước 2: Cấu hình Git LFS trong clone repo
    print("\n=== Bước 2: Cài đặt Git LFS ===")
    run(["git", "lfs", "install"], cwd=CLONE_DIR)

    # Track các file model lớn bằng LFS
    run(["git", "lfs", "track", "*.pth"], cwd=CLONE_DIR)
    run(["git", "lfs", "track", "*.pt"],  cwd=CLONE_DIR)

    # Bước 3: Copy các file cần thiết vào clone repo
    print("\n=== Bước 3: Copy files vào repo ===")
    for item in FILES_TO_COPY:
        src = os.path.join(SRC_DIR, item)
        dst = os.path.join(CLONE_DIR, item)
        if not os.path.exists(src):
            print(f"  [SKIP] Không tìm thấy: {src}")
            continue
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("desktop.ini"))
            print(f"  [DIR]  {item}/")
        else:
            shutil.copy2(src, dst)
            print(f"  [FILE] {item}")

    # Bước 4: Tạo .gitattributes (LFS tracking) nếu chưa commit
    gitattr = os.path.join(CLONE_DIR, ".gitattributes")
    print(f"\n=== Bước 4: .gitattributes ===")
    print(f"  -> {gitattr}")

    # Bước 5: Git add, commit, push
    print("\n=== Bước 5: Commit & Push ===")
    run(["git", "add", "-A"], cwd=CLONE_DIR)
    run(["git", "commit", "-m", "Upload demo CCCD extraction app"], cwd=CLONE_DIR)
    run(["git", "push"], cwd=CLONE_DIR)

    print("\n✅ DONE! Space của bạn đã được cập nhật tại:")
    print(f"   https://huggingface.co/spaces/{HF_SPACE}")

if __name__ == "__main__":
    main()
