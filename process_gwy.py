import subprocess
from pathlib import Path

# preprocess all files
print("-- Preparing data -- ")
#use pathlib to get current working dir
cwd = Path.cwd()
print(cwd)
source_dir = Path(r"D:\Publications\Paper4-Wetting-ridge\test_data\mult")
# source_dir = Path(r"D:\Messungen\Profilometry\Lasermikroskop\Wetting_Ridge\dynamics_soft")

glb = source_dir.rglob("*_raw.lext")

skip_existing = True

for file in glb:
    python2_command = f'C:/Python27/python.exe -W ignore {cwd/"gwypy_lext_processing.py"} -p -o {Path(file).parent / "tiffs" } "{Path(file)}"'
    if skip_existing and (Path(file).with_suffix(".gz").exists() or Path(file).with_suffix(".tif").exists()):
        print(f"Skipping {file} as it already exists")
        continue
    ret = subprocess.check_output(python2_command)
    print(ret.decode())