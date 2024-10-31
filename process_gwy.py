import subprocess
from pathlib import Path

# preprocess all files
print("-- Preparing data -- ")
#use pathlib to get current working dir
cwd = Path.cwd()
print(cwd)
source_dir = Path(r"D:\Publications\Paper4-Wetting-ridge\test_data\mult")
glb = source_dir.rglob("*_raw.lext")

for file in glb:
    python2_command = f'C:/Python27/python.exe -W ignore {cwd/"gwypy_lext_processing.py"} -p -o {Path(file).parent } "{Path(file)}"'
    ret = subprocess.check_output(python2_command)
    print(ret.decode())