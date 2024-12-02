import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# preprocess all files
print("-- Preparing data -- ")
#use pathlib to get current working dir
cwd = Path.cwd()
print(cwd)
source_dir = Path(r"D:\Publications\Paper4-Wetting-ridge\test_data\mult\021224")
# source_dir = Path(r"D:\Messungen\Profilometry\Lasermikroskop\Wetting_Ridge\dynamics_soft")

glb = source_dir.rglob("*_raw.lext")


for file in glb:
    python2_command = f'C:/Python27/python.exe -W ignore {cwd/"gwypy_lext_processing.py"} -p -o {Path(file).parent/"polylevel3" } "{Path(file)}"'
    ret = subprocess.check_output(python2_command)
    print(ret.decode())

# python2_commands = []
# for file in glb:
#     python2_commands.append(f'C:/Python27/python.exe -W ignore {cwd/"gwypy_lext_processing.py"} -p -o {Path(file).parent } "{Path(file)}"')

# cpus = 4
# with ProcessPoolExecutor(max_workers=cpus) as executor:
#     for ret in executor.map(subprocess.check_output, python2_commands):
#         print(ret.decode())
