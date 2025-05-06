import os
import numpy as np
import pandas as pd
import random
from itertools import combinations
from scipy.spatial.distance import cdist
import zipfile
import subprocess

def binned_distance_matrix(coords, bin_size=15):
    assert coords.shape[0] % bin_size == 0
    n_bins = coords.shape[0] // bin_size
    full_dist = cdist(coords, coords)
    pooled = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            block = full_dist[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size]
            pooled[i, j] = block.mean()
    return pooled

def random_shift_vector(length=15.0):
    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)
    return direction * length

# Папки назначения
base_output = "/home/user/In-silico_Hi-C_GAM_SPRITE"
paths = {
    "structures": os.path.join(base_output, "data/structures"),
    "pairs": os.path.join(base_output, "data/pairs"),
    "distance": os.path.join(base_output, "Distance_matrix"),
    "HiC": os.path.join(base_output, "Hi-C"),
    "GAM": os.path.join(base_output, "GAM"),
    "SPRITE": os.path.join(base_output, "SPRITE"),
    "HiC Cool": os.path.join(base_output, "Hi-C Cool"),
    "HiC Balanced": os.path.join(base_output, "Hi-C Balanced"),
    "SPRITE Cool": os.path.join(base_output, "SPRITE Cool"),
    "SPRITE Balanced": os.path.join(base_output, "SPRITE Balanced")
}
for path in paths.values():
    os.makedirs(path, exist_ok=True)

# Пути к данным
dataset_folder = "/home/user/dataset"
n_structures = 100
n_pairs = 250

# Обрабатываем все ensemble*.npy и block*.zip
files = sorted([f for f in os.listdir(dataset_folder) if f.startswith("ensemble")])
for ens_file in files:
    n = int(ens_file.replace("ensemble", "").replace(".npy", ""))
    zip_file = f"block{n}.zip"

    ens_path = os.path.join(dataset_folder, ens_file)
    zip_path = os.path.join(dataset_folder, zip_file)

    # Пути к будущим выходным файлам
    out_matrix = os.path.join(paths['distance'], f"distance_{n}.txt")
    out_hic    = os.path.join(paths['HiC'],     f"hic_{n}.txt")
    out_gam    = os.path.join(paths['GAM'],     f"gam_{n}.txt")
    out_sprite = os.path.join(paths['SPRITE'],  f"sprite_{n}.txt")
    out_hic_balanced    = os.path.join(paths['HiC Balanced'],     f"hic_{n}.txt")

    # Проверка, если уже всё сделано
    if all(os.path.exists(p) for p in [out_matrix, out_hic, out_gam, out_sprite]):
    #if all(os.path.exists(p) for p in [out_matrix, out_hic, out_gam, out_sprite, out_hic_balanced]) or str(n).startswith(('1', '2', '0', '3', '41')):
        print(f"[✓] Блок {n} уже обработан — пропускаем.")
        continue

    if all(os.path.exists(p) for p in [out_matrix, out_hic, out_gam, out_sprite]):
        print(f"Блок {n} нуждается в детализации с балансировкой.")
        

    print(f"[→] Обработка блока {n}...")

    # Загрузка ensemble
    ens = np.load(ens_path)

    # Обработка ZIP
    matrixs = []
    for i in range(n_structures):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(f"block_{i}.pdb") as file:
                df = pd.read_csv(file, sep='\s+', skiprows=1, header=None, nrows=2250)
                coords = df.iloc[:, [6, 7, 8]].values
                matrixs.append(binned_distance_matrix(coords))
                column = ens[i, :].reshape(-1, 1)
                coords = np.hstack((coords, column))
                np.savetxt(f"{paths['structures']}/structure_{i+1}.txt", coords, delimiter=' ')

    # Средняя матрица
    matrix = sum(matrixs) / len(matrixs)
    np.savetxt(out_matrix, matrix, delimiter=' ')

    # Генерация пар
    structures = {i: pd.read_csv(f"{paths['structures']}/structure_{i}.txt", sep='\s+', header=None)
                  for i in range(1, n_structures+1)}
    pairs = random.sample(list(combinations(range(1, n_structures+1), 2)), n_pairs)

    for idx, (i, j) in enumerate(pairs, 1):
        df1 = structures[i]
        df2 = structures[j].copy()
        shift = random_shift_vector(length=50.0)
        df2.iloc[:, :3] += shift
        combined = pd.concat([df1, df2], axis=1).values
        np.savetxt(f"{paths['pairs']}/pair_{idx}.txt", combined, delimiter=' ')

    # Запуск C-программ
    exec_path = os.path.join(base_output, "main/main.out")
    for method in [1, 2, 3]:

        subprocess.run(
            [exec_path, str(method), "3000", "0.05"],
            check=True,
            cwd=base_output+'/main'  # устанавливаем рабочую директорию
        )

    # Перемещение матриц
    os.rename(f"{base_output}/main/hic_mat_3000_005.txt", out_hic)
    os.rename(f"{base_output}/main/gam_mat_3000_005.txt", out_gam)
    os.rename(f"{base_output}/main/sprite_mat_3000_005.txt", out_sprite)

    print(f"[✓] Обработка блока {n} завершена.")


