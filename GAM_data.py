import pandas as pd
import numpy as np
from pathlib import Path

# Путь к таблице
df = pd.read_csv("./4DNFIZCJ4CEY.csv", sep = '\t')

# Отделим координаты и бинарную часть
coords = df.iloc[:, :3]
S_all = df.iloc[:, 3:].to_numpy(dtype=np.float32)

# Объединим обратно, чтобы проще было разбить
df_coords = pd.concat([coords, pd.DataFrame(S_all)], axis=1)

# Папка для сохранения результатов
out_dir = Path("gam_matrices_by_chr")
out_dir.mkdir(exist_ok=True)

# Обработка по хромосомам
for chrom in df_coords['chrom'].unique():
    df_chr = df_coords[df_coords['chrom'] == chrom].reset_index(drop=True)

    # Бинарная матрица (бин x срез)
    S = df_chr.iloc[:, 3:].to_numpy(dtype=np.float32)
    M = S.shape[1]

    # Ко-сегрегационная матрица
    C_raw = (S @ S.T) / M
    C_diag = np.diag(C_raw)

    # Сохраняем
    np.save(out_dir / f"C_{chrom}_raw.npy", C_raw)
