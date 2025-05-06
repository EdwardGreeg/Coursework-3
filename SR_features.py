import torch
import numpy as np
import logging
import os

MODEL_LOAD_PATH = '/content/unet_hic_to_distance_best.pth' 
DATA_DIR_HIC = '/content/output_matrices'
DATA_DIR_DIST = '/content/output_matrices' # Нужен только для инициализации Dataset
N_SR_SAMPLES = 10000  
TARGET_ROW = 20      # Пример: Целевая строка выходной матрицы D
TARGET_COL = 130     # Пример: Целевой столбец выходной матрицы D 
PATCH_SIZE = 5

def extract_features(c_matrix, u, v, matrix_size=150, k_local=5):
    """
    Извлекает набор признаков из входной матрицы C для предсказания D_uv.

    Args:
        c_matrix (np.ndarray): Входная матрица C (150x150).
        u (int): Индекс целевой строки.
        v (int): Индекс целевого столбца.
        matrix_size (int): Размер матрицы (150).
        k_local (int): Размер окна для локальных признаков (нечетное).

    Returns:
        np.ndarray: 1D массив признаков.
    """
    features = []
    k_radius = k_local // 2

    # 1. Расстояние от диагонали
    diag_dist = abs(u - v)
    features.append(diag_dist)

    # 2. Значение C_uv (если не на диагонали)
    c_uv = c_matrix[u, v] if u != v else 0 # Или c_matrix[u, u] если u==v и это нужно
    features.append(c_uv)

    # 3. Локальное среднее в окне k x k
    row_start = max(0, u - k_radius)
    row_end = min(matrix_size, u + k_radius + 1)
    col_start = max(0, v - k_radius)
    col_end = min(matrix_size, v + k_radius + 1)
    local_patch = c_matrix[row_start:row_end, col_start:col_end]
    local_mean = np.mean(local_patch) if local_patch.size > 0 else 0
    features.append(local_mean)

    # 4. Среднее в строке u
    row_u_mean = np.mean(c_matrix[u, :])
    features.append(row_u_mean)

    # 5. Среднее в столбце v
    col_v_mean = np.mean(c_matrix[:, v])
    features.append(col_v_mean)

    # 6. Сумма в строке u
    row_u_sum = np.sum(c_matrix[u, :])
    features.append(row_u_sum)

    # 7. Сумма в столбце v
    col_v_sum = np.sum(c_matrix[:, v])
    features.append(col_v_sum)

    # 8. Нормированная позиция u
    pos_u = u / (matrix_size - 1.0)
    features.append(pos_u)

    # 9. Нормированная позиция v
    pos_v = v / (matrix_size - 1.0)
    features.append(pos_v)

    # 10. Глобальная разреженность
    global_sparsity = np.mean(c_matrix == 0)
    features.append(global_sparsity)

    return np.array(features, dtype=np.float32)

FEATURE_NAMES = [
    "diag_dist",
    "c_uv",
    f"local_mean_{PATCH_SIZE}x{PATCH_SIZE}", 
    "row_u_mean",
    "col_v_mean",
    "row_u_sum",
    "col_v_sum",
    "pos_u",
    "pos_v",
    "global_sparsity", 
]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Используется устройство для SR (Features): {device}")
try:

    model_unet = UNet(n_channels_in=1, n_channels_out=1).to(device)
    model_unet.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    model_unet.eval() # Режим оценки
    logging.info(f"Стандартная U-Net модель загружена с {MODEL_LOAD_PATH}")
except FileNotFoundError:
    logging.error(f"Файл модели не найден: {MODEL_LOAD_PATH}")
    exit()
except NameError:
     logging.error("Класс UNet не определен.")
     exit()
except Exception as e:
    logging.error(f"Не удалось загрузить U-Net модель: {e}")
    exit()

logging.info(f"Генерация данных для SR с использованием признаков (макс. {N_SR_SAMPLES} сэмплов)...")
X_sr_list = [] 
Y_sr_list = [] 
input_matrices_for_sr = [] 

try:

    temp_dataset = HiCDistanceDatasetIndexed(hic_dir=DATA_DIR_HIC, distance_dir=DATA_DIR_DIST)
    num_available_samples = len(temp_dataset.indices)
    if num_available_samples == 0: raise ValueError("Датасет пуст.")
    actual_sr_samples = min(N_SR_SAMPLES, num_available_samples)
    if actual_sr_samples < N_SR_SAMPLES: logging.warning(f"Доступно только {actual_sr_samples} сэмплов.")
    if actual_sr_samples == 0: raise ValueError("Нет доступных сэмплов.")


    logging.info(f"Загрузка {actual_sr_samples} входных матриц C...")
    for idx in range(actual_sr_samples):
        try:
            numerical_index = temp_dataset.indices[idx]
            input_filename = f"{temp_dataset.input_prefix}{numerical_index}.txt"
            input_path = os.path.join(temp_dataset.input_dir, input_filename)
            c_matrix_np = np.nan_to_num(np.loadtxt(input_path, dtype=np.float32))
            np.fill_diagonal(c_matrix_np, 0)

            if c_matrix_np.shape == (150, 150): input_matrices_for_sr.append(c_matrix_np)
            else: logging.warning(f"Пропуск {input_filename} из-за размера {c_matrix_np.shape}")
        except Exception as e: logging.warning(f"Ошибка загрузки {input_filename}: {e}")

    if not input_matrices_for_sr: raise ValueError("Не загружено ни одной матрицы C.")
    logging.info(f"Загружено {len(input_matrices_for_sr)} матриц C.")


    logging.info("Получение предсказаний D_uv и извлечение признаков...")
    batch_size_inference = 32
    with torch.no_grad(): 
        for i in range(0, len(input_matrices_for_sr), batch_size_inference):
            batch_c_np = np.array(input_matrices_for_sr[i:i+batch_size_inference])
            batch_c_tensor = torch.from_numpy(batch_c_np).unsqueeze(1).to(device)


            batch_d_tensor_pred = model_unet(batch_c_tensor) # [B, 1, H, W]
            batch_d_pred_np = batch_d_tensor_pred.squeeze(1).cpu().numpy() # [B, H, W]

            for idx_in_batch in range(batch_d_pred_np.shape[0]):
                 c_matrix_np_single = batch_c_np[idx_in_batch]
                 d_matrix_pred_single = batch_d_pred_np[idx_in_batch]

                 target_value = d_matrix_pred_single[TARGET_ROW, TARGET_COL]
                 Y_sr_list.append(target_value)

                 features_vector = extract_features(c_matrix_np_single,
                                                    TARGET_ROW, TARGET_COL,
                                                    k_local=PATCH_SIZE) 
                 X_sr_list.append(features_vector)

    X_sr = np.array(X_sr_list)
    Y_sr = np.array(Y_sr_list)

    if X_sr.size == 0 or Y_sr.size == 0: raise ValueError("Массивы X_sr или Y_sr пусты.")
    if X_sr.ndim != 2 or X_sr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Неожиданная форма X_sr: {X_sr.shape}. Ожидалось (?, {len(FEATURE_NAMES)})")

    logging.info(f"Данные для SR сгенерированы. Форма X_sr (признаки): {X_sr.shape}, Форма Y_sr: {Y_sr.shape}")

except Exception as e:
    logging.error(f"Ошибка при генерации данных для SR с признаками: {e}")
    X_sr = np.array([])
    Y_sr = np.array([])

if X_sr.ndim == 2 and Y_sr.ndim == 1 and X_sr.shape[0] > 0 and X_sr.shape[0] == Y_sr.shape[0]:
    logging.info("Запуск PySR с инженерными признаками...")
    try:
        from pysr import PySRRegressor # Импортируем здесь

        variable_names = FEATURE_NAMES
        if len(variable_names) != X_sr.shape[1]:
             raise ValueError(f"Количество имен переменных ({len(variable_names)}) не совпадает с количеством признаков в X_sr ({X_sr.shape[1]})")

        model_sr_features = PySRRegressor(
            niterations=50, 
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["cos", "exp", "log", "sin", "sqrt", "abs", "tanh", "inv(x)=1/x"],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            variable_names=variable_names, 
            progress=True,
            model_selection="best"
        )

        model_sr_features.fit(X_sr, Y_sr)

        print("--- Результаты PySR с инженерными признаками ---")
        try:
             print("Лучшая формула (SymPy):")
             print(model_sr_features.sympy())
        except ImportError:
             print("Установите sympy для красивого вывода формулы: pip install sympy")
             print(f"Лучшая формула (строка): {model_sr_features.latex()}")

    except ImportError:
         logging.error("Библиотека PySR не найдена. Установите её: pip install -U pysr")
    except ValueError as e:
         logging.error(f"Ошибка значения при запуске PySR: {e}")
    except Exception as e:
         logging.error(f"Ошибка во время работы PySR: {e}")
         logging.error("Убедитесь, что Julia установлена и PySR настроен корректно.")
else:
    logging.error("Не удалось запустить PySR с признаками: входные данные X_sr или Y_sr некорректны или пусты.")
