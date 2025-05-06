import sympy
import torch
import numpy as np
from pysr import PySRRegressor
import logging
import os 

MODEL_LOAD_PATH = '/content/unet_hic_to_distance_best.pth'      # На примере Hi-C
DATA_DIR_HIC = '/content/output_matrices'
DATA_DIR_DIST = '/content/output_matrices' # Нужен только для инициализации Dataset, но не для таргетных значений SR
N_SR_SAMPLES = 10000   # Количество сэмплов для обучения SR
PATCH_SIZE = 5     # Размер локального патча (нечетное число)
TARGET_ROW = 75     # Целевая строка выходной матрицы Distance
TARGET_COL = 95    # Целевой столбец выходной матрицы Distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:

    model_unet = UNet(n_channels_in=1, n_channels_out=1).to(device)
    model_unet.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    model_unet.eval() 
except FileNotFoundError:
    logging.error(f"Файл модели не найден: {MODEL_LOAD_PATH}")
    exit()
except NameError:
     logging.error("Класс UNet не определен.")
     exit()
except Exception as e:
    logging.error(f"Не удалось загрузить U-Net модель: {e}")
    exit()

# Генерация данных для SR
X_sr_list = [] # Входные данные для SR (патчи) 
Y_sr_list = [] # Выходные данные для SR (предсказания U-Net) 
input_matrices_for_sr = [] # Список для хранения загруженных input-матриц

try:

    temp_dataset = HiCDistanceDatasetIndexed(hic_dir=DATA_DIR_HIC, distance_dir=DATA_DIR_DIST) # target_dir нужен для конструктора, но не используется здесь

    num_available_samples = len(temp_dataset.indices)
    if num_available_samples == 0:
        raise ValueError("Не найдено подходящих файлов с совпадающими индексами в датасете.")

    actual_sr_samples = min(N_SR_SAMPLES, num_available_samples)
    if actual_sr_samples < N_SR_SAMPLES:
         logging.warning(f"Доступно только {actual_sr_samples} сэмплов, используется это количество.")
    if actual_sr_samples == 0:
         raise ValueError("Нет доступных сэмплов для генерации SR данных.")

    logging.info(f"Загрузка {actual_sr_samples} входных матриц C...")
    for idx in range(actual_sr_samples):
        try:
            numerical_index = temp_dataset.indices[idx]
            input_filename = f"{temp_dataset.input_prefix}{numerical_index}.txt"
            input_path = os.path.join(temp_dataset.input_dir, input_filename)

            c_matrix_np = np.nan_to_num(np.loadtxt(input_path, dtype=np.float32))
            np.fill_diagonal(c_matrix_np, 0)

            if c_matrix_np.shape == (150, 150):
                input_matrices_for_sr.append(c_matrix_np)
            else:
                logging.warning(f"Пропуск файла {input_filename} из-за неверной размерности {c_matrix_np.shape}")
        except FileNotFoundError:
             logging.warning(f"Файл {input_filename} не найден при загрузке для SR.")
        except ValueError as e:
             logging.warning(f"Ошибка значения при загрузке файла {input_filename} для SR: {e}")
        except Exception as e:
             logging.warning(f"Не удалось загрузить файл {input_filename} для SR: {e}")

    if not input_matrices_for_sr:
         raise ValueError("Не удалось загрузить ни одной корректной входной матрицы для SR.")

    logging.info(f"Загружено {len(input_matrices_for_sr)} входных матриц C.")
    logging.info("Получение предсказаний от U-Net и извлечение данных для SR...")

    patch_radius = PATCH_SIZE // 2
    batch_size_inference = 32 

    for i in range(0, len(input_matrices_for_sr), batch_size_inference):
        batch_c_np = np.array(input_matrices_for_sr[i:i+batch_size_inference]) 

        batch_c_tensor = torch.from_numpy(batch_c_np).unsqueeze(1).to(device)

        # Получаем предсказания D от U-Net
        with torch.no_grad():
            batch_d_tensor_pred = model_unet(batch_c_tensor) # [B, 1, H, W]

        # Убираем измерение канала [B, 1, H, W] -> [B, H, W]
        batch_d_pred_np = batch_d_tensor_pred.squeeze(1).cpu().numpy()

        # Извлекаем данные для SR из текущего батча
        for idx_in_batch in range(batch_d_pred_np.shape[0]):
             # Матрица C для текущего элемента батча
             c_matrix_np_single = batch_c_np[idx_in_batch]
             # Матрица D (предсказанная) для текущего элемента батча
             d_matrix_pred_single = batch_d_pred_np[idx_in_batch]

             # Извлекаем целевое значение Y из предсказания D
             target_value = d_matrix_pred_single[TARGET_ROW, TARGET_COL]
             Y_sr_list.append(target_value)

             # Извлекаем входной патч X из оригинальной матрицы C
             row_start = max(0, TARGET_ROW - patch_radius)
             row_end = min(150, TARGET_ROW + patch_radius + 1)
             col_start = max(0, TARGET_COL - patch_radius)
             col_end = min(150, TARGET_COL + patch_radius + 1)
             patch = c_matrix_np_single[row_start:row_end, col_start:col_end]

             # Паддинг, если патч на границе
             if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                 padded_patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                 # Рассчитываем смещения для вставки патча в центр паддинга
                 r_offset = patch_radius - (TARGET_ROW - row_start)
                 c_offset = patch_radius - (TARGET_COL - col_start)
                 # Проверяем границы перед вставкой
                 if r_offset >= 0 and c_offset >= 0 and r_offset + patch.shape[0] <= PATCH_SIZE and c_offset + patch.shape[1] <= PATCH_SIZE:
                     padded_patch[r_offset:r_offset+patch.shape[0], c_offset:c_offset+patch.shape[1]] = patch
                 else:
                      logging.warning(f"Ошибка паддинга для элемента {i+idx_in_batch}, патч пропущен.")
                      Y_sr_list.pop() # Удаляем последний добавленный Y, так как X не будет добавлен
                      continue # Пропускаем добавление этого патча
                 patch = padded_patch

             # Вытягиваем патч в 1D вектор и добавляем в X_sr_list
             X_sr_list.append(patch.flatten())

    X_sr = np.array(X_sr_list)
    Y_sr = np.array(Y_sr_list)

    if X_sr.size == 0 or Y_sr.size == 0: # Проверка на пустые массивы
         raise ValueError("Массивы X_sr или Y_sr пусты после генерации данных.")

    logging.info(f"Данные для SR сгенерированы. Форма X_sr: {X_sr.shape}, Форма Y_sr: {Y_sr.shape}")

except ValueError as e:
    logging.error(f"Ошибка значения при генерации данных для SR: {e}")
    X_sr = np.array([]) # Устанавливаем пустой массив в случае ошибки
    Y_sr = np.array([])
except FileNotFoundError as e:
    logging.error(f"Ошибка FileNotFoundError при генерации данных для SR: {e}")
    X_sr = np.array([])
    Y_sr = np.array([])
except Exception as e:
    logging.error(f"Непредвиденная ошибка при генерации данных для SR: {e}")
    X_sr = np.array([])
    Y_sr = np.array([])


# Запуск PySR 

if X_sr.ndim == 2 and Y_sr.ndim == 1 and X_sr.shape[0] > 0 and X_sr.shape[0] == Y_sr.shape[0]:
    logging.info("Запуск PySR...")
    try:
        # Генерация имен переменных для патча
        variable_names = [f"p{r}{c}" for r in range(PATCH_SIZE) for c in range(PATCH_SIZE)]
        if len(variable_names) != X_sr.shape[1]:
             raise ValueError(f"Количество имен переменных ({len(variable_names)}) не совпадает с количеством признаков в X_sr ({X_sr.shape[1]})")

        model_sr = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["cos", "exp", "log", "sin", "sqrt", "abs", "tanh", 'inv(x) = 1/x'],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            variable_names=variable_names,
            progress=True,
            model_selection="best"
        )

        model_sr.fit(X_sr, Y_sr)

        print("--- Результаты PySR ---")
        # Вывод лучшего уравнения
        try:
             print("Лучшая формула (SymPy):")
             print(model_sr.sympy())
        except ImportError:
             print("Установите sympy для красивого вывода формулы: pip install sympy")
             print(f"Лучшая формула (строка): {model_sr.latex()}") # Или выведем LaTeX

    except ImportError:
         logging.error("Библиотека PySR не найдена. Установите её: pip install -U pysr")
    except ValueError as e:
         logging.error(f"Ошибка значения при запуске PySR: {e}")
    except Exception as e:
         logging.error(f"Ошибка во время работы PySR: {e}")
         logging.error("Убедитесь, что Julia установлена и PySR настроен корректно.")

else:
    logging.error("Не удалось запустить PySR: входные данные X_sr или Y_sr некорректны или пусты.")
