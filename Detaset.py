import os
import re 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseMatrixDatasetIndexed(Dataset):
    """
    Базовый класс Dataset для загрузки пар матриц из txt файлов
    с именами вида 'prefix_index.txt'.
    """
    def __init__(self, input_dir, target_dir, input_prefix, target_prefix, transform=None):
        """
        Args:
            input_dir (string): Директория с входными матрицами (X).
            target_dir (string): Директория с целевыми матрицами (Y, distance).
            input_prefix (string): Префикс для имен входных файлов (например, "hic_").
            target_prefix (string): Префикс для имен целевых файлов (например, "distance_").
            transform (callable, optional): Опциональные преобразования для сэмплов.
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_prefix = input_prefix
        self.target_prefix = target_prefix
        self.transform = transform
        self.indices = [] 

        try:
            input_indices = set()
            target_indices = set()

            input_pattern = re.compile(rf"^{re.escape(input_prefix)}(\d+)\.txt$")
            target_pattern = re.compile(rf"^{re.escape(target_prefix)}(\d+)\.txt$")

            for filename in os.listdir(input_dir):
                match = input_pattern.match(filename)
                if match:
                    try:
                        index = int(match.group(1)) # Числовая часть
                        input_indices.add(index)
                    except ValueError:
                        logging.warning(f"Не удалось извлечь числовой индекс из файла: {filename} в {input_dir}")
                    except IndexError:
                         logging.warning(f"Не найдена группа индекса в имени файла: {filename} в {input_dir}")


            # Ищем индексы в целевой директории
            if not os.path.isdir(target_dir):
                 raise FileNotFoundError(f"Директория не найдена: {target_dir}")
            for filename in os.listdir(target_dir):
                match = target_pattern.match(filename)
                if match:
                    try:
                        index = int(match.group(1)) 
                        target_indices.add(index)
                    except ValueError:
                        logging.warning(f"Не удалось извлечь числовой индекс из файла: {filename} в {target_dir}")
                    except IndexError:
                         logging.warning(f"Не найдена группа индекса в имени файла: {filename} в {target_dir}")

            # Находим общие индексы и сортируем их
            common_indices = sorted(list(input_indices & target_indices))

            if not common_indices:
                raise ValueError(f"Не найдено файлов с совпадающими индексами для префиксов '{input_prefix}' и '{target_prefix}' в директориях {input_dir} и {target_dir}")

            self.indices = common_indices
            logging.info(f"Найдено {len(self.indices)} файлов с совпадающими индексами.")

        except FileNotFoundError as e:
            logging.error(e)
            raise
        except ValueError as e:
            logging.error(e)
            raise
        except Exception as e:
            logging.error(f"Произошла непредвиденная ошибка при инициализации Dataset: {e}")
            raise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        numerical_index = self.indices[idx]

        input_filename = f"{self.input_prefix}{numerical_index}.txt"
        target_filename = f"{self.target_prefix}{numerical_index}.txt"

        input_path = os.path.join(self.input_dir, input_filename)
        target_path = os.path.join(self.target_dir, target_filename)

        try:

            input_matrix = np.loadtxt(input_path, dtype=np.float32)
            if self.input_dir == '/content/Hi-C_Balanced' or self.input_dir == '/content/SPRITE_Balanced':       # Hi-C и SPRITE мы сразу преобразовываем log1p + зануляем NaN`ы
              input_matrix = np.log1p(np.nan_to_num(input_matrix))
              np.fill_diagonal(input_matrix, 0)                                     # На самом деле на диагонали и так уже нули, но на всякий случай =)
             
            target_matrix = np.loadtxt(target_path, dtype=np.float32)
            np.fill_diagonal(target_matrix, 0)                         # В матрице дистанций тоже зануляем диагональ (изначально она ненулевая в силу особенностей её вычисления)


            # Проверка размерности
            if input_matrix.shape != (150, 150) or target_matrix.shape != (150, 150):
                raise ValueError(f"Матрицы в файлах {input_filename}/{target_filename} имеют неверную размерность: input{input_matrix.shape}, target{target_matrix.shape}.")

            input_tensor = torch.from_numpy(input_matrix).unsqueeze(0)
            target_tensor = torch.from_numpy(target_matrix).unsqueeze(0)

            sample = {'input': input_tensor, 'target': target_tensor}

            if self.transform:
                sample = self.transform(sample)

            return sample['input'], sample['target']

        except ValueError as e:
             logging.error(f"Ошибка значения при загрузке/обработке индекса {numerical_index} (файлы {input_filename}/{target_filename}): {e}")
             raise
        except FileNotFoundError:
             logging.error(f"Файл не найден при попытке загрузки индекса {numerical_index} (ожидались {input_filename}/{target_filename})")
             raise
        except Exception as e:
             logging.error(f"Ошибка при загрузке/обработке индекса {numerical_index} (файлы {input_filename}/{target_filename}): {e}")
             raise


# Теперь создаем специфичные классы, передавая префиксы

class HiCDistanceDatasetIndexed(BaseMatrixDatasetIndexed):
    def __init__(self, hic_dir, distance_dir, transform=None):
        super().__init__(input_dir=hic_dir, target_dir=distance_dir,
                         input_prefix="hic_", target_prefix="distance_", 
                         transform=transform)

class GAMDistanceDatasetIndexed(BaseMatrixDatasetIndexed):
    def __init__(self, gam_dir, distance_dir, transform=None):
        super().__init__(input_dir=gam_dir, target_dir=distance_dir,
                         input_prefix="gam_", target_prefix="distance_", 
                         transform=transform)

class SPRITEDistanceDatasetIndexed(BaseMatrixDatasetIndexed):
    def __init__(self, sprite_dir, distance_dir, transform=None):
        super().__init__(input_dir=sprite_dir, target_dir=distance_dir,
                         input_prefix="sprite_", target_prefix="distance_", 
                         transform=transform)

