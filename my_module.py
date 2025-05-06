import random
from collections import Counter
import numpy as np
import os
import json
import math

import random
from collections import Counter
import math
import numpy as np 

def generate_counters_and_interactions(
    n_bins=150,
    total=7500,
    types=range(1, 17), # Ожидаем типы от 1 до N
    segment_min=20,
    segment_max=50,
    dominant_fraction_range=(0.2, 0.7),
    dominant_count_range=(1, 3),
    min_coverage=0.9,
    # Параметры для флуктуации типа 1
    type1_spike_prob=0.08,
    type1_high_range=(3000, 4500),
    type1_low_range=(0, 300),
    # Параметры контроля сегментов
    dominant_type_cap=3500,
    subsegment_variation=0.2,
    localization_penalty=0.1,
    # Параметры матрицы взаимодействий
    self_interaction_value=0.28, # Притяжение одноименных (кроме типа 1)
    intra_segment_attraction=0.22, # Притяжение доминантов внутри сегмента
    inter_segment_repulsion=-0.18 # Отталкивание доминантов из разных сегментов
):
    """
    Генерирует список счетчиков и матрицу взаимодействий на основе сегментации.

    Args:
        # ... (аргументы из предыдущей версии) ...
        self_interaction_value (float): Значение на диагонали матрицы (кроме типа 1).
        intra_segment_attraction (float): Значение для притягивающихся пар (доминанты в одном сегменте).
        inter_segment_repulsion (float): Значение для отталкивающихся пар (доминанты в разных сегментах).


    Returns:
        tuple(list[Counter], np.ndarray):
            Кортеж, содержащий:
            - Список объектов Counter для каждого бина.
            - Матрицу взаимодействий NumPy (размер max(types) x max(types)).
    """
    counters = [Counter() for _ in range(n_bins)]
    type1 = 1.0
    # Убедимся, что типы - это список целых чисел для индексации
    all_types_list = sorted(list(set(int(t) for t in types)))
    max_type = max(all_types_list) if all_types_list else 0
    other_types = [t for t in all_types_list if t != int(type1)]


    if not other_types:
        raise ValueError("Требуется хотя бы один тип, отличный от типа 1.")
    if int(type1) not in all_types_list:
         raise ValueError("Тип 1 должен быть в списке 'types'.")

    # --- Инициализация матрицы взаимодействий ---
    # Размер N+1, чтобы использовать индексы 1..N напрямую
    matrix_size = max_type + 1
    interaction_matrix = np.zeros((matrix_size, matrix_size), dtype=float)

    # Заполнение диагонали
    for t in other_types:
        interaction_matrix[t, t] = self_interaction_value
    # Тип 1 не взаимодействует сам с собой (остается 0)

    # --- Шаг 1: Распределение типа 1 (без изменений) ---
    idx_type1 = int(type1)
    for i in range(n_bins):
        count1 = 0
        if random.random() < type1_spike_prob:
            count1 = random.randint(type1_high_range[0], type1_high_range[1])
        else:
            count1 = random.randint(type1_low_range[0], type1_low_range[1])
        # Используем float ключ в Counter, но int для индексации матрицы
        counters[i][type1] = min(count1, total)

    # --- Шаг 2: Расчет оставшихся сумм (без изменений) ---
    remaining_totals = [max(0, total - counters[i][type1]) for i in range(n_bins)]

    # --- Шаг 3: Определение сегментов (без изменений) ---
    assigned_bins = set()
    bin_indices = list(range(n_bins))
    segments = [] # Список списков индексов бинов
    covered = 0
    max_attempts = n_bins * 5
    attempts = 0
    while covered < int(n_bins * min_coverage) and attempts < max_attempts:
        attempts += 1
        seg_len = random.randint(segment_min, segment_max)
        possible_starts = [i for i in bin_indices if i not in assigned_bins]
        if not possible_starts: break
        start = random.choice(possible_starts)
        end = min(start + seg_len, n_bins)
        segment_indices = list(range(start, end))
        if any(i in assigned_bins for i in segment_indices): continue
        segments.append(segment_indices)
        assigned_bins.update(segment_indices)
        covered += len(segment_indices)

    # --- Шаг 4: Распределение других типов и ЗАПОЛНЕНИЕ МАТРИЦЫ ---
    all_dominant_types_ever = set()
    segment_dominant_map = {} # Карта: индекс сегмента -> {доминантные типы (int)}

    # Определяем доминантные типы УНИКАЛЬНО для каждого сегмента
    available_dominants_pool = list(other_types) # Используем копию
    random.shuffle(available_dominants_pool) # Перемешиваем для случайности выбора

    for i, segment in enumerate(segments):
        current_dominants = set()
        # Определяем, сколько доминантов можем взять
        num_to_take = random.randint(dominant_count_range[0], dominant_count_range[1])
        num_can_take = min(num_to_take, len(available_dominants_pool)) # Не больше, чем осталось

        if num_can_take > 0:
             # Берем уникальные типы из начала перемешанного пула
             chosen_dominants = available_dominants_pool[:num_can_take]
             current_dominants = set(chosen_dominants)
             # Убираем выбранные из пула
             available_dominants_pool = available_dominants_pool[num_can_take:]

        segment_dominant_map[i] = current_dominants
        all_dominant_types_ever.update(current_dominants)

        # --- Заполнение матрицы: Притяжение внутри сегмента ---
        dominant_list = list(current_dominants)
        for j1 in range(len(dominant_list)):
            for j2 in range(j1 + 1, len(dominant_list)): # Пары без повторений
                t1 = dominant_list[j1]
                t2 = dominant_list[j2]
                # Устанавливаем симметрично
                interaction_matrix[t1, t2] = intra_segment_attraction
                interaction_matrix[t2, t1] = intra_segment_attraction


    # --- Заполнение матрицы: Отталкивание между сегментами ---
    segment_indices = list(segment_dominant_map.keys())
    for i1 in range(len(segment_indices)):
        for i2 in range(i1 + 1, len(segment_indices)): # Пары разных сегментов
            seg_idx1 = segment_indices[i1]
            seg_idx2 = segment_indices[i2]
            dom1 = segment_dominant_map[seg_idx1]
            dom2 = segment_dominant_map[seg_idx2]

            if not dom1 or not dom2: continue # Пропускаем, если в одном из сегментов нет доминантов

            for t1 in dom1:
                for t2 in dom2:
                    # Отталкивание между доминантами разных сегментов
                    interaction_matrix[t1, t2] = inter_segment_repulsion
                    interaction_matrix[t2, t1] = inter_segment_repulsion


    # --- Распределение частот в бинах ---
    for seg_idx, segment in enumerate(segments):
        dominant_types_int = segment_dominant_map[seg_idx] # int типы
        dominant_types_float = {float(t) for t in dominant_types_int} # float для Counter
        if not dominant_types_int: continue

        split_point = len(segment) // 2
        if len(segment) < 4: split_point = -1

        for i, bin_idx in enumerate(segment):
            remaining_total = remaining_totals[bin_idx]
            if remaining_total <= 0: continue

            subsegment_multiplier = 1.0
            if split_point != -1:
                subsegment_multiplier = (1.0 + subsegment_variation) if i < split_point else (1.0 - subsegment_variation)
                subsegment_multiplier = max(0.1, subsegment_multiplier)

            dom_frac = random.uniform(*dominant_fraction_range) * subsegment_multiplier
            dom_frac = max(0, min(1, dom_frac))
            dom_total_alloc = int(remaining_total * dom_frac)
            excess_from_cap = 0
            allocated_to_dominants = 0

            if dominant_types_float and dom_total_alloc > 0:
                dom_weights = [random.expovariate(1.0) + 1e-6 for _ in dominant_types_float]
                total_weight = sum(dom_weights)

                for t_float, w in zip(dominant_types_float, dom_weights):
                    initial_alloc = int(w / total_weight * dom_total_alloc)
                    capped_alloc = min(initial_alloc, dominant_type_cap)
                    excess_from_cap += (initial_alloc - capped_alloc)
                    counters[bin_idx][t_float] += capped_alloc
                    allocated_to_dominants += capped_alloc

            current_sum_added_in_step4 = allocated_to_dominants
            remaining_for_bg = remaining_total - current_sum_added_in_step4 + excess_from_cap
            remaining_for_bg = max(0, remaining_for_bg)

            background_types_pool_int = list(set(other_types) - dominant_types_int)

            if background_types_pool_int and remaining_for_bg > 0:
                max_bg_count = len(background_types_pool_int)
                min_bg_count = min(1, max_bg_count)
                num_bg_types_to_select = random.randint(min_bg_count, max_bg_count)

                bg_weights = []
                bg_types_selected_int = []
                sampled_pool_int = random.sample(background_types_pool_int, num_bg_types_to_select)

                for t_int in sampled_pool_int:
                     weight = random.expovariate(1.0) + 1e-6
                     # Штраф, если тип доминирует где-то еще (используем int типы)
                     if t_int in all_dominant_types_ever: 
                         weight *= localization_penalty
                     if weight > 1e-7:
                        bg_weights.append(weight)
                        bg_types_selected_int.append(t_int) 

                if bg_types_selected_int:
                    total_bg_weight = sum(bg_weights)
                    if total_bg_weight > 0:
                        norm_bg = [int(w / total_bg_weight * remaining_for_bg) for w in bg_weights]

                        for t_int, c in zip(bg_types_selected_int, norm_bg):
                             if c > 0:
                                counters[bin_idx][float(t_int)] += c

    # --- Шаг 5: Заполнение бинов вне сегментов (с учетом локализации) ---
    for i in range(n_bins):
        if i not in assigned_bins:
            remaining_total = remaining_totals[i]
            if remaining_total <= 0: continue

            if other_types: # other_types содержит int
                max_chosen_count = len(other_types)
                min_chosen_count = min(1, max_chosen_count)
                num_chosen = random.randint(min_chosen_count, max_chosen_count)

                weights = []
                chosen_types_selected_int = []
                sampled_pool_int = random.sample(other_types, num_chosen)
                for t_int in sampled_pool_int:
                    weight = random.expovariate(1.0) + 1e-6
                    # Штраф, если тип доминирует где-то (используем int типы)
                    if t_int in all_dominant_types_ever:
                        weight *= localization_penalty
                    if weight > 1e-7:
                        weights.append(weight)
                        chosen_types_selected_int.append(t_int) # Сохраняем int

                if chosen_types_selected_int:
                    total_weight = sum(weights)
                    if total_weight > 0:
                        norm_counts = [int(w / total_weight * remaining_total) for w in weights]
                        # Добавляем в Counter с float ключом
                        for t_int, c in zip(chosen_types_selected_int, norm_counts):
                             if c > 0:
                                counters[i][float(t_int)] += c

    # --- Шаг 6: Финальная корректировка ---
    other_types_float = {float(t) for t in other_types}
    for i in range(n_bins):
        current_sum = sum(counters[i].values())
        diff = total - current_sum

        if diff > 0:
            # Ищем существующие float типы (кроме 1) для добавления
            adjustable_types = [t for t in other_types_float if t in counters[i]]
            if not adjustable_types: adjustable_types = [type1] if counters[i].get(type1, 0) > 0 else []
            if not adjustable_types:
                # Если нечего увеличивать, берем случайные float типы
                adjustable_types = random.sample(list(other_types_float), k=min(diff, len(other_types_float)))

            for _ in range(diff):
                if not adjustable_types:
                    chosen_type = random.choice(list(other_types_float)) # Берем float
                    counters[i][chosen_type] = counters[i].get(chosen_type, 0) + 1
                else:
                    t_to_increment = random.choice(adjustable_types) # float
                    counters[i][t_to_increment] = counters[i].get(t_to_increment, 0) + 1

        elif diff < 0:
            # Ищем существующие float типы (кроме 1) для вычитания
            adjustable_types = [t for t in other_types_float if counters[i].get(t, 0) > 0]
            if not adjustable_types: adjustable_types = [type1] if counters[i].get(type1, 0) > 0 else []

            for _ in range(abs(diff)):
                if not adjustable_types: break
                t_to_decrement = random.choice(adjustable_types) # float
                counters[i][t_to_decrement] -= 1
                if counters[i][t_to_decrement] <= 0: # Проверяем <= 0 на всякий случай
                    if t_to_decrement in adjustable_types: # Доп. проверка перед удалением
                       adjustable_types.remove(t_to_decrement)
                    del counters[i][t_to_decrement] # Удаляем ключ
                    if not adjustable_types: # Обновляем список, если опустел
                        adjustable_types = [t for t in other_types_float if counters[i].get(t, 0) > 0]
                        if not adjustable_types: adjustable_types = [type1] if counters[i].get(type1, 0) > 0 else []

        # Очистка нулевых значений
        zero_keys = [k for k, v in counters[i].items() if v <= 0]
        for k in zero_keys:
            if k in counters[i]: # Проверка перед удалением
               del counters[i][k]

    return counters, interaction_matrix

def generate_random_ensemble_from_probabilistic_counters(
    bin_counters, n_structures=100, beads_per_bin=15, seed=None
):
    if seed:
        np.random.seed(seed)
    n_bins = len(bin_counters)
    total_beads = n_bins * beads_per_bin
    ensemble = np.zeros((n_structures, total_beads), dtype=float)

    for bin_idx, counter in enumerate(bin_counters):
        values, counts = zip(*counter.items())
        probs = np.array(counts, dtype=float) / sum(counts)  
        for s in range(n_structures):
            sampled = np.random.choice(values, size=beads_per_bin, p=probs)
            ensemble[s, bin_idx * beads_per_bin:(bin_idx + 1) * beads_per_bin] = sampled

    return ensemble
