import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os 

DATA_DIR_GAM = '/content/GAM'                         # На примере GAM
DATA_DIR_DIST = '/content/Distance_matrix'
BEST_MODEL_SAVE_PATH = './unet_hic_to_distance_best.pth' # Лучшая модель
FINAL_MODEL_SAVE_PATH = './unet_hic_to_distance_final.pth' # Финальная модель
PLOTS_DIR = './training_plots/' # Папка для сохранения графиков

LEARNING_RATE = 1e-4 
BATCH_SIZE = 20      
NUM_EPOCHS = 80
VALIDATION_SPLIT = 0.1

if not os.path.exists(os.path.dirname(BEST_MODEL_SAVE_PATH)):
    os.makedirs(os.path.dirname(BEST_MODEL_SAVE_PATH))
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

try:

    full_dataset = GAMDistanceDatasetIndexed(gam_dir=DATA_DIR_GAM, distance_dir=DATA_DIR_DIST)

    n_val = int(len(full_dataset) * VALIDATION_SPLIT)
    n_train = len(full_dataset) - n_val
 
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

except Exception as e:
    print(f"Ошибка при подготовке данных: {e}")
    exit() 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

try:
    model = UNet(n_channels_in=1, n_channels_out=1).to(device) 
except NameError:
     print("Класс UNet (или UNetLatent) не определен.")
     exit()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

train_losses = []
val_losses = []
best_val_loss = float('inf') 

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    num_batches_train = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            print(f"Обнаружен NaN в функции потерь на эпохе {epoch+1}, батче {i+1}. Прерывание обучения.")
            exit()
            
        loss.backward()
        # Добавляем Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        num_batches_train += 1

    avg_train_loss = running_loss / num_batches_train if num_batches_train > 0 else 0
    train_losses.append(avg_train_loss) # Сохраняем для графика
    print(f'--- Конец Эпохи [{epoch+1}/{NUM_EPOCHS}], Средние Тренировочные Потери: {avg_train_loss:.6f} ---')

    model.eval()
    running_val_loss = 0.0
    num_batches_val = 0
    with torch.no_grad():
        for inputs_val, targets_val in val_loader:
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            outputs_val = model(inputs_val)
            val_loss = criterion(outputs_val, targets_val)
            if torch.isnan(val_loss):
                 print(f"Обнаружен NaN в валидационной ошибке на эпохе {epoch+1}. Пропуск батча валидации.")
                 continue # Пропускаем этот батч, но не прерываем все
            running_val_loss += val_loss.item()
            num_batches_val += 1

    avg_val_loss = running_val_loss / num_batches_val if num_batches_val > 0 else float('inf')
    val_losses.append(avg_val_loss) 
    print(f'--- Валидационные Потери после Эпохи [{epoch+1}/{NUM_EPOCHS}]: {avg_val_loss:.6f} ---')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        try:
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f'*** Найдена лучшая модель! Сохранена в: {BEST_MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.6f}) ***')
        except Exception as e:
            print(f"Ошибка при сохранении лучшей модели: {e}")

try:
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print(f"Финальная модель сохранена в: {FINAL_MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Ошибка при сохранении финальной модели: {e}")

end_time = time.time()
print(f"Обучение завершено за {(end_time - start_time)/60:.2f} минут.")
print(f"Лучшая валидационная ошибка: {best_val_loss:.6f}")
