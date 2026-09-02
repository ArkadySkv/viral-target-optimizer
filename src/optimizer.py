import numpy as np

def calculate_loss(current_logp, current_mw, target_logp, target_mw):
    """
    Вычисляет среднеквадратичную ошибку между текущими и целевыми параметрами.
    Параметры нормализуются для выравнивания масштабов (MW обычно значительно больше LogP).
    """
    # Нормализация (масштабирование): делим разницу MW на 100, чтобы сбалансировать вес признаков
    error_logp = (current_logp - target_logp) ** 2
    error_mw = ((current_mw - target_mw) / 100) ** 2
    return error_logp + error_mw

def optimize_parameters(target_logp, target_mw, iterations=5000):
    """
    Поиск оптимальных параметров методом случайного поиска (Random Search).
    Возвращает лучшие найденные параметры и значение функции потерь.
    """
    best_loss = float('inf')
    best_logp = 0.0
    best_mw = 0.0

    # Пространство поиска для хемоинформатического скрининга
    min_logp, max_logp = -2.0, 6.0
    min_mw, max_mw = 150.0, 600.0

    for _ in range(iterations):
        # Генерация случайного кандидата
        candidate_logp = np.random.uniform(min_logp, max_logp)
        candidate_mw = np.random.uniform(min_mw, max_mw)
        
        current_loss = calculate_loss(candidate_logp, candidate_mw, target_logp, target_mw)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_logp = candidate_logp
            best_mw = candidate_mw

    return round(best_logp, 2), round(best_mw, 1), round(best_loss, 5)
