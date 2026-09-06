import numpy as np

def calculate_loss(current_logp, current_mw, target_logp, target_mw):
    error_logp = (current_logp - target_logp) ** 2
    error_mw = ((current_mw - target_mw) / 100) ** 2
    return error_logp + error_mw

def run_scaled_pgd(target_logp, target_mw, learning_rate, method="static", eps=1e-5, max_iter=2000):
    """
    Выполняет условную оптимизацию в нормированном пространстве признаков.
    Поддерживает три режима: 'static' (базовый PGD), 'theory' (L-step) и 'nesterov' (ускорение NAG).
    """
    min_logp, max_logp = -2.0, 6.0
    min_mw, max_mw = 150.0, 600.0

    current_state = np.array([
        (min_logp + max_logp) / 2.0,
        ((min_mw + max_mw) / 2.0) / 100.0
    ])

    target = np.array([target_logp, target_mw / 100.0])
    min_bounds = np.array([min_logp, min_mw / 100.0])
    max_bounds = np.array([max_logp, max_mw / 100.0])

    # Переменные для метода Нестерова
    v = np.zeros(2)
    momentum = 0.9
    
    iterations_taken = 0
    
    for i in range(max_iter):
        if method == "nesterov":
            # Шаг экстраполяции Нестерова (взгляд в будущее)
            lookahead_state = current_state + momentum * v
            grad = 2 * (lookahead_state - target)
            
            if np.linalg.norm(grad) < eps:
                break
                
            v_next = momentum * v - learning_rate * grad
            next_state = np.clip(current_state + v_next, min_bounds, max_bounds)
            v = v_next
        else:
            # Стандартный шаг градиентного спуска
            grad = 2 * (current_state - target)
            if np.linalg.norm(grad) < eps:
                break
            intermediate_state = current_state - learning_rate * grad
            next_state = np.clip(intermediate_state, min_bounds, max_bounds)
        
        if np.linalg.norm(next_state - current_state) < eps:
            break
            
        current_state = next_state
        iterations_taken += 1

    # Исправлено декодирование масштаба: умножаем на 100 только координату MW
    final_state = current_state.copy()
    final_state[1] = final_state[1] * 100.0
    return final_state, iterations_taken

def optimize_parameters(target_logp, target_mw):
    # 1. Теоретический оптимальный шаг (lr = 1/L = 0.5) -> Сходимость за 1 шаг
    state_theory, steps_theory = run_scaled_pgd(target_logp, target_mw, learning_rate=0.5, method="theory")
    loss_theory = calculate_loss(state_theory[0], state_theory[1], target_logp, target_mw)

    # 2. Ускоренный метод Нестерова (lr = 0.05) -> Сходится быстрее статического за счет импульса
    state_nesterov, steps_nesterov = run_scaled_pgd(target_logp, target_mw, learning_rate=0.05, method="nesterov")

    # 3. Базовый статический шаг (lr = 0.05) -> Самый медленный базовый контур
    state_heuristic, steps_heuristic = run_scaled_pgd(target_logp, target_mw, learning_rate=0.05, method="static")
    
    # Исправлено: заменена неопределенная переменная steps_intern на корректную steps_nesterov
    if steps_heuristic > 0:
        nesterov_gain_pct = ((steps_heuristic - steps_nesterov) / steps_heuristic) * 100
    else:
        nesterov_gain_pct = 0.0

    metric_summary = {
        "steps_theory": steps_theory,
        "steps_nesterov": steps_nesterov,
        "steps_heuristic": steps_heuristic,
        "nesterov_gain_pct": round(nesterov_gain_pct, 1)
    }

    # Исправлено: убран ошибочный вызов словаря как функции
    return round(state_theory[0], 2), round(state_theory[1], 1), round(loss_theory, 5), metric_summary
