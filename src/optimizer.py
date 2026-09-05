import numpy as np

def calculate_loss(current_logp, current_mw, target_logp, target_mw):
    error_logp = (current_logp - target_logp) ** 2
    error_mw = ((current_mw - target_mw) / 100) ** 2
    return error_logp + error_mw

def run_scaled_pgd(target_logp, target_mw, learning_rate, eps=1e-5, max_iter=2000):
    """
    Выполняет оптимизацию в нормированном пространстве состояний: x = [LogP, MW / 100]
    Это полностью устраняет проблему жесткой жесткости и плохой обусловленности.
    """
    min_logp, max_logp = -2.0, 6.0
    min_mw, max_mw = 150.0, 600.0

    # Перевод начального состояния в масштабированные координаты [LogP, MW/100]
    current_state = np.array([
        (min_logp + max_logp) / 2.0,
        ((min_mw + max_mw) / 2.0) / 100.0
    ])

    # Вектор целей в масштабированных координатах
    target = np.array([target_logp, target_mw / 100.0])
    
    # Ограничения множества S в масштабированных координатах
    min_bounds = np.array([min_logp, min_mw / 100.0])
    max_bounds = np.array([max_logp, max_mw / 100.0])

    iterations_taken = 0
    for i in range(max_iter):
        # В нормированном пространстве f(x) = ||x - t||^2, следовательно grad = 2 * (x - t)
        grad = 2 * (current_state - target)
        
        if np.linalg.norm(grad) < eps:
            break
            
        intermediate_state = current_state - learning_rate * grad
        next_state = np.clip(intermediate_state, min_bounds, max_bounds)
        
        if np.linalg.norm(next_state - current_state) < eps:
            break
            
        current_state = next_state
        iterations_taken += 1

    # Обратное декодирование в реальные физические величины перед возвратом
    final_state = current_state.copy()
    final_state[1] = final_state[1] * 100.0
    
    return final_state, iterations_taken

def optimize_parameters(target_logp, target_mw):
    # 1. Теоретический шаг на основе константы Липшица (L = 2.0 -> lr = 1/L = 0.5)
    # В нормированном пространстве этот метод сойдется ровно за 1 итерацию!
    theoretical_lr = 0.5
    state_theory, steps_theory = run_scaled_pgd(target_logp, target_mw, learning_rate=theoretical_lr)
    loss_theory = calculate_loss(state_theory[0], state_theory[1], target_logp, target_mw)

    # 2. Эвристический подход с меньшим шагом (например, консервативный шаг 0.05)
    heuristic_lr = 0.05
    state_heuristic, steps_heuristic = run_scaled_pgd(target_logp, target_mw, learning_rate=heuristic_lr)
    
    if steps_heuristic > 0:
        efficiency_gain = ((steps_heuristic - steps_theory) / steps_heuristic) * 100
    else:
        efficiency_gain = 0.0

    metric_summary = {
        "steps_theory": steps_theory,
        "steps_heuristic": steps_heuristic,
        "efficiency_gain_pct": round(efficiency_gain, 1)
    }

    return round(state_theory[0], 2), round(state_theory[1], 1), round(loss_theory, 5), metric_summary
