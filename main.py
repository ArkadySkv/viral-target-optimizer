from src.loader import load_virus_data, save_optimization_results
from src.optimizer import optimize_parameters
from tabulate import tabulate

def main():
    try:
        targets = load_virus_data()
        print(f"Успешно загружено мишеней из базы данных: {len(targets)}\n")
        
        results_table = []
        json_results_log = []
        
        for target in targets:
            virus = target["virus_name"]
            protein = target["target_protein"]
            t_logp = target["ideal_inhibitor_properties"]["target_logp"]
            t_mw = target["ideal_inhibitor_properties"]["target_mw"]
            
            opt_logp, opt_mw, loss, metrics = optimize_parameters(t_logp, t_mw)
            
            # Наглядное сопоставление трех подходов оптимизации
            steps_comparison = f"L-Step: {metrics['steps_theory']} | NAG: {metrics['steps_nesterov']} | Static: {metrics['steps_heuristic']}"
            acceleration = f"NAG Speedup: +{metrics['nesterov_gain_pct']}%"
            
            results_table.append([
                virus, protein, f"{t_logp} / {t_mw}", f"{opt_logp} / {opt_mw}", steps_comparison, acceleration
            ])
            
            json_results_log.append({
                "virus_name": virus,
                "target_protein": protein,
                "benchmarks": metrics,
                "final_loss": loss
            })
            
        headers = ["Вирус", "Белок-мишень", "Цель (LogP/MW)", "Найдено (LogP/MW)", "Итерации (L-Step vs NAG vs Static)", "Эффективность импульса"]
        print(tabulate(results_table, headers=headers, tablefmt="grid"))
        
        save_optimization_results(json_results_log)
        print(f"\nВычислительный бенчмарк успешно сохранен.")
        
    except Exception as e:
        print(f"Ошибка выполнения скрипта: {e}")

if __name__ == "__main__":
    main()