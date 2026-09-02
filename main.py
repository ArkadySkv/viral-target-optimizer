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
            
            # Запуск математического поиска
            opt_logp, opt_mw, loss = optimize_parameters(t_logp, t_mw)
            
            # Формирование строки для вывода в консоль
            results_table.append([
                virus, protein, f"{t_logp} / {t_mw}", f"{opt_logp} / {opt_mw}", loss
            ])
            
            # Формирование структуры для экспорта в JSON
            json_results_log.append({
                "virus_name": virus,
                "target_protein": protein,
                "target_properties": {
                    "logp": t_logp,
                    "mw": t_mw
                },
                "optimized_properties": {
                    "logp": opt_logp,
                    "mw": opt_mw
                },
                "final_loss": loss
            })
            
        # Вывод таблицы в терминал
        headers = ["Вирус", "Белок-мишень", "Цель (LogP/MW)", "Найдено (LogP/MW)", "Ошибка (Loss)"]
        print(tabulate(results_table, headers=headers, tablefmt="grid"))
        
        # Сохранение результатов в файл
        output_path = "data/optimization_results.json"
        save_optimization_results(json_results_log, filepath=output_path)
        print(f"\nРезультаты расчетов успешно сохранены в файл: {output_path}")
        
    except Exception as e:
        print(f"Ошибка выполнения скрипта: {e}")

if __name__ == "__main__":
    main()
