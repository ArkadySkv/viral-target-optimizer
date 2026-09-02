import json
import os

def load_virus_data(filepath="data/target_virus_proteins.json"):
    """
    Загружает данные о вирусных мишенях из JSON-файла.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл данных не найден по адресу: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("virus_targets", [])

def save_optimization_results(results, filepath="data/optimization_results.json"):
    """
    Сохраняет результаты оптимизации в JSON-файл.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    output_data = {
        "metadata": {
            "algorithm": "Random Search (MSE Optimization)",
            "status": "Completed"
        },
        "results": results
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
