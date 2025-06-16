import json
import os
import time
import logging
import random
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
import statsmodels.api as sm  # type: ignore
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm  # type: ignore
import re
import datetime
from typing import Dict, List, Tuple, Any, Union

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Parametry eksperymentu
PROMPTS_FILE = "integrated_prompts.json"
QUESTIONNAIRES_FILE = "questionnaires.json"
BASE_OUTPUT_DIR = "wyniki_zintegrowane"  # Zmieniono na bazowy katalog wyjściowy
REPEATS_PER_CONDITION = 1
INDEPENDENT_QUESTIONS = [1, 3, 5, 7, 9, 10, 13, 15, 18, 20, 22, 24, 25, 27, 29]
INTERDEPENDENT_QUESTIONS = [2, 4, 6, 8, 11, 12, 14, 16, 17, 19, 21, 23, 26, 28, 30]

# Warunki eksperymentalne
LANGUAGES = ["polski", "angielski"]
IDENTITIES = ["amerykanska", "polska", "neutralna", "japonska"]
PROMPT_STRENGTHS = ["weak", "strong"]
CONDITIONS = [f"{lang}_{identity}_{strength}"
              for lang in LANGUAGES
              for identity in IDENTITIES
              for strength in PROMPT_STRENGTHS]

MODELS_TO_TEST = ["gpt-4.1-mini"]  # Lista modeli do przetestowania
MAX_RETRIES = 5
RETRY_DELAY = 10

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicjalizacja klienta OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("Nie znaleziono klucza API OpenAI")


def load_json_file(file_path: str) -> Dict:
    """Ładuje dane z pliku JSON."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Dodana funkcja serializująca
def custom_json_serializer(obj: Any) -> Any:
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None  # lub str(obj) jeśli preferujesz tekst
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    logger.warning(f"Object of type {type(obj)} with value {obj} is not JSON serializable directly.")
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_json_file(data: Dict, file_path: str) -> None:
    """Zapisuje dane do pliku JSON używając niestandardowego serializatora."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=custom_json_serializer)


def create_prompt(condition: str) -> str:
    """Tworzy pełny prompt dla danego warunku."""
    prompts = load_json_file(PROMPTS_FILE)
    questionnaires = load_json_file(QUESTIONNAIRES_FILE)

    language, identity, strength = condition.split('_')
    prompt_key = f"{language}_{identity}_{strength}"

    if prompt_key not in prompts:
        raise ValueError(f"Nie znaleziono promptu dla warunku: {prompt_key}")

    prompt_text = prompts[prompt_key]

    if language == "polski":
        questionnaire = questionnaires["pl"]
        instruction = questionnaires["instruction_pl"]
    else:
        questionnaire = questionnaires["en"]
        instruction = questionnaires["instruction_en"]

    return prompt_text + questionnaire + instruction


def call_openai_api(prompt: str, model_to_use: str) -> Dict[str, int]:
    """Wykonuje wywołanie API OpenAI dla określonego modelu i przetwarza odpowiedź."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Wysyłanie zapytania do modelu: {model_to_use}, próba {attempt + 1}")
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )

            model_response = response.choices[0].message.content

            json_match = re.search(r'```json\s*(.*?)\s*```', model_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = model_response.strip()

            try:
                answers = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Błąd dekodowania JSON z odpowiedzi modelu {model_to_use}: {e}. Odpowiedź: '{model_response}'")
                raise ValueError(f"Nieprawidłowy format JSON w odpowiedzi od {model_to_use}")

            if not all(str(q) in answers for q in range(1, 31)):
                logger.warning(
                    f"Niepełna odpowiedź od modelu {model_to_use}. Brakujące pytania: {[q for q in range(1, 31) if str(q) not in answers]}")
                raise ValueError(f"Niepełna odpowiedź od modelu {model_to_use}")

            validated_answers = {}
            for q_str, val_str in answers.items():
                try:
                    q_num = int(q_str)  # Klucze powinny być stringami, ale na wszelki wypadek
                    val_num = int(val_str)
                    if 1 <= val_num <= 7:
                        validated_answers[str(q_num)] = val_num  # Upewnijmy się, że klucze to stringi
                    else:
                        logger.warning(
                            f"Wartość {val_num} poza zakresem (1-7) dla pytania {q_num} od modelu {model_to_use}")
                        raise ValueError(f"Wartość poza zakresem dla pytania {q_num} od modelu {model_to_use}")
                except ValueError:
                    logger.warning(
                        f"Nieprawidłowa wartość '{val_str}' lub klucz '{q_str}' w odpowiedzi od modelu {model_to_use}")
                    raise ValueError(f"Nieprawidłowa wartość w odpowiedzi od modelu {model_to_use}")

            return validated_answers

        except Exception as e:
            logger.warning(f"Błąd API dla modelu {model_to_use} (próba {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                delay_time = RETRY_DELAY * (2 ** attempt)
                logger.info(f"Oczekiwanie {delay_time}s przed kolejną próbą.")
                time.sleep(delay_time)
            else:
                logger.error(f"Osiągnięto maksymalną liczbę prób dla modelu {model_to_use} dla tego promptu.")
                raise


def calculate_scores(responses: Dict[str, int]) -> Tuple[float, float]:
    """Oblicza wyniki dla konstruktów niezależnego i współzależnego."""
    independent_score = sum(responses[str(q)] for q in INDEPENDENT_QUESTIONS) / len(INDEPENDENT_QUESTIONS)
    interdependent_score = sum(responses[str(q)] for q in INTERDEPENDENT_QUESTIONS) / len(INTERDEPENDENT_QUESTIONS)
    return independent_score, interdependent_score


def collect_data(model_to_use: str, current_output_dir: str,
                 repeats_per_condition: int = REPEATS_PER_CONDITION) -> pd.DataFrame:
    """Zbiera dane dla wszystkich warunków eksperymentalnych dla danego modelu."""
    results = []

    raw_data_dir = os.path.join(current_output_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    partial_results_path = os.path.join(current_output_dir, "partial_results.csv")

    if os.path.exists(partial_results_path):
        try:
            partial_df = pd.read_csv(partial_results_path)
            # Filtrujemy tylko wyniki dla bieżącego modelu, jeśli plik był współdzielony
            # Jednak przy zapisie do model_specific_output_dir, to nie jest konieczne.
            # Zakładamy, że partial_results.csv jest specyficzny dla modelu w tym katalogu.
            results = partial_df.to_dict('records')
            logger.info(
                f"Wczytano {len(results)} częściowych wyników dla modelu {model_to_use} z {partial_results_path}")
        except Exception as e:
            logger.warning(f"Nie udało się wczytać częściowych wyników dla modelu {model_to_use}: {e}")

    condition_counts = {}
    for result in results:
        # Upewniamy się, że liczymy tylko powtórzenia dla bieżącego modelu
        if result.get("model") == model_to_use:
            cond = result["condition"]
            condition_counts[cond] = condition_counts.get(cond, 0) + 1

    for condition in CONDITIONS:
        completed = condition_counts.get(condition, 0)
        remaining = repeats_per_condition - completed

        if remaining <= 0:
            logger.info(
                f"Model {model_to_use}, Warunek {condition}: wszystkie {repeats_per_condition} powtórzeń już wykonane.")
            continue

        language, identity, strength = condition.split('_')
        logger.info(f"Model {model_to_use}, Warunek {condition}: pozostało {remaining} powtórzeń")

        for repeat in tqdm(range(completed + 1, repeats_per_condition + 1),
                           desc=f"Model {model_to_use}, Warunek {condition}"):
            prompt = create_prompt(condition)
            try:
                if repeat == 1:  # Zapisz prompt tylko raz dla danego warunku (niezależnie od modelu, bo prompt jest ten sam)
                    base_raw_data_dir = os.path.join(BASE_OUTPUT_DIR, "raw_data_prompts")  # Osobny folder na prompty
                    os.makedirs(base_raw_data_dir, exist_ok=True)
                    prompt_filename = f"{condition}_sample_prompt.txt"
                    prompt_path = os.path.join(base_raw_data_dir, prompt_filename)
                    if not os.path.exists(prompt_path):  # Zapisz tylko jeśli jeszcze nie istnieje
                        with open(prompt_path, "w", encoding="utf-8") as f:
                            f.write(prompt)

                responses = call_openai_api(prompt, model_to_use)

                response_filename = f"{condition}_repeat{repeat}_response.json"  # Nazwa pliku nie musi zawierać modelu, bo jest w model_specific_output_dir
                with open(os.path.join(raw_data_dir, response_filename), "w", encoding="utf-8") as f:
                    json.dump(responses, f, indent=2)

                independent_score, interdependent_score = calculate_scores(responses)

                result_entry = {
                    "model": model_to_use,  # Dodajemy informację o modelu
                    "condition": condition,
                    "language": language,
                    "identity": identity,
                    "strength": strength,
                    "repeat": repeat,
                    "independent_score": independent_score,
                    "interdependent_score": interdependent_score,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                for q, resp_val in responses.items():
                    result_entry[f"q{q}"] = resp_val
                results.append(result_entry)

                if len(results) > 0 and (
                        repeat % 10 == 0 or repeat == repeats_per_condition):  # Sprawdzenie czy results nie jest puste
                    current_df = pd.DataFrame(results)
                    current_df.to_csv(partial_results_path, index=False)

                time.sleep(random.uniform(1, 3))
            except Exception as e:
                logger.error(f"Błąd (Model {model_to_use}, Warunek {condition}, Powtórzenie {repeat}): {e}")

    if results:
        final_df = pd.DataFrame(results)
        # Filtrujemy jeszcze raz, aby upewnić się, że mamy tylko dane dla bieżącego modelu
        final_df = final_df[final_df['model'] == model_to_use]
        if not final_df.empty:
            csv_path = os.path.join(current_output_dir, "results.csv")
            final_df.to_csv(csv_path, index=False)
            logger.info(f"Zapisano dane dla modelu {model_to_use} do {csv_path}")
            return final_df

    logger.warning(f"Nie zebrano żadnych nowych wyników dla modelu {model_to_use}.")
    return pd.DataFrame()


def perform_three_way_anova(data: pd.DataFrame, dependent_var: str) -> Dict[str, Any]:
    """Przeprowadza trzyczynnikową analizę wariancji."""
    if data.empty or data[dependent_var].isnull().all():  # Dodano sprawdzenie na puste lub same NaN
        logger.warning(f"Brak danych lub same wartości NaN dla zmiennej {dependent_var} do analizy ANOVA.")
        return {"error": f"Brak danych lub same wartości NaN dla {dependent_var}"}
    if len(data['language'].unique()) < 2 or len(data['identity'].unique()) < 2 or len(data['strength'].unique()) < 2:
        logger.warning(
            f"Niewystarczająca liczba poziomów czynników dla ANOVA dla zmiennej {dependent_var}. Dane: {data[['language', 'identity', 'strength']].nunique()}")
        # Można zwrócić uproszczoną analizę lub błąd
        return {"error": "Niewystarczająca liczba poziomów czynników dla pełnej ANOVA"}

    try:
        formula = f"{dependent_var} ~ C(language) * C(identity) * C(strength)"  # Uproszczony zapis interakcji
        model = sm.formula.ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, type=2)

        results = {
            "model_summary": {
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
                "p_value": float(model.f_pvalue) if model.f_pvalue is not None else None,
            },
            "anova_table": {},
            "effect_sizes": {}
        }

        total_ss = anova_table["sum_sq"].sum()

        for index, row in anova_table.iterrows():
            clean_name = index.replace("C(", "").replace(")", "")
            results["anova_table"][clean_name] = {
                "sum_sq": float(row["sum_sq"]),
                "df": int(row["df"]),
                "f": float(row["F"]) if not np.isnan(row["F"]) else None,
                "p": float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
                "significant": float(row["PR(>F)"]) < 0.05 if not np.isnan(row["PR(>F)"]) else False
            }
            eta_sq = float(row["sum_sq"] / total_ss) if total_ss > 0 else 0
            results["effect_sizes"][clean_name] = {
                "eta_sq": eta_sq,
                "interpretation": "Mały efekt" if eta_sq < 0.06 else "Średni efekt" if eta_sq < 0.14 else "Duży efekt"
            }
        return results
    except Exception as e:
        logger.error(f"Błąd podczas ANOVA dla {dependent_var}: {e}")
        return {"error": str(e)}


def analyze_anglocentrism(data: pd.DataFrame) -> Dict[str, Any]:
    """Przeprowadza analizę anglocentryzmu."""
    if data.empty:
        return {"error": "Brak danych do analizy anglocentryzmu"}

    results = {
        "independent_score": {},
        "interdependent_score": {},
        "euclidean_analysis": {}
    }
    # ... (reszta funkcji analyze_anglocentrism, analyze_single_dimension, analyze_euclidean, summarize_anglocentrism - bez zmian)
    for strength in PROMPT_STRENGTHS:
        strength_data = data[data["strength"] == strength]
        if strength_data.empty: continue

        for language in LANGUAGES:
            lang_strength_data = strength_data[strength_data["language"] == language]
            if lang_strength_data.empty: continue

            neutral_ind = lang_strength_data[lang_strength_data["identity"] == "neutralna"]["independent_score"]
            american_ind = lang_strength_data[lang_strength_data["identity"] == "amerykanska"]["independent_score"]
            polish_ind = lang_strength_data[lang_strength_data["identity"] == "polska"]["independent_score"]
            japanese_ind = lang_strength_data[lang_strength_data["identity"] == "japonska"]["independent_score"]

            neutral_int = lang_strength_data[lang_strength_data["identity"] == "neutralna"]["interdependent_score"]
            american_int = lang_strength_data[lang_strength_data["identity"] == "amerykanska"]["interdependent_score"]
            polish_int = lang_strength_data[lang_strength_data["identity"] == "polska"]["interdependent_score"]
            japanese_int = lang_strength_data[lang_strength_data["identity"] == "japonska"]["interdependent_score"]

            key = f"{language}_{strength}"

            if all(len(x) > 0 for x in [neutral_ind, american_ind, polish_ind, japanese_ind]):
                result_ind = analyze_single_dimension(neutral_ind, american_ind, polish_ind, japanese_ind)
                results["independent_score"][key] = result_ind

            if all(len(x) > 0 for x in [neutral_int, american_int, polish_int, japanese_int]):
                result_int = analyze_single_dimension(neutral_int, american_int, polish_int,
                                                      japanese_int)  # Używamy tej samej funkcji dla obu wymiarów
                results["interdependent_score"][key] = result_int

            if all(len(x) > 0 for x in
                   [neutral_ind, american_ind, polish_ind, japanese_ind, neutral_int, american_int, polish_int,
                    japanese_int]):
                result_eucl = analyze_euclidean(
                    neutral_ind, american_ind, polish_ind, japanese_ind,
                    neutral_int, american_int, polish_int, japanese_int
                )
                results["euclidean_analysis"][key] = result_eucl

    results["overall_conclusion"] = {
        "traditional_analysis": summarize_anglocentrism(results, "traditional"),
        "euclidean_analysis": summarize_anglocentrism(results, "euclidean")
    }
    return results


def analyze_single_dimension(neutral_data: pd.Series, american_data: pd.Series, polish_data: pd.Series,
                             japanese_data: pd.Series) -> Dict[str, Any]:
    """Analizuje anglocentryzm dla pojedynczego wymiaru."""
    # Sprawdzenie czy dane nie są puste
    if any(d.empty for d in [neutral_data, american_data, polish_data, japanese_data]):
        return {"error": "Brak danych dla jednej z grup w analizie pojedynczego wymiaru"}

    neutral_mean = neutral_data.mean()
    american_mean = american_data.mean()
    polish_mean = polish_data.mean()
    japanese_mean = japanese_data.mean()

    diff_american = abs(neutral_mean - american_mean)
    diff_polish = abs(neutral_mean - polish_mean)
    diff_japanese = abs(neutral_mean - japanese_mean)

    diffs = [("amerykanska", diff_american), ("polska", diff_polish), ("japonska", diff_japanese)]
    closest_culture_entry = min(diffs, key=lambda x: x[1]) if diffs else ("N/A", float('inf'))
    closest_culture = closest_culture_entry[0]

    n_bootstrap = 10000
    delta_np_na_list = []
    delta_nj_na_list = []

    # Konwersja na numpy array dla bootstrapu, jeśli nie są puste
    na_arr, am_arr, pl_arr, jp_arr = (d.to_numpy() for d in [neutral_data, american_data, polish_data, japanese_data])

    for _ in range(n_bootstrap):
        boot_neutral = np.random.choice(na_arr, size=len(na_arr), replace=True)
        boot_american = np.random.choice(am_arr, size=len(am_arr), replace=True)
        boot_polish = np.random.choice(pl_arr, size=len(pl_arr), replace=True)
        boot_japanese = np.random.choice(jp_arr, size=len(jp_arr), replace=True)

        boot_diff_american = abs(boot_neutral.mean() - boot_american.mean())
        boot_diff_polish = abs(boot_neutral.mean() - boot_polish.mean())
        boot_diff_japanese = abs(boot_neutral.mean() - boot_japanese.mean())

        delta_np_na_list.append(boot_diff_polish - boot_diff_american)
        delta_nj_na_list.append(boot_diff_japanese - boot_diff_american)

    ci_np_na = np.percentile(delta_np_na_list, [2.5, 97.5]) if delta_np_na_list else [np.nan, np.nan]
    ci_nj_na = np.percentile(delta_nj_na_list, [2.5, 97.5]) if delta_nj_na_list else [np.nan, np.nan]

    sig_vs_polish = ci_np_na[0] > 0 if not np.isnan(ci_np_na[0]) else False
    sig_vs_japanese = ci_nj_na[0] > 0 if not np.isnan(ci_nj_na[0]) else False
    anglocentrism_significant = closest_culture == "amerykanska" and (sig_vs_polish or sig_vs_japanese)

    return {
        "neutral_mean": float(neutral_mean), "american_mean": float(american_mean),
        "polish_mean": float(polish_mean), "japanese_mean": float(japanese_mean),
        "diff_neutral_american": float(diff_american), "diff_neutral_polish": float(diff_polish),
        "diff_neutral_japanese": float(diff_japanese), "closest_to": closest_culture,
        "ci_np_na": ci_np_na.tolist(), "ci_nj_na": ci_nj_na.tolist(),
        "sig_closer_than_polish": sig_vs_polish, "sig_closer_than_japanese": sig_vs_japanese,
        "anglocentrism_present": closest_culture == "amerykanska",
        "anglocentrism_statistically_significant": anglocentrism_significant
    }


def analyze_euclidean(neutral_ind: pd.Series, american_ind: pd.Series, polish_ind: pd.Series, japanese_ind: pd.Series,
                      neutral_int: pd.Series, american_int: pd.Series, polish_int: pd.Series,
                      japanese_int: pd.Series) -> Dict[str, Any]:
    """Analizuje anglocentryzm w przestrzeni 2D."""
    if any(d.empty for d in
           [neutral_ind, american_ind, polish_ind, japanese_ind, neutral_int, american_int, polish_int, japanese_int]):
        return {"error": "Brak danych dla jednej z grup w analizie euklidesowej"}

    neutral_point = (neutral_ind.mean(), neutral_int.mean())
    american_point = (american_ind.mean(), american_int.mean())
    polish_point = (polish_ind.mean(), polish_int.mean())
    japanese_point = (japanese_ind.mean(), japanese_int.mean())

    dist_american = np.sqrt(np.sum(np.square(np.array(neutral_point) - np.array(american_point))))
    dist_polish = np.sqrt(np.sum(np.square(np.array(neutral_point) - np.array(polish_point))))
    dist_japanese = np.sqrt(np.sum(np.square(np.array(neutral_point) - np.array(japanese_point))))

    dists = [("amerykanska", dist_american), ("polska", dist_polish), ("japonska", dist_japanese)]
    closest_culture_entry = min(dists, key=lambda x: x[1]) if dists else ("N/A", float('inf'))
    closest_culture = closest_culture_entry[0]

    n_bootstrap = 10000
    delta_np_na_list = []
    delta_nj_na_list = []

    # Konwersja na numpy array
    ni_arr, ai_arr, pi_arr, ji_arr = (d.to_numpy() for d in [neutral_ind, american_ind, polish_ind, japanese_ind])
    nt_arr, at_arr, pt_arr, jt_arr = (d.to_numpy() for d in [neutral_int, american_int, polish_int, japanese_int])

    for _ in range(n_bootstrap):
        boot_ni, boot_ai, boot_pi, boot_ji = (np.random.choice(arr, size=len(arr), replace=True) for arr in
                                              [ni_arr, ai_arr, pi_arr, ji_arr])
        boot_nt, boot_at, boot_pt, boot_jt = (np.random.choice(arr, size=len(arr), replace=True) for arr in
                                              [nt_arr, at_arr, pt_arr, jt_arr])

        boot_neutral_point = (boot_ni.mean(), boot_nt.mean())
        boot_american_point = (boot_ai.mean(), boot_at.mean())
        boot_polish_point = (boot_pi.mean(), boot_pt.mean())
        boot_japanese_point = (boot_ji.mean(), boot_jt.mean())

        boot_dist_american = np.sqrt(np.sum(np.square(np.array(boot_neutral_point) - np.array(boot_american_point))))
        boot_dist_polish = np.sqrt(np.sum(np.square(np.array(boot_neutral_point) - np.array(boot_polish_point))))
        boot_dist_japanese = np.sqrt(np.sum(np.square(np.array(boot_neutral_point) - np.array(boot_japanese_point))))

        delta_np_na_list.append(boot_dist_polish - boot_dist_american)
        delta_nj_na_list.append(boot_dist_japanese - boot_dist_american)

    ci_np_na = np.percentile(delta_np_na_list, [2.5, 97.5]) if delta_np_na_list else [np.nan, np.nan]
    ci_nj_na = np.percentile(delta_nj_na_list, [2.5, 97.5]) if delta_nj_na_list else [np.nan, np.nan]

    sig_vs_polish = ci_np_na[0] > 0 if not np.isnan(ci_np_na[0]) else False
    sig_vs_japanese = ci_nj_na[0] > 0 if not np.isnan(ci_nj_na[0]) else False
    anglocentrism_significant = closest_culture == "amerykanska" and (sig_vs_polish or sig_vs_japanese)

    return {
        "neutral_point": list(neutral_point), "american_point": list(american_point),
        "polish_point": list(polish_point), "japanese_point": list(japanese_point),
        "dist_neutral_american": float(dist_american), "dist_neutral_polish": float(dist_polish),
        "dist_neutral_japanese": float(dist_japanese), "closest_to": closest_culture,
        "ci_np_na": ci_np_na.tolist(), "ci_nj_na": ci_nj_na.tolist(),
        "sig_closer_than_polish": sig_vs_polish, "sig_closer_than_japanese": sig_vs_japanese,
        "anglocentrism_present": closest_culture == "amerykanska",
        "anglocentrism_statistically_significant": anglocentrism_significant
    }


def summarize_anglocentrism(results: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Podsumowuje wyniki analizy anglocentryzmu."""
    # ... (bez zmian)
    if method == "traditional":
        anglocentrism_count = 0
        significant_count = 0
        total_count = 0

        for construct_key in ["independent_score", "interdependent_score"]:
            construct_results = results.get(construct_key, {})
            for condition_result in construct_results.values():
                if isinstance(condition_result, dict) and "anglocentrism_present" in condition_result:
                    total_count += 1
                    if condition_result["anglocentrism_present"]:
                        anglocentrism_count += 1
                    if condition_result["anglocentrism_statistically_significant"]:
                        significant_count += 1
        return {
            "anglocentrism_present": anglocentrism_count > total_count / 2 if total_count > 0 else False,
            "anglocentrism_statistically_significant": significant_count > 0,
            "proportion_anglocentric": anglocentrism_count / total_count if total_count > 0 else 0.0,
            "proportion_significant": significant_count / total_count if total_count > 0 else 0.0
        }
    elif method == "euclidean":
        anglocentrism_count = 0
        significant_count = 0
        total_count = 0
        euclidean_results = results.get("euclidean_analysis", {})
        for condition_result in euclidean_results.values():
            if isinstance(condition_result, dict) and "anglocentrism_present" in condition_result:
                total_count += 1
                if condition_result["anglocentrism_present"]:
                    anglocentrism_count += 1
                if condition_result["anglocentrism_statistically_significant"]:
                    significant_count += 1
        return {
            "anglocentrism_present": anglocentrism_count > total_count / 2 if total_count > 0 else False,
            "anglocentrism_statistically_significant": significant_count > 0,
            "proportion_anglocentric": anglocentrism_count / total_count if total_count > 0 else 0.0,
            "proportion_significant": significant_count / total_count if total_count > 0 else 0.0
        }
    return {}


def analyze_data(data: pd.DataFrame, current_output_dir: str, model_name: str) -> Dict[str, Any]:
    """Przeprowadza pełną analizę danych dla danego modelu."""
    if data.empty:
        logger.warning(f"Brak danych do analizy dla modelu {model_name}.")
        return {"error": "Brak danych do analizy"}

    results = {
        "model_name": model_name,  # Dodajemy informację o modelu
        "descriptive_stats": {},
        "anova_results": {},
        "anglocentrism_analysis": {},
        "timestamp": datetime.datetime.now().isoformat()
    }

    for construct in ["independent_score", "interdependent_score"]:
        results["descriptive_stats"][construct] = {
            "overall": {
                "mean": float(data[construct].mean()), "std": float(data[construct].std()),
                "min": float(data[construct].min()), "max": float(data[construct].max())
            }
        }
        for factor in ["language", "identity", "strength"]:  # Usunięto 'model' z tej pętli
            results["descriptive_stats"][construct][factor] = {}
            if factor in data.columns:  # Sprawdzenie czy kolumna istnieje
                for value in data[factor].unique():
                    subset = data[data[factor] == value][construct]
                    results["descriptive_stats"][construct][factor][value] = {
                        "mean": float(subset.mean()), "std": float(subset.std()),
                        "n": int(len(subset))
                    }
            else:
                logger.warning(f"Czynnik '{factor}' nie znaleziony w danych dla statystyk opisowych.")

    results["anova_results"]["independent_score"] = perform_three_way_anova(data, "independent_score")
    results["anova_results"]["interdependent_score"] = perform_three_way_anova(data, "interdependent_score")
    results["anglocentrism_analysis"] = analyze_anglocentrism(data)

    json_path = os.path.join(current_output_dir, "analysis_results.json")
    save_json_file(results, json_path)
    logger.info(f"Zapisano wyniki analizy dla modelu {model_name} do {json_path}")
    return results


def run_experiment():
    """Przeprowadza eksperyment dla każdego zdefiniowanego modelu."""
    logger.info("Rozpoczęcie eksperymentu dla wszystkich zdefiniowanych modeli")

    all_models_data = []
    all_models_analysis_results = {}

    # Sprawdzenie plików konfiguracyjnych
    if not os.path.exists(PROMPTS_FILE):
        logger.error(f"Brak pliku {PROMPTS_FILE}")
        return {"error": f"Brak pliku {PROMPTS_FILE}"}
    if not os.path.exists(QUESTIONNAIRES_FILE):
        logger.error(f"Brak pliku {QUESTIONNAIRES_FILE}")
        return {"error": f"Brak pliku {QUESTIONNAIRES_FILE}"}

    for model_name in MODELS_TO_TEST:
        logger.info(f"=== Rozpoczęcie przetwarzania dla modelu: {model_name} ===")
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR,
                                                 model_name.replace("/", "_"))  # Tworzenie bezpiecznej nazwy katalogu
        os.makedirs(model_specific_output_dir, exist_ok=True)

        try:
            logger.info(f"Rozpoczęcie zbierania danych dla modelu: {model_name}")
            model_data = collect_data(model_name, model_specific_output_dir, REPEATS_PER_CONDITION)

            if model_data.empty:
                logger.warning(f"Nie zebrano żadnych danych dla modelu: {model_name}. Pomijanie analizy.")
                all_models_analysis_results[model_name] = {"error": "Brak danych"}
                continue

            all_models_data.append(model_data)

            logger.info(f"Rozpoczęcie analizy danych dla modelu: {model_name}")
            analysis_results = analyze_data(model_data, model_specific_output_dir, model_name)
            all_models_analysis_results[model_name] = analysis_results

            logger.info(f"=== Zakończono przetwarzanie dla modelu: {model_name} ===")

        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania modelu {model_name}: {e}", exc_info=True)
            all_models_analysis_results[model_name] = {"error": str(e)}

    # Opcjonalnie: Połącz wszystkie dane i zapisz zbiorczy plik CSV
    if all_models_data:
        combined_df = pd.concat(all_models_data, ignore_index=True)
        combined_csv_path = os.path.join(BASE_OUTPUT_DIR, "all_models_results.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        logger.info(f"Zapisano połączone dane wszystkich modeli do {combined_csv_path}")

    # Zapisz zbiorcze wyniki analizy (lub podsumowanie)
    overall_summary_path = os.path.join(BASE_OUTPUT_DIR, "all_models_analysis_summary.json")
    save_json_file(all_models_analysis_results, overall_summary_path)
    logger.info(f"Zapisano podsumowanie analiz dla wszystkich modeli do {overall_summary_path}")

    logger.info("Eksperyment dla wszystkich modeli zakończony.")
    return {
        "all_data": all_models_data,  # Lista DataFrame'ów, po jednym dla każdego modelu
        "all_analysis_results": all_models_analysis_results,  # Słownik wyników analizy, kluczem jest nazwa modelu
        "base_output_dir": BASE_OUTPUT_DIR
    }


if __name__ == "__main__":
    result = run_experiment()

    if "error" not in result and result.get("all_analysis_results"):
        print(f"\nEksperyment zakończony sukcesem!")
        for model_name, analysis_res in result["all_analysis_results"].items():
            if "error" not in analysis_res:
                # Sprawdzamy czy istnieje DataFrame dla tego modelu
                model_df_list = [df for df in result.get("all_data", []) if
                                 not df.empty and df['model'].iloc[0] == model_name]
                num_observations = len(model_df_list[0]) if model_df_list else "N/A (brak danych)"
                print(f"  Model: {model_name}, Liczba obserwacji: {num_observations}")
            else:
                print(f"  Model: {model_name}, Wystąpił błąd: {analysis_res['error']}")
        print(f"Wyniki zapisane w katalogu bazowym: {result['base_output_dir']}")
    elif "error" in result:
        print(f"\nEksperyment zakończony z błędem: {result['error']}")
    else:
        print(f"\nEksperyment zakończony, ale nie przetworzono żadnych modeli pomyślnie.")