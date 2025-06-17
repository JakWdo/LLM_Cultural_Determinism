import json
import os
import time
import logging
import random
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import re
import datetime
from typing import Dict, List, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Parametry eksperymentu
PROMPTS_FILE = "integrated_prompts.json"
QUESTIONNAIRES_FILE = "questionnaires.json"
BASE_OUTPUT_DIR = "wyniki_zintegrowane"
REPEATS_PER_CONDITION = 5
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

MODELS_TO_TEST = ["gpt-3.5-turbo", "gpt-4.1-mini"]
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


def custom_json_serializer(obj: Any) -> Any:
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
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
                    q_num = int(q_str)
                    val_num = int(val_str)
                    if 1 <= val_num <= 7:
                        validated_answers[str(q_num)] = val_num
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
            results = partial_df.to_dict('records')
            logger.info(
                f"Wczytano {len(results)} częściowych wyników dla modelu {model_to_use} z {partial_results_path}")
        except Exception as e:
            logger.warning(f"Nie udało się wczytać częściowych wyników dla modelu {model_to_use}: {e}")

    condition_counts = {}
    for result in results:
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
                if repeat == 1:
                    base_raw_data_dir = os.path.join(BASE_OUTPUT_DIR, "raw_data_prompts")
                    os.makedirs(base_raw_data_dir, exist_ok=True)
                    prompt_filename = f"{condition}_sample_prompt.txt"
                    prompt_path = os.path.join(base_raw_data_dir, prompt_filename)
                    if not os.path.exists(prompt_path):
                        with open(prompt_path, "w", encoding="utf-8") as f:
                            f.write(prompt)

                responses = call_openai_api(prompt, model_to_use)

                response_filename = f"{condition}_repeat{repeat}_response.json"
                with open(os.path.join(raw_data_dir, response_filename), "w", encoding="utf-8") as f:
                    json.dump(responses, f, indent=2)

                independent_score, interdependent_score = calculate_scores(responses)

                result_entry = {
                    "model": model_to_use,
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
                        repeat % 10 == 0 or repeat == repeats_per_condition):
                    current_df = pd.DataFrame(results)
                    current_df.to_csv(partial_results_path, index=False)

                time.sleep(random.uniform(1, 3))
            except Exception as e:
                logger.error(f"Błąd (Model {model_to_use}, Warunek {condition}, Powtórzenie {repeat}): {e}")

    if results:
        final_df = pd.DataFrame(results)
        final_df = final_df[final_df['model'] == model_to_use]
        if not final_df.empty:
            csv_path = os.path.join(current_output_dir, "results.csv")
            final_df.to_csv(csv_path, index=False)
            logger.info(f"Zapisano dane dla modelu {model_to_use} do {csv_path}")
            return final_df

    logger.warning(f"Nie zebrano żadnych nowych wyników dla modelu {model_to_use}.")
    return pd.DataFrame()


def perform_three_way_anova_with_eta_squared(data: pd.DataFrame, dependent_var: str) -> Dict[str, Any]:
    """
    METODA 1: Przeprowadza trzyczynnikową analizę wariancji z obliczeniem eta-kwadrat.
    """
    if data.empty or data[dependent_var].isnull().all():
        logger.warning(f"Brak danych lub same wartości NaN dla zmiennej {dependent_var} do analizy ANOVA.")
        return {"error": f"Brak danych lub same wartości NaN dla {dependent_var}"}

    if len(data['language'].unique()) < 2 or len(data['identity'].unique()) < 2 or len(data['strength'].unique()) < 2:
        logger.warning(
            f"Niewystarczająca liczba poziomów czynników dla ANOVA dla zmiennej {dependent_var}.")
        return {"error": "Niewystarczająca liczba poziomów czynników dla pełnej ANOVA"}

    try:
        formula = f"{dependent_var} ~ C(language) * C(identity) * C(strength)"
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
            "effect_sizes": {},
            "interpretation_warning": "UWAGA: Wartości p nie mogą być interpretowane standardowo, "
                                      "ponieważ wszystkie obserwacje pochodzą od jednego modelu AI. "
                                      "Eta-kwadrat służy jedynie jako miara deskryptywna wielkości efektu."
        }

        total_ss = anova_table["sum_sq"].sum()

        # Szczególne zainteresowanie efektem tożsamości
        identity_effect_found = False

        for index, row in anova_table.iterrows():
            clean_name = index.replace("C(", "").replace(")", "")
            eta_sq = float(row["sum_sq"] / total_ss) if total_ss > 0 else 0

            results["anova_table"][clean_name] = {
                "sum_sq": float(row["sum_sq"]),
                "df": int(row["df"]),
                "f": float(row["F"]) if not np.isnan(row["F"]) else None,
                "p": float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
                "eta_squared": eta_sq,
                "percent_variance_explained": eta_sq * 100
            }

            # Interpretacja wielkości efektu według Cohena
            if eta_sq < 0.01:
                effect_interpretation = "Znikomy efekt"
            elif eta_sq < 0.06:
                effect_interpretation = "Mały efekt"
            elif eta_sq < 0.14:
                effect_interpretation = "Średni efekt"
            else:
                effect_interpretation = "Duży efekt"

            results["effect_sizes"][clean_name] = {
                "eta_squared": eta_sq,
                "interpretation": effect_interpretation,
                "percent_variance": eta_sq * 100
            }

            # Sprawdzenie czy to efekt główny tożsamości
            if clean_name == "identity":
                identity_effect_found = True
                results["identity_dominance"] = {
                    "eta_squared": eta_sq,
                    "percent_variance": eta_sq * 100,
                    "dominates_model": eta_sq > 0.90,  # Czy tożsamość wyjaśnia >90% wariancji
                    "interpretation": f"Tożsamość wyjaśnia {eta_sq * 100:.1f}% całkowitej wariancji"
                }

        return results
    except Exception as e:
        logger.error(f"Błąd podczas ANOVA dla {dependent_var}: {e}")
        return {"error": str(e)}


def analyze_anglocentrism_with_bootstrap(data: pd.DataFrame) -> Dict[str, Any]:
    """
    METODA 2: Analiza anglocentryzmu z bootstrapem.
    Zawiera analizę 1D (osobno dla każdego wymiaru) oraz 2D (łącznie).
    """
    if data.empty:
        return {"error": "Brak danych do analizy anglocentryzmu"}

    results = {
        "hypothesis_H1": "W warunkach neutralnych model domyślnie przyjmuje profil amerykański",
        "method": "Bootstrap analysis of cultural distances",
        "bootstrap_iterations": 10000,
        "independent_score_1D": {},
        "interdependent_score_1D": {},
        "euclidean_2D": {},
        "overall_test": {}
    }

    for strength in PROMPT_STRENGTHS:
        for language in LANGUAGES:
            key = f"{language}_{strength}"
            strength_lang_data = data[(data["strength"] == strength) & (data["language"] == language)]

            if strength_lang_data.empty:
                continue

            # Dane dla różnych tożsamości
            neutral_data = strength_lang_data[strength_lang_data["identity"] == "neutralna"]
            american_data = strength_lang_data[strength_lang_data["identity"] == "amerykanska"]
            polish_data = strength_lang_data[strength_lang_data["identity"] == "polska"]
            japanese_data = strength_lang_data[strength_lang_data["identity"] == "japonska"]

            if any(d.empty for d in [neutral_data, american_data, polish_data, japanese_data]):
                continue

            # Analiza 1D dla independent score
            neutral_ind = neutral_data["independent_score"]
            american_ind = american_data["independent_score"]
            polish_ind = polish_data["independent_score"]
            japanese_ind = japanese_data["independent_score"]

            if all(len(x) > 0 for x in [neutral_ind, american_ind, polish_ind, japanese_ind]):
                result_ind = analyze_single_dimension_with_bootstrap(
                    neutral_ind, american_ind, polish_ind, japanese_ind
                )
                results["independent_score_1D"][key] = result_ind

            # Analiza 1D dla interdependent score
            neutral_int = neutral_data["interdependent_score"]
            american_int = american_data["interdependent_score"]
            polish_int = polish_data["interdependent_score"]
            japanese_int = japanese_data["interdependent_score"]

            if all(len(x) > 0 for x in [neutral_int, american_int, polish_int, japanese_int]):
                result_int = analyze_single_dimension_with_bootstrap(
                    neutral_int, american_int, polish_int, japanese_int
                )
                results["interdependent_score_1D"][key] = result_int

            # Analiza 2D (euklidesowa)
            result_2d = analyze_euclidean_distances_with_bootstrap(
                neutral_data, american_data, polish_data, japanese_data
            )
            results["euclidean_2D"][key] = result_2d

    # Podsumowanie dla H1 - uwzględniamy wszystkie analizy
    h1_confirmations = 0
    total_tests = 0

    # Zliczanie wyników z analiz 1D i 2D
    for analysis_type in ["independent_score_1D", "interdependent_score_1D", "euclidean_2D"]:
        for condition_result in results[analysis_type].values():
            if "anglocentrism_statistically_significant" in condition_result:
                total_tests += 1
                if condition_result["anglocentrism_statistically_significant"]:
                    h1_confirmations += 1

    results["overall_test"] = {
        "h1_confirmed": h1_confirmations > total_tests / 2 if total_tests > 0 else False,
        "proportion_confirming_h1": h1_confirmations / total_tests if total_tests > 0 else 0,
        "total_conditions_tested": total_tests,
        "conditions_confirming_h1": h1_confirmations,
        "interpretation": "H1 potwierdzona" if h1_confirmations > total_tests / 2 else "H1 odrzucona"
    }

    return results


def analyze_single_dimension_with_bootstrap(neutral_data: pd.Series,
                                            american_data: pd.Series,
                                            polish_data: pd.Series,
                                            japanese_data: pd.Series) -> Dict[str, Any]:
    """
    Analizuje anglocentryzm dla pojedynczego wymiaru (1D) z bootstrapem.
    """
    if any(d.empty for d in [neutral_data, american_data, polish_data, japanese_data]):
        return {"error": "Brak danych dla jednej z grup w analizie 1D"}

    # Średnie
    neutral_mean = neutral_data.mean()
    american_mean = american_data.mean()
    polish_mean = polish_data.mean()
    japanese_mean = japanese_data.mean()

    # Odległości (wartości bezwzględne różnic)
    diff_american = abs(neutral_mean - american_mean)
    diff_polish = abs(neutral_mean - polish_mean)
    diff_japanese = abs(neutral_mean - japanese_mean)

    # Która kultura jest najbliższa?
    diffs = [("amerykanska", diff_american), ("polska", diff_polish), ("japonska", diff_japanese)]
    closest_culture = min(diffs, key=lambda x: x[1])[0]

    # Bootstrap
    n_bootstrap = 10000
    delta_polish_american = []
    delta_japanese_american = []

    # Konwersja na numpy arrays
    n_arr = neutral_data.to_numpy()
    a_arr = american_data.to_numpy()
    p_arr = polish_data.to_numpy()
    j_arr = japanese_data.to_numpy()

    for _ in range(n_bootstrap):
        # Bootstrap samples
        boot_neutral = np.random.choice(n_arr, size=len(n_arr), replace=True)
        boot_american = np.random.choice(a_arr, size=len(a_arr), replace=True)
        boot_polish = np.random.choice(p_arr, size=len(p_arr), replace=True)
        boot_japanese = np.random.choice(j_arr, size=len(j_arr), replace=True)

        # Bootstrap różnice
        boot_diff_american = abs(boot_neutral.mean() - boot_american.mean())
        boot_diff_polish = abs(boot_neutral.mean() - boot_polish.mean())
        boot_diff_japanese = abs(boot_neutral.mean() - boot_japanese.mean())

        # Różnice różnic (dodatnia wartość = amerykański bliższy)
        delta_polish_american.append(boot_diff_polish - boot_diff_american)
        delta_japanese_american.append(boot_diff_japanese - boot_diff_american)

    # 95% przedziały ufności
    ci_polish_american = np.percentile(delta_polish_american, [2.5, 97.5])
    ci_japanese_american = np.percentile(delta_japanese_american, [2.5, 97.5])

    # Testy istotności (CI całkowicie powyżej 0 = amerykański istotnie bliższy)
    sig_closer_than_polish = ci_polish_american[0] > 0
    sig_closer_than_japanese = ci_japanese_american[0] > 0

    anglocentrism_significant = closest_culture == "amerykanska" and (
                sig_closer_than_polish or sig_closer_than_japanese)

    return {
        "means": {
            "neutral": float(neutral_mean),
            "american": float(american_mean),
            "polish": float(polish_mean),
            "japanese": float(japanese_mean)
        },
        "distances": {
            "neutral_to_american": float(diff_american),
            "neutral_to_polish": float(diff_polish),
            "neutral_to_japanese": float(diff_japanese)
        },
        "closest_culture": closest_culture,
        "bootstrap_results": {
            "polish_minus_american": {
                "mean_difference": float(np.mean(delta_polish_american)),
                "ci_95": [float(ci_polish_american[0]), float(ci_polish_american[1])],
                "significant": sig_closer_than_polish,
                "interpretation": "Amerykański bliższy niż polski" if sig_closer_than_polish else "Brak istotnej różnicy"
            },
            "japanese_minus_american": {
                "mean_difference": float(np.mean(delta_japanese_american)),
                "ci_95": [float(ci_japanese_american[0]), float(ci_japanese_american[1])],
                "significant": sig_closer_than_japanese,
                "interpretation": "Amerykański bliższy niż japoński" if sig_closer_than_japanese else "Brak istotnej różnicy"
            }
        },
        "anglocentrism_present": closest_culture == "amerykanska",
        "anglocentrism_statistically_significant": anglocentrism_significant
    }


def analyze_euclidean_distances_with_bootstrap(neutral_data: pd.DataFrame,
                                               american_data: pd.DataFrame,
                                               polish_data: pd.DataFrame,
                                               japanese_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Pomocnicza funkcja do analizy odległości euklidesowych z bootstrapem.
    """
    # Oblicz średnie punkty w przestrzeni 2D
    neutral_point = np.array([
        neutral_data["independent_score"].mean(),
        neutral_data["interdependent_score"].mean()
    ])
    american_point = np.array([
        american_data["independent_score"].mean(),
        american_data["interdependent_score"].mean()
    ])
    polish_point = np.array([
        polish_data["independent_score"].mean(),
        polish_data["interdependent_score"].mean()
    ])
    japanese_point = np.array([
        japanese_data["independent_score"].mean(),
        japanese_data["interdependent_score"].mean()
    ])

    # Oblicz odległości
    dist_american = np.linalg.norm(neutral_point - american_point)
    dist_polish = np.linalg.norm(neutral_point - polish_point)
    dist_japanese = np.linalg.norm(neutral_point - japanese_point)

    # Bootstrap
    n_bootstrap = 10000
    delta_polish_american = []
    delta_japanese_american = []

    for _ in range(n_bootstrap):
        # Resample z każdej grupy
        n_idx = np.random.choice(len(neutral_data), len(neutral_data), replace=True)
        a_idx = np.random.choice(len(american_data), len(american_data), replace=True)
        p_idx = np.random.choice(len(polish_data), len(polish_data), replace=True)
        j_idx = np.random.choice(len(japanese_data), len(japanese_data), replace=True)

        # Bootstrap średnie
        boot_neutral = np.array([
            neutral_data.iloc[n_idx]["independent_score"].mean(),
            neutral_data.iloc[n_idx]["interdependent_score"].mean()
        ])
        boot_american = np.array([
            american_data.iloc[a_idx]["independent_score"].mean(),
            american_data.iloc[a_idx]["interdependent_score"].mean()
        ])
        boot_polish = np.array([
            polish_data.iloc[p_idx]["independent_score"].mean(),
            polish_data.iloc[p_idx]["interdependent_score"].mean()
        ])
        boot_japanese = np.array([
            japanese_data.iloc[j_idx]["independent_score"].mean(),
            japanese_data.iloc[j_idx]["interdependent_score"].mean()
        ])

        # Bootstrap odległości
        boot_dist_american = np.linalg.norm(boot_neutral - boot_american)
        boot_dist_polish = np.linalg.norm(boot_neutral - boot_polish)
        boot_dist_japanese = np.linalg.norm(boot_neutral - boot_japanese)

        # Różnice odległości
        delta_polish_american.append(boot_dist_polish - boot_dist_american)
        delta_japanese_american.append(boot_dist_japanese - boot_dist_american)

    # 95% przedziały ufności
    ci_polish_american = np.percentile(delta_polish_american, [2.5, 97.5])
    ci_japanese_american = np.percentile(delta_japanese_american, [2.5, 97.5])

    # Test czy amerykański jest istotnie bliższy
    sig_closer_than_polish = ci_polish_american[0] > 0
    sig_closer_than_japanese = ci_japanese_american[0] > 0

    closest_culture = min(
        [("amerykanska", dist_american), ("polska", dist_polish), ("japonska", dist_japanese)],
        key=lambda x: x[1]
    )[0]

    return {
        "distances": {
            "neutral_to_american": float(dist_american),
            "neutral_to_polish": float(dist_polish),
            "neutral_to_japanese": float(dist_japanese)
        },
        "closest_culture": closest_culture,
        "bootstrap_results": {
            "polish_minus_american": {
                "mean_difference": float(np.mean(delta_polish_american)),
                "ci_95": [float(ci_polish_american[0]), float(ci_polish_american[1])],
                "significant": sig_closer_than_polish,
                "interpretation": "Amerykański bliższy niż polski" if sig_closer_than_polish else "Brak istotnej różnicy"
            },
            "japanese_minus_american": {
                "mean_difference": float(np.mean(delta_japanese_american)),
                "ci_95": [float(ci_japanese_american[0]), float(ci_japanese_american[1])],
                "significant": sig_closer_than_japanese,
                "interpretation": "Amerykański bliższy niż japoński" if sig_closer_than_japanese else "Brak istotnej różnicy"
            }
        },
        "anglocentrism_present": closest_culture == "amerykanska",
        "anglocentrism_statistically_significant": closest_culture == "amerykanska" and (
                    sig_closer_than_polish or sig_closer_than_japanese)
    }


def analyze_cultural_separation(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analizuje separację między grupami kulturowymi - dodatkowy dowód na karykaturalne profile.
    """
    results = {
        "mean_positions": {},
        "pairwise_distances": {},
        "within_group_variance": {},
        "between_group_variance": 0
    }

    # Oblicz średnie pozycje dla każdej kultury
    for identity in IDENTITIES:
        subset = data[data['identity'] == identity]
        results["mean_positions"][identity] = {
            "independent": float(subset['independent_score'].mean()),
            "interdependent": float(subset['interdependent_score'].mean()),
            "n": len(subset)
        }

    # Oblicz odległości między parami kultur
    for i, id1 in enumerate(IDENTITIES):
        for id2 in IDENTITIES[i + 1:]:
            pos1 = np.array([results["mean_positions"][id1]["independent"],
                             results["mean_positions"][id1]["interdependent"]])
            pos2 = np.array([results["mean_positions"][id2]["independent"],
                             results["mean_positions"][id2]["interdependent"]])
            distance = float(np.linalg.norm(pos1 - pos2))
            results["pairwise_distances"][f"{id1}-{id2}"] = distance

    # Oblicz wariancję wewnątrzgrupową
    for identity in IDENTITIES:
        subset = data[data['identity'] == identity]
        if len(subset) > 1:
            var_ind = float(subset['independent_score'].var())
            var_int = float(subset['interdependent_score'].var())
            results["within_group_variance"][identity] = {
                "independent_var": var_ind,
                "interdependent_var": var_int,
                "total_var": var_ind + var_int
            }

    # Średnia wariancja wewnątrzgrupowa vs międzygrupowa
    avg_within_var = np.mean([v["total_var"] for v in results["within_group_variance"].values()])
    avg_between_dist = np.mean(list(results["pairwise_distances"].values()))

    results["separation_index"] = float(avg_between_dist / avg_within_var) if avg_within_var > 0 else float('inf')
    results["interpretation"] = (
        "Wysoki indeks separacji (stosunek odległości międzygrupowej do wariancji wewnątrzgrupowej) "
        f"= {results['separation_index']:.2f} potwierdza karykaturalne, odrębne profile")

    return results


def visualize_cultural_space_and_classify(data: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    METODA 3: Wizualizacja przestrzeni kulturowej i klasyfikacja - dowód na H2.
    Pokazuje karykaturalne, doskonale separowalne profile kulturowe.
    """
    results = {
        "hypothesis_H2": "Po narzuceniu tożsamości model przyjmuje skrajny, stereotypowy wariant",
        "visualization": {},
        "classification": {},
        "threshold_for_caricature": 90.0  # 90% dokładności = karykaturalne profile
    }

    # Przygotowanie danych
    # Używamy odpowiedzi na wszystkie pytania jako features
    question_columns = [f"q{i}" for i in range(1, 31)]

    # Sprawdzenie czy mamy wszystkie kolumny
    missing_columns = [col for col in question_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Brakujące kolumny pytań: {missing_columns}")
        return {"error": "Brakujące dane pytań"}

    X = data[question_columns].values
    y = data['identity'].values

    # WIZUALIZACJE
    logger.info("Tworzenie wizualizacji przestrzeni kulturowej...")

    # 1. PROSTY SCATTERPLOT - niezależność vs współzależność
    plt.figure(figsize=(12, 10))

    # Kolory dla różnych tożsamości
    colors = {
        'amerykanska': '#0066CC',  # niebieski
        'polska': '#DC143C',  # czerwony
        'neutralna': '#696969',  # szary
        'japonska': '#228B22'  # zielony
    }
    # Labele po angielsku
    identity_labels_en = {
        'amerykanska': 'American',
        'polska': 'Polish',
        'neutralna': 'Neutral',
        'japonska': 'Japanese'
    }

    # Rysowanie każdej odpowiedzi jako punkt
    for identity in IDENTITIES:
        mask = data['identity'] == identity
        subset = data[mask]

        # Pobranie angielskiej etykiety ze słownika
        english_label = identity_labels_en.get(identity, identity.capitalize())

        plt.scatter(subset['independent_score'], subset['interdependent_score'],
                    c=colors[identity],
                    # Użycie angielskiej etykiety w legendzie
                    label=f'{english_label} (n={len(subset)})',
                    alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

    plt.xlabel('Independent Score', fontsize=14)
    plt.ylabel('Interdependent Score', fontsize=14)

    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Dodaj linie średnich dla każdej kultury
    for identity in IDENTITIES:
        mask = data['identity'] == identity
        subset = data[mask]
        mean_x = subset['independent_score'].mean()
        mean_y = subset['interdependent_score'].mean()
        plt.scatter(mean_x, mean_y, c=colors[identity], s=500, marker='*',
                    edgecolors='black', linewidth=2, alpha=1.0)
        plt.annotate(f'{identity[:3].upper()}', (mean_x, mean_y),
                     xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

    # Ustawienie pełnego zakresu osi od 1 do 7
    plt.xlim(1, 7)
    plt.ylim(1, 7)

    scatter_path = os.path.join(output_dir, "cultural_space_scatterplot.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. WYKRES PUDEŁKOWY - rozkłady dla każdej kultury
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    english_tick_labels = [identity_labels_en.get(id, id.capitalize()) for id in IDENTITIES]

    # Niezależność
    data_ind = [data[data['identity'] == id]['independent_score'].values for id in IDENTITIES]
    bp1 = ax1.boxplot(data_ind, labels=english_tick_labels, patch_artist=True)
    for patch, identity in zip(bp1['boxes'], IDENTITIES):
        patch.set_facecolor(colors[identity])
    ax1.set_ylabel('Independent Score', fontsize=12)
    ax1.set_title('Independent Scale', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1, 7)

    # Współzależność
    data_int = [data[data['identity'] == id]['interdependent_score'].values for id in IDENTITIES]
    bp2 = ax2.boxplot(data_int, labels=english_tick_labels, patch_artist=True)
    for patch, identity in zip(bp2['boxes'], IDENTITIES):
        patch.set_facecolor(colors[identity])
    ax2.set_ylabel('Interdependent Score', fontsize=12)
    ax2.set_title('Interdependence Scale', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1, 7)
    boxplot_path = os.path.join(output_dir, "cultural_profiles_boxplot.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. PCA dla wizualizacji wszystkich 30 pytań
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

    plt.figure(figsize=(12, 8))

    # Rysowanie punktów z PCA
    for identity in IDENTITIES:
        mask = y == identity
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[identity], label=identity,
                    alpha=0.6, s=100, edgecolors='black')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)', fontsize=12)
    plt.title('PCA przestrzeni 30-wymiarowej (wszystkie pytania)\nDoskonała separacja = karykaturalne profile',
              fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    pca_path = os.path.join(output_dir, "cultural_space_pca.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Słownik mapujący nazwy wewnętrzne na polskie etykiety do wyświetlania
    identity_labels_pl = {
        'amerykanska': 'Amerykańska',
        'polska': 'Polska',
        'neutralna': 'Neutralna',
        'japonska': 'Japońska'
    }

    # Stworzenie listy polskich etykiet w odpowiedniej kolejności dla osi heatmapy
    polish_tick_labels = [identity_labels_pl.get(id, id.capitalize()) for id in IDENTITIES]

    # 4. Heatmapa odległości między kulturami
    logger.info("Tworzenie heatmapy odległości kulturowych...")
    identity_labels_en = {
        'amerykanska': 'American',
        'polska': 'Polish',
        'neutralna': 'Neutral',
        'japonska': 'Japanese'
    }
    english_tick_labels = [identity_labels_en.get(id, id.capitalize()) for id in IDENTITIES]

    distance_matrix = np.zeros((len(IDENTITIES), len(IDENTITIES)))
    for i, id1 in enumerate(IDENTITIES):
        for j, id2 in enumerate(IDENTITIES):
            if i != j:
                subset1 = data[data['identity'] == id1]
                subset2 = data[data['identity'] == id2]
                pos1 = np.array([subset1['independent_score'].mean(),
                                 subset1['interdependent_score'].mean()])
                pos2 = np.array([subset2['independent_score'].mean(),
                                 subset2['interdependent_score'].mean()])
                distance_matrix[i, j] = np.linalg.norm(pos1 - pos2)

    plt.figure(figsize=(10, 8))  # Nieco większy rozmiar dla lepszej czytelności etykiet
    sns.heatmap(distance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=english_tick_labels, yticklabels=english_tick_labels,
                cbar_kws={'label': 'Euclidean Distance'})
    plt.title('Distance Between Cultural Profiles', fontsize=14)

    heatmap_path = os.path.join(output_dir, "cultural_distances_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    results["visualization"] = {
        "scatterplot_path": scatter_path,
        "boxplot_path": boxplot_path,
        "pca_path": pca_path,
        "heatmap_path": heatmap_path,
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "interpretation": "Wizualizacje pokazują karykaturalne profile - każda kultura tworzy zwartą, odrębną grupę"
    }

    # KLASYFIKACJA - test
    logger.info("Przeprowadzanie klasyfikacji tożsamości...")

    # Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Leave-One-Out cross-validation dla maksymalnej rzetelności
    loo = LeaveOneOut()
    cv_scores = cross_val_score(knn, X_scaled, y, cv=loo)

    # Trenowanie na pełnych danych dla raportu
    knn.fit(X_scaled, y)
    y_pred = knn.predict(X_scaled)

    # Macierz pomyłek
    cm = confusion_matrix(y, y_pred)

    # Raport klasyfikacji
    class_report = classification_report(y, y_pred, output_dict=True)

    accuracy_percentage = float(np.mean(cv_scores) * 100)
    caricature_confirmed = accuracy_percentage >= 90.0  # Próg 90%

    results["classification"] = {
        "method": "k-Nearest Neighbors (k=3)",
        "features": "Odpowiedzi na 30 pytań kwestionariusza",
        "accuracy_loocv": float(np.mean(cv_scores)),
        "accuracy_percentage": accuracy_percentage,
        "caricature_profiles_confirmed": caricature_confirmed,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "interpretation": f"Dokładność {accuracy_percentage:.1f}% {'POTWIERDZA' if caricature_confirmed else 'nie potwierdza'} "
                          f"hipotezę o karykaturalnych profilach (próg: 90%)"
    }

    # Dodatkowa analiza - które pytania najlepiej różnicują tożsamości
    feature_importance = analyze_feature_importance(X, y, question_columns)
    results["classification"]["most_discriminative_questions"] = feature_importance

    # Wizualizacja macierzy pomyłek
    identity_labels_en = {
        'amerykanska': 'American',
        'polska': 'Polish',
        'neutralna': 'Neutral',
        'japonska': 'Japanese'
    }

    english_tick_labels = [identity_labels_en.get(id, id.capitalize()) for id in IDENTITIES]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=english_tick_labels, yticklabels=english_tick_labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Identity', fontsize=12)
    plt.xlabel('Predicted Identity', fontsize=12)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    results["classification"]["confusion_matrix_plot"] = cm_path

    # Dodatkowa analiza separacji kulturowej
    separation_analysis = analyze_cultural_separation(data)
    results["cultural_separation"] = separation_analysis

    return results


def analyze_feature_importance(X: np.ndarray, y: np.ndarray,
                               feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Analizuje które pytania najlepiej różnicują tożsamości kulturowe.
    """
    from sklearn.ensemble import RandomForestClassifier

    # Random Forest dla oceny ważności cech
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Sortowanie pytań według ważności
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_features = []
    for i in range(10):  # Top 10 pytań
        idx = indices[i]
        top_features.append({
            "question": feature_names[idx],
            "importance": float(importances[idx]),
            "rank": i + 1
        })

    return top_features


def analyze_data(data: pd.DataFrame, current_output_dir: str, model_name: str) -> Dict[str, Any]:
    """Przeprowadza pełną analizę danych dla danego modelu zgodnie z hipotezami H1 i H2."""
    if data.empty:
        logger.warning(f"Brak danych do analizy dla modelu {model_name}.")
        return {"error": "Brak danych do analizy"}

    results = {
        "model_name": model_name,
        "hypotheses": {
            "H1": "Anglocentryczny domyślny: W warunkach neutralnych model przyjmuje profil amerykański",
            "H2": "Karykaturalne przejmowanie tożsamości: Model przyjmuje skrajne, stereotypowe warianty kultur"
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "methods": {}
    }

    # METODA 1: ANOVA z eta-kwadrat
    logger.info("METODA 1: Analiza wariancji z eta-kwadrat...")
    results["methods"]["anova_eta_squared"] = {
        "independent_score": perform_three_way_anova_with_eta_squared(data, "independent_score"),
        "interdependent_score": perform_three_way_anova_with_eta_squared(data, "interdependent_score")
    }

    # METODA 2: Analiza odległości z bootstrapem (H1)
    logger.info("METODA 2: Analiza anglocentryzmu z bootstrapem...")
    results["methods"]["bootstrap_distance_analysis"] = analyze_anglocentrism_with_bootstrap(data)

    # METODA 3: Wizualizacja i klasyfikacja (H2)
    logger.info("METODA 3: Wizualizacja przestrzeni kulturowej i klasyfikacja...")
    results["methods"]["visualization_and_classification"] = visualize_cultural_space_and_classify(data,
                                                                                                   current_output_dir)

    # Podsumowanie wyników
    results["summary"] = create_analysis_summary(results)

    # Zapisz wyniki
    json_path = os.path.join(current_output_dir, "hypothesis_testing_results.json")
    save_json_file(results, json_path)
    logger.info(f"Zapisano wyniki testowania hipotez dla modelu {model_name} do {json_path}")

    return results


def create_analysis_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Tworzy podsumowanie wyników testowania hipotez."""
    summary = {
        "H1_anglocentrism": {
            "1D_analysis": {},
            "2D_analysis": {},
            "combined": {}
        },
        "H2_caricature_profiles": {},
        "overall_conclusions": []
    }

    # Podsumowanie H1
    bootstrap_results = results["methods"].get("bootstrap_distance_analysis", {})

    # Analiza 1D
    if "independent_score_1D" in bootstrap_results:
        ind_confirmations = sum(1 for r in bootstrap_results["independent_score_1D"].values()
                                if r.get("anglocentrism_statistically_significant", False))
        ind_total = len(bootstrap_results["independent_score_1D"])
        summary["H1_anglocentrism"]["1D_analysis"]["independent"] = {
            "confirmations": ind_confirmations,
            "total": ind_total,
            "proportion": ind_confirmations / ind_total if ind_total > 0 else 0
        }

    if "interdependent_score_1D" in bootstrap_results:
        int_confirmations = sum(1 for r in bootstrap_results["interdependent_score_1D"].values()
                                if r.get("anglocentrism_statistically_significant", False))
        int_total = len(bootstrap_results["interdependent_score_1D"])
        summary["H1_anglocentrism"]["1D_analysis"]["interdependent"] = {
            "confirmations": int_confirmations,
            "total": int_total,
            "proportion": int_confirmations / int_total if int_total > 0 else 0
        }

    # Analiza 2D
    if "euclidean_2D" in bootstrap_results:
        eucl_confirmations = sum(1 for r in bootstrap_results["euclidean_2D"].values()
                                 if r.get("anglocentrism_statistically_significant", False))
        eucl_total = len(bootstrap_results["euclidean_2D"])
        summary["H1_anglocentrism"]["2D_analysis"] = {
            "confirmations": eucl_confirmations,
            "total": eucl_total,
            "proportion": eucl_confirmations / eucl_total if eucl_total > 0 else 0
        }

    # Łączne podsumowanie H1
    if "overall_test" in bootstrap_results:
        summary["H1_anglocentrism"]["combined"] = {
            "confirmed": bootstrap_results["overall_test"]["h1_confirmed"],
            "evidence_strength": bootstrap_results["overall_test"]["proportion_confirming_h1"],
            "interpretation": bootstrap_results["overall_test"]["interpretation"]
        }

    # Podsumowanie H2
    classification_results = results["methods"].get("visualization_and_classification", {})
    if "classification" in classification_results:
        classification = classification_results["classification"]
        summary["H2_caricature_profiles"] = {
            "confirmed": classification.get("caricature_profiles_confirmed", False),
            "classification_accuracy": classification.get("accuracy_percentage", 0),
            "interpretation": classification.get("interpretation", ""),
            "threshold": 90.0
        }

    # Eta-kwadrat dla tożsamości
    anova_results = results["methods"].get("anova_eta_squared", {})
    identity_effects = []
    for construct in ["independent_score", "interdependent_score"]:
        if construct in anova_results and "identity_dominance" in anova_results[construct]:
            dominance = anova_results[construct]["identity_dominance"]
            identity_effects.append({
                "construct": construct,
                "variance_explained": dominance["percent_variance"],
                "dominates": dominance["dominates_model"]
            })

    summary["H2_caricature_profiles"]["identity_effect_sizes"] = identity_effects

    # Ogólne wnioski
    h1_confirmed = summary["H1_anglocentrism"].get("combined", {}).get("confirmed", False)
    h2_confirmed = summary["H2_caricature_profiles"].get("confirmed", False)

    if h1_confirmed and h2_confirmed:
        summary["overall_conclusions"].append(
            "Obie hipotezy zostały potwierdzone: Model wykazuje anglocentryzm w warunkach neutralnych "
            "oraz przyjmuje karykaturalne, deterministyczne profile kulturowe."
        )
    elif h1_confirmed:
        summary["overall_conclusions"].append(
            "Hipoteza H1 została potwierdzona: Model wykazuje anglocentryzm w warunkach neutralnych."
        )
    elif h2_confirmed:
        summary["overall_conclusions"].append(
            "Hipoteza H2 została potwierdzona: Model przyjmuje karykaturalne, deterministyczne profile kulturowe."
        )
    else:
        summary["overall_conclusions"].append(
            "Żadna z hipotez nie została w pełni potwierdzona."
        )

    return summary


def run_experiment():
    """Przeprowadza eksperyment dla każdego zdefiniowanego modelu."""
    logger.info("Rozpoczęcie eksperymentu testującego hipotezy H1 i H2")

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
                                                 model_name.replace("/", "_"))
        os.makedirs(model_specific_output_dir, exist_ok=True)

        try:
            logger.info(f"Rozpoczęcie zbierania danych dla modelu: {model_name}")
            model_data = collect_data(model_name, model_specific_output_dir, REPEATS_PER_CONDITION)

            if model_data.empty:
                logger.warning(f"Nie zebrano żadnych danych dla modelu: {model_name}. Pomijanie analizy.")
                all_models_analysis_results[model_name] = {"error": "Brak danych"}
                continue

            all_models_data.append(model_data)

            logger.info(f"Rozpoczęcie analizy hipotez dla modelu: {model_name}")
            analysis_results = analyze_data(model_data, model_specific_output_dir, model_name)
            all_models_analysis_results[model_name] = analysis_results

            logger.info(f"=== Zakończono przetwarzanie dla modelu: {model_name} ===")

        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania modelu {model_name}: {e}", exc_info=True)
            all_models_analysis_results[model_name] = {"error": str(e)}

    # Połącz wszystkie dane i zapisz zbiorczy plik CSV
    if all_models_data:
        combined_df = pd.concat(all_models_data, ignore_index=True)
        combined_csv_path = os.path.join(BASE_OUTPUT_DIR, "all_models_results.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        logger.info(f"Zapisano połączone dane wszystkich modeli do {combined_csv_path}")

    # Zapisz zbiorcze wyniki analizy
    overall_summary_path = os.path.join(BASE_OUTPUT_DIR, "hypothesis_testing_summary.json")
    save_json_file(all_models_analysis_results, overall_summary_path)
    logger.info(f"Zapisano podsumowanie testowania hipotez dla wszystkich modeli do {overall_summary_path}")

    logger.info("Eksperyment testowania hipotez H1 i H2 zakończony.")
    return {
        "all_data": all_models_data,
        "all_analysis_results": all_models_analysis_results,
        "base_output_dir": BASE_OUTPUT_DIR
    }


if __name__ == "__main__":
    result = run_experiment()

    if "error" not in result and result.get("all_analysis_results"):
        print(f"\nEksperyment zakończony sukcesem!")
        print("\n=== PODSUMOWANIE TESTOWANIA HIPOTEZ ===")

        for model_name, analysis_res in result["all_analysis_results"].items():
            if "error" not in analysis_res:
                print(f"\nModel: {model_name}")

                if "summary" in analysis_res:
                    summary = analysis_res["summary"]

                    # H1
                    h1_result = summary.get("H1_anglocentrism", {})
                    h1_combined = h1_result.get("combined", {})
                    print(f"  H1 (Anglocentryzm): {'POTWIERDZONA' if h1_combined.get('confirmed') else 'ODRZUCONA'}")
                    if "evidence_strength" in h1_combined:
                        print(
                            f"    - Siła dowodu (łącznie): {h1_combined['evidence_strength'] * 100:.1f}% testów potwierdza")

                    # Szczegóły analizy 1D
                    if "1D_analysis" in h1_result:
                        if "independent" in h1_result["1D_analysis"]:
                            ind_1d = h1_result["1D_analysis"]["independent"]
                            print(f"    - Analiza 1D (niezależność): {ind_1d['proportion'] * 100:.1f}% potwierdzeń")
                        if "interdependent" in h1_result["1D_analysis"]:
                            int_1d = h1_result["1D_analysis"]["interdependent"]
                            print(f"    - Analiza 1D (współzależność): {int_1d['proportion'] * 100:.1f}% potwierdzeń")

                    # Analiza 2D
                    if "2D_analysis" in h1_result:
                        eucl_2d = h1_result["2D_analysis"]
                        print(f"    - Analiza 2D (euklidesowa): {eucl_2d['proportion'] * 100:.1f}% potwierdzeń")

                    # H2
                    h2_result = summary.get("H2_caricature_profiles", {})
                    print(
                        f"  H2 (Karykaturalne profile): {'POTWIERDZONA' if h2_result.get('confirmed') else 'ODRZUCONA'}")
                    if "classification_accuracy" in h2_result:
                        print(f"    - Dokładność klasyfikacji: {h2_result['classification_accuracy']:.1f}%")
                        print(f"    - Próg potwierdzenia: {h2_result.get('threshold', 90)}%")

                    # Wariancja wyjaśniona przez tożsamość
                    if "identity_effect_sizes" in h2_result:
                        for effect in h2_result["identity_effect_sizes"]:
                            print(
                                f"    - Tożsamość wyjaśnia {effect['variance_explained']:.1f}% wariancji w {effect['construct']}")

                    # Wnioski
                    if "overall_conclusions" in summary:
                        print("  Wnioski:")
                        for conclusion in summary["overall_conclusions"]:
                            print(f"    {conclusion}")

            else:
                print(f"\nModel: {model_name}, Wystąpił błąd: {analysis_res['error']}")

        print(f"\nWyniki zapisane w katalogu: {result['base_output_dir']}")
    elif "error" in result:
        print(f"\nEksperyment zakończony z błędem: {result['error']}")
    else:
        print(f"\nEksperyment zakończony, ale nie przetworzono żadnych modeli pomyślnie.")
