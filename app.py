import streamlit as st
import openai
import pandas as pd
from pycaret.regression import load_model, predict_model
from io import BytesIO
from audiorecorder import audiorecorder  # type: ignore
from openai import OpenAI
from hashlib import md5
import re
import json
import ast

# Wczytaj wytrenowany model
model = load_model('best_insurance_model')
AUDIO_TRANSCRIBE_MODEL = "whisper-1"

if "api_key" not in st.session_state:
    api_key = st.text_input("Podaj swój klucz OpenAI API", type="password", key="openai_api_key")
    if api_key:
        st.session_state["api_key"] = api_key
        st.rerun()
    else:
        st.warning("Podaj swój klucz OpenAI API.")
        st.stop()
else:
    api_key = st.session_state["api_key"]

def transcribe_audio(audio_bytes, api_key):
    openai_client = OpenAI(api_key=api_key)
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json"
    )
    return transcript.text

st.image("insurance.png", use_container_width=True)
st.title("Oblicz koszty ubezpieczenia zdrowotnego")
st.markdown("Wprowadź dane pacjenta:")



# Dane wejściowe
age = st.number_input("Wiek", min_value=0, max_value=120, value=30)
sex = st.selectbox("Płeć", ['Mężczyzna', 'Kobieta'])
# Nowe pola do obliczania BMI
st.markdown("##### Oblicz swoje BMI")
waga = st.number_input("Waga (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1, key="waga")
wzrost = st.number_input("Wzrost (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1, key="wzrost")

if st.button("Oblicz BMI"):
    bmi_value = waga / ((wzrost / 100) ** 2)
    st.session_state["bmi"] = round(bmi_value, 2)
    st.success(f"Twoje BMI: {st.session_state['bmi']}")

# Pole BMI z automatycznym uzupełnianiem
bmi = st.number_input(
    "BMI", 
    min_value=10.0, 
    max_value=50.0, 
    value=st.session_state.get("bmi", 25.0), 
    key="bmi_input"
)
children = st.slider("Liczba dzieci", 0, 5, 0)
smoker = st.selectbox("Czy pali?", ['Tak', 'Nie'])
region = st.selectbox("Region", ['południowy wschód', 'południowy zachód', 'północny wschód', 'północny zachód'],)

# Przygotuj dane do predykcji
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}])



# Przycisk do uruchomienia predykcji
if st.button("Oblicz koszt ubezpieczenia"):
    prediction = predict_model(model, data=input_data)
    predicted_charge = prediction.loc[0, 'prediction_label']
    st.success(f"💰 Szacowany koszt ubezpieczenia: {predicted_charge:.2f} USD")

    st.markdown("---")
    st.header("Jak obniżyć koszty ubezpieczenia?")

    if api_key:
        improved_bmi = max(18.5, bmi - 2) if bmi > 22 else bmi
        improved_smoker = "Nie" if smoker == "Tak" else smoker

        improved_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': improved_bmi,
            'children': children,
            'smoker': improved_smoker,
            'region': region
        }])
        improved_prediction = predict_model(model, data=improved_data)
        improved_charge = improved_prediction.loc[0, 'prediction_label']
        savings = predicted_charge - improved_charge

        prompt = (
            f"Pacjent: wiek {age}, płeć {sex}, BMI {bmi}, liczba dzieci {children}, "
            f"pali: {smoker}, region: {region}. "
            f"Obecny koszt ubezpieczenia: {predicted_charge:.2f} USD. "
            f"Po zmianie BMI na {improved_bmi} i statusu palenia na '{improved_smoker}' koszt spada do {improved_charge:.2f} USD. "
            f"Oszczędność wynosi {savings:.2f} USD. "
            "Jakie są 2-3 praktyczne sposoby na obniżenie kosztów ubezpieczenia zdrowotnego dla tej osoby? "
            "Odpowiedz po polsku, zwięźle i konkretnie."
        )
        try:
            with st.spinner("Generuję podpowiedź..."):
                openai.api_key = api_key
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7,
                )
            answer = response.choices[0].message.content
            st.markdown("#### Oto co proponuje:")
            st.markdown(f"<div style='padding:16px; border-radius:8px; font-size:16px'>{answer}</div>", unsafe_allow_html=True)
            st.markdown(f"**Potencjalna oszczędność na składce: {savings:.2f} USD**")
        except Exception as e:
            st.error(f"Błąd podczas komunikacji z OpenAI: {e}")
   


def extract_json_from_text(text):
    """
    Próbuj wyciągnąć fragment JSON z tekstu.
    """

    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str
    return text

def extract_data(text):
    data = {}
    age = re.search(r"(\d+)\s*(lat|lata|roków)", text)
    if age: data["wiek"] = int(age.group(1))
    gender = re.search(r"(mężczyzna|kobieta)", text)
    if gender: data["płeć"] = "Mężczyzna" if "mężczyzna" in gender.group(1) else "Kobieta"
    bmi = re.search(r"bmi[:\s]*([0-9]+(?:[.,][0-9]+)?)", text, re.IGNORECASE)
    if bmi: data["bmi"] = float(bmi.group(1).replace(",", "."))
    children = re.search(r"(\d+)\s*(dzieci|dziecko)", text)
    if children: data["dzieci"] = int(children.group(1))
    smoker = re.search(r"(palę|nie palę|tak pali|nie pali)", text)
    if smoker:
        data["pali"] = "Tak" if "pal" in smoker.group(1) and "nie" not in smoker.group(1) else "Nie"
    region = re.search(r"(południowy wschód|południowy zachód|północny wschód|północny zachód)", text)
    if region: data["region"] = region.group(1)
    return data

#Notatka w sidebarze

with st.sidebar:
    st.image("insurance1.png", use_container_width=True)
    st.header("🎤 Nagraj się żeby obliczyć składkę")
    audio = audiorecorder("Nagraj", "Zatrzymaj nagrywanie")
    if audio:
        audio_bytes = audio.export(format="wav").read()
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Oblicz składkę i podpowiedz jak ją obniżyć"):
            try:
                transcript = transcribe_audio(audio_bytes, api_key)
                st.success(f"Transkrypcja: {transcript}")

                # Najpierw spróbuj wyciągnąć dane jako JSON
                json_candidate = extract_json_from_text(transcript)
                try:
                    data_json = json.loads(json_candidate)
                except Exception:
                    try:
                        import ast
                        data_json = ast.literal_eval(json_candidate)
                    except Exception:
                        # Wyciągamy dane z tekstu
                        data_json = extract_data(transcript)
                        if not data_json:
                            st.error("Nie udało się odczytać danych z notatki głosowej. Upewnij się, że notatka zawiera dane w formacie JSON lub czytelne liczby i słowa kluczowe.")
                            st.stop()

                age_val = int(data_json.get("wiek", 30))
                bmi_val = float(str(data_json.get("bmi", 25.0)).replace(",", "."))
                sex_val = data_json.get("płeć", "Mężczyzna").capitalize()
                smoker_val = data_json.get("pali", "Nie").capitalize()
                children_val = int(data_json.get("dzieci", 0))
                region_val = data_json.get("region", "południowy wschód")

                sidebar_data = pd.DataFrame([{
                    'age': age_val,
                    'sex': sex_val,
                    'bmi': bmi_val,
                    'children': children_val,
                    'smoker': smoker_val,
                    'region': region_val
                }])

                prediction = predict_model(model, data=sidebar_data)
                predicted_charge = prediction.loc[0, 'prediction_label']
                st.success(f"💰 Szacowany koszt ubezpieczenia: {predicted_charge:.2f} USD")

                improved_bmi = max(18.5, bmi_val - 2) if bmi_val > 22 else bmi_val
                improved_smoker = "Nie" if smoker_val == "Tak" else smoker_val

                improved_data = pd.DataFrame([{
                    'age': age_val,
                    'sex': sex_val,
                    'bmi': improved_bmi,
                    'children': children_val,
                    'smoker': improved_smoker,
                    'region': region_val
                }])
                improved_prediction = predict_model(model, data=improved_data)
                improved_charge = improved_prediction.loc[0, 'prediction_label']
                savings = predicted_charge - improved_charge

                prompt = (
                    f"Pacjent: wiek {age_val}, płeć {sex_val}, BMI {bmi_val}, liczba dzieci {children_val}, "
                    f"pali: {smoker_val}, region: {region_val}. "
                    f"Obecny koszt ubezpieczenia: {predicted_charge:.2f} USD. "
                    f"Po zmianie BMI na {improved_bmi} i statusu palenia na '{improved_smoker}' koszt spada do {improved_charge:.2f} USD. "
                    f"Oszczędność wynosi {savings:.2f} USD. "
                    "Wypisz 3 praktyczne sposoby na obniżenie kosztów ubezpieczenia zdrowotnego dla tej osoby w punktach. "
                    "Odpowiedz po polsku, zwięźle i konkretnie."
                )
                with st.spinner("Generuję podpowiedź..."):
                    openai.api_key = api_key
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.7,
                    )
                answer = response.choices[0].message.content
                st.markdown("#### 3 sposoby na obniżenie składki:")
                st.markdown(f"<div style='padding:16px; border-radius:8px; font-size:16px'>{answer}</div>", unsafe_allow_html=True)
                st.markdown(f"**Potencjalna oszczędność na składce: {savings:.2f} USD**")
            except Exception as e:
                st.error(f"Błąd podczas przetwarzania notatki: {e}")