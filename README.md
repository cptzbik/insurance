Kalkulator Ubezpieczeń Zdrowotnych
Aplikacja Streamlit do szacowania kosztów ubezpieczenia zdrowotnego na podstawie danych użytkownika. Pozwala także nagrać notatkę głosową, automatycznie rozpoznać dane, obliczyć składkę i uzyskać podpowiedzi, jak ją obniżyć dzięki AI (OpenAI GPT).

Funkcje
Wprowadzanie danych ręcznie lub głosowo (transkrypcja przez OpenAI Whisper)
Automatyczne wyciąganie danych z tekstu lub JSON
Predykcja kosztu ubezpieczenia na podstawie wytrenowanego modelu
Generowanie podpowiedzi, jak obniżyć składkę (AI)
Przejrzysty interfejs Streamlit
Wymagania
Python 3.8+
Klucz API OpenAI (do transkrypcji i podpowiedzi)
Plik modelu best_insurance_model (PyCaret)
Pliki graficzne: insurance.png, insurance1.png (opcjonalnie własne)

Instalacja
git clone https://github.com/cptzbik/insurance.git
cd cptzbik-insurance
pip install -r requirements.txt

Uruchomienie
streamlit run app.py

Użycie
Podaj swój klucz OpenAI API.
Wprowadź dane ręcznie lub nagraj notatkę głosową z danymi (wiek, płeć, BMI, liczba dzieci, palenie, region).
Odczytaj szacowany koszt ubezpieczenia i propozycje obniżenia składki.
Przykład notatki głosowej
Mam 35 lat, jestem mężczyzną, BMI 27, mam 2 dzieci, nie palę, region południowy wschód.
