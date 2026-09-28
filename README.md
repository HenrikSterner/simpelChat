# Simpel AI-chat

En lokal Streamlit-chat, som bruger **Ollama** og modellen **phi3.5**. Modellen bliver
hentet første gang, du trykker på **Indlæs AI**, og svarene streames direkte i chatten.

## Start appen

1. Installér [Ollama](https://ollama.com/), og sørg for, at programmet kører.
2. Installér Python-afhængighederne:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start Streamlit:

   ```powershell
   python -m streamlit run app.py
   ```

4. Åbn den viste adresse i browseren, og tryk på **Indlæs AI**.

Første indlæsning kan tage nogle minutter, fordi Ollama skal hente modellen. Senere
starter den hurtigere. Hvis Ollama kører på en anden adresse, kan den angives med
miljøvariablen `OLLAMA_URL`.

