# Studievejleder

En lokal Streamlit-app, der bruger **Ollama** og modellen **phi3.5** til at hjælpe gymnasieelever med at udforske videregående uddannelser. Profilen, svarene og favoritterne gemmes i Streamlit-sessionen.

## Start appen

1. Installér Ollama fra https://ollama.com/, og sørg for, at programmet kører.
2. Installér afhængighederne med: python -m pip install -r requirements.txt
3. Start appen med: python -m streamlit run app.py
4. Vælg **Indlæs AI** i sidepanelet. Første gang hentes modellen automatisk.

## Sådan bruges appen

Udfyld spørgeskemaet, gennemgå dine svar og besvar eventuelle afklarende spørgsmål. AI'en stiller kun opfølgning, når den vurderer, at profilen er uklar eller mangelfuld, og stiller højst fem spørgsmål. Derefter kan du generere uddannelsesforslag, sammenligne forslag, gemme favoritter og chatte videre med profilen som kontekst.

Du kan når som helst ændre profilen. Når du laver nye forslag efter en ændring, vises hvilke uddannelser der er kommet til eller faldet fra. Favoritter kan ses og sammenlignes på deres egen side.

Uddannelsesforslag er vejledning og inspiration. Kontrollér altid aktuelle adgangskrav og uddannelsesdetaljer på institutionens officielle hjemmeside. AI'en kan mangle eller tage fejl af oplysninger.

Hvis Ollama kører på en anden adresse, kan den angives med miljøvariablen OLLAMA_URL.
