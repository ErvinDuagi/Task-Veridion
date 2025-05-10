
# Veridion ML Insurance Classifier

Acest proiect reprezintă soluția mea pentru challenge-ul tehnic Veridion – task-ul #2: clasificarea companiilor în funcție de o taxonomie din industria asigurărilor.

## 🔍 Scopul proiectului
Clasificarea automată a companiilor într-una sau mai multe etichete relevante dintr-o taxonomie statică. Pe baza descrierii, tag-urilor și clasificărilor oferite, modelul învață să mapeze fiecare companie la nișe economice specializate.
Trebuie să dezvolt un model de clasificare care, pornind de la datele despre companii (descriere, taguri, sector, categorie, nișă), să atribuie una sau mai multe etichete relevante dintr-o taxonomie fixă, specifică domeniului de asigurări.

**IMPORTANT** !!! 

****Aceasta este doar o schita a proietului, pentru a intelege in totalitate cum am gandit si cum am incercat sa rezolv accest task va rog sa va uitati peste documentatie.**
**
## 🗂 Structura proiectului
```
veridion_project/
├── data/
│   ├── ml_insurance_challenge.csv           # setul de date original
│   ├── insurance_taxonomy.xlsx              # taxonomia fixă (etichetare)
│   └── processed_data.csv                   # datele preprocesate
├── notebooks/
│   ├── 00_eda_raw.ipynb                     # EDA pe date brute
│   └── 01_eda_explorare.ipynb               # EDA pe date prelucrate
├── src/
│   ├── data_loader.py                       # script de preprocesare
│   ├── model_training.py                    # antrenarea modelului (urmează)
│   └── utils.py                             # funcții auxiliare (dacă e nevoie)
└── README.md
```

## ⚙️ Stack tehnologic
- Python (pandas, scikit-learn)
- Jupyter Notebooks
- VSCode 
- CSV + Excel (.xlsx)

## 🔧 Pași parcurși
**1. Explorare inițială a datelor brute**

- `ml_insurance_challenge.csv` — 9494 companii cu:
  - `description` (text liber),
  - `business_tags` (cuvinte-cheie),
  - `sector`, `category`, `niche`.
- `insurance_taxonomy.xlsx` — 220 etichete specializate care definesc taxonomia de clasificare.
Deși sector, category și niche sunt informative si usor de inteles pentru un om, din perspectivă de ML acestea au o cardinalitate mare și oferă context limitat față de description și tags, care conțin semnale semantice mai bogate. 
**2. Preprocesare textuală a coloanelor:**
  Realizată prin `data_loader.py`. Pașii principali:

- curățarea textelor din `description`, `tags`, `category`, `niche`;
- conversia tag-urilor în șiruri utilizabile (list → string);
- completarea valorilor lipsă;
- crearea unei coloane `full_profile` = input standardizat pentru clasificare.

description_clean: textul descriptiv este convertit la lowercase, golurile sunt completate cu șiruri goale, iar caracterele speciale sunt eliminate pentru a-l pregăti pentru vectorizare (TF-IDF, embeddings etc.).

tags_clean: lista de business_tags este convertită din format JSON într-un șir unificat de cuvinte-cheie relevante, cu spațiile înlocuite de _, pentru a fi compatibile cu modelele de NLP.

full_profile: câmp compus care concatenează description_clean, tags_clean, category și niche — oferind un rezumat semantic complet per companie, utilizat drept input principal în clasificare.

****3. Explorare date noi (EDA) ****
    Salvare fișier procesat
    EDA pe datele prelucrate (distribuții, lipsuri, patternuri) realizata in jupyter notebook
    Detalii în:
- `00_eda_raw.ipynb`: analiza inițială pe datele brute.
- `01_eda_explorare.ipynb`: analiză pe `processed_data.csv`.
**4. Crearea setului de antrenare**
În această etapă, am generat automat etichete de antrenament pentru fiecare companie prin măsurarea similarității semantice între profilul textual (full_profile) și descrierile din taxonomia de etichete. Folosind TF-IDF și cosine similarity, am atribuit fiecărei companii cele mai apropiate 3 etichete, salvate într-un fișier annotated_training_set.csv ce va fi folosit pentru antrenarea modelului.

Inițial, am explorat o abordare semi-automată de adnotare a unui subset de date folosind un Jupyter Notebook (02_annotation_tool.ipynb). Această metodă presupunea generarea unor sugestii simple pe baza unor reguli logice și completarea manuală a etichetelor pentru fiecare companie. Cu toate acestea, metoda s-a dovedit a fi incompletă și dificil de scalat, necesitând un efort substanțial pentru adnotare manuală și neacoperind suficientă diversitate semantică.

Din acest motiv, am optat ulterior pentru o abordare complet automată, mai robustă și scalabilă, folosind tehnici de procesare a limbajului natural (TF-IDF + cosine similarity). Această metodă permite etichetarea întregului set de companii, bazându-se pe gradul de asemănare semantică dintre profilul textual al companiei și descrierile etichetelor din taxonomie. Astfel, am obținut un set adnotat realist, coerent și pregătit pentru antrenarea unui model de clasificare multi-label.
Pașii parcurși in aceasta etapa:
1.	Crearea unui fișier de taxonomie extinsă (insurance_taxonomy_enhanced.csv)
Fișierul original (insurance_taxonomy.csv) conținea doar o listă de etichete (label). Am generat automat o coloană description pentru fiecare etichetă, oferind context semantic suplimentar. Aceste descrieri au fost create cu ajutorul unui script Python care a aplicat reguli semantice simple pe baza conținutului textual al fiecărui label.
2.	Vectorizarea textelor
Am aplicat TF-IDF vectorization atât asupra câmpului full_profile din fișierul cu companii (processed_data.csv), cât și asupra descrierilor din taxonomie.
3.	Calculul similarității
Am folosit cosine similarity pentru a măsura asemănarea semantică dintre fiecare companie și toate descrierile etichetelor.
4.	Selectarea etichetelor cele mai probabile
Pentru fiecare companie, au fost selectate cele mai apropiate 3 etichete (cu scoruri de similaritate cele mai mari) și salvate într-o coloană selected_labels.
5.	Generarea fișierului annotated_training_set.csv
Acest fișier conține câmpurile full_profile și selected_labels și va fi folosit în etapa următoare pentru antrenarea modelului de clasificare.

**5. Construirea și antrenarea modelului**
După generarea fișierului annotated_training_set.csv, am constatat că datele brute conțineau peste 200 de etichete, dintre care multe erau extrem de rare. Acest lucru a dus la rezultate slabe pentru Logistic Regression și Random Forest, cu valori 0 la precision și recall pentru multe etichete. Pentru a îmbunătăți performanța, am decis să păstrăm doar cele mai frecvente 50 de etichete și să reconstruim setul de antrenare (annotated_training_set_filtered-regresie.csv).

Pe baza acestuia, am testat trei clasificatori multi-label: Logistic Regression, LinearSVC și Random Forest. Am vectorizat textul folosind TF-IDF și am binarizat etichetele cu MultiLabelBinarizer. Rezultatele obținute au fost:

LinearSVC: cele mai bune scoruri generale (F1-weighted: 0.53)

Random Forest: performanță rezonabilă după filtrarea etichetelor (F1-weighted: 0.50)

Logistic Regression: rezultate acceptabile, folosit ca variantă de backup (F1-weighted: 0.52)

Am păstrat LinearSVC ca model principal pentru testare ulterioară, datorită echilibrului bun între precizie și acoperire, fără ajustări suplimentare.

**6. Testarea modelelor și concluzii finale**
În etapa finală, modelele antrenate anterior au fost aplicate pe întreg setul de 9494 companii pentru a prezice una sau mai multe etichete de asigurări.

Linear SVC a fost singurul model antrenat pe toate cele 220 de etichete și este considerat modelul principal al soluției.

Logistic Regression și Random Forest au fost antrenate doar pe top 50 cele mai frecvente etichete și au fost utilizate în scop comparativ.

Rezultate:
Linear SVC a generat predicții complete pentru toate companiile, cu o estimare de 56.1% predicții acceptabile (conform analizei manuale și comparative).

Logistic Regression: 4488 predicții non-goale (acoperire 47.2%)

Random Forest: 1747 predicții non-goale (acoperire 18.4%)

Suprapunere parțială între modele:

987 companii cu etichete comune între Linear SVC și Random Forest

683 între Linear SVC și Logistic Regression

Linear SVC rămâne cea mai robustă și completă soluție, capabilă să generalizeze eficient pe tot setul de date.

**6. Posibiile Imbunatatiri**
Sugestii de îmbunătățire
Trecerea de la TF-IDF la BERT: Modelele Transformer precum BERT pot înțelege mai bine contextul semantic al descrierilor companiilor. Folosirea unui model precum all-MiniLM-L6-v2 ar putea crește considerabil acuratețea predicțiilor.

Praguri adaptative pentru etichete: În loc de un scor fix (ex: 0.5) pentru toate clasele, se pot optimiza praguri individuale pentru fiecare etichetă, crescând precizia și reducând etichetele eronate.

Etichetare umană pe subset: Adnotarea manuală a unui eșantion de companii ar permite o evaluare mai realistă a modelelor și îmbunătățirea ulterioară a setului de antrenament.

Scalabilitate cu Apache Spark: În cazul unor seturi de date la scară industrială (milioane de companii), o implementare distribuită cu Apache Spark ar permite procesarea paralelă a datelor text, antrenarea mai rapidă a modelelor și aplicarea scalabilă a predicțiilor.

## 🧠 Observații
- Etichetarea nu are un ground truth clar → validare prin analiză logică
 Etichetele sunt foarte specializate (ex: agricultură, mediu, servicii).
- Task-ul nu e o clasificare simplă (Life vs. P&C), ci un mapping către nișe complexe.
- Provocarea constă în extragerea de semnale relevante din descrieri + taguri.
- Se acceptă clasificare multi-label (compania poate primi mai multe etichete)
- Soluția trebuie să fie clar explicată și scalabilă conceptual

## 📌 Status: în desfășurare
