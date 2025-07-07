# Progetto Esame

Questo progetto contiene una simulazione di batteri con caratteristiche biologiche e comportamentali personalizzabili.

## Struttura del progetto

- `Bacteria.py`: contiene la definizione della classe `Bacteria`, che rappresenta un singolo batterio con tutte le sue proprietà.
- `Trait.py`: contiene le classi `Trait` e `Characteristic`, fornite dai professori, utilizzate per definire le proprietà e i comportamenti di ogni batterio.

## Creazione di un nuovo batterio

Per istanziare un nuovo oggetto `Bacteria`, sono necessari i seguenti parametri:

- `specie_name` (`str`): il nome della specie del batterio.
- `caratteristiche` (`List[Characteristic]`): una lista di caratteristiche biologiche.
- `tratti_scelti` (`Dict[str, Trait]`): un dizionario che associa ad ogni caratteristica un tratto specifico scelto per il batterio.
- `p_riproduzione` (`float`): probabilità di riproduzione ad ogni tentativo.
- `T` (`int`): numero di epoche necessarie prima di tentare una riproduzione.
- `consumo_ambiente` (`dict`): elementi dell'ambiente consumati dal batterio.
- `rilascio_ambiente` (`dict`): elementi rilasciati dal batterio nell'ambiente.

### Esempio di creazione di un batterio

```python
SpecieA = Bacteria(
    specie_name="Escherichia coli: Specie A",
    caratteristiche=[
        caratteristica_nitriti,
        caratteristica_ossigeno,
        caratteristica_carbonio,
        caratteristica_ossidazione_ammoniaca,
        caratteristica_trasformazione_pet
    ],
    tratti_scelti={
        "Nitriti in Nitrati": nitriti_in_nitrati_si,
        "Consumo di Ossigeno": consuma_ossigeno_basso,
        "Consumo di Carbonio": consuma_carbonio_no,
        "Ossidazione Ammoniaca": ossida_ammoniaca_no,
        "Trasformazione PET": trasforma_pet_no
    },
    p_riproduzione=0.7,
    T=10,
    consumo_ambiente=consuma_ambienteA,
    rilascio_ambiente=rilascio_ambienteA
)
````

## Metodi nel costruttore

* **Ambiente**: ogni batterio crea un ambiente casuale tramite il metodo `crea_ambiente()`, secondo regole predefinite.
* **Genoma**: ogni batterio costruisce un genoma basato sui tratti scelti, tramite il metodo `costruisci_genoma()`, con un controllo di validità (minimo 10 basi).
* **Controllo**: la validità delle caratteristiche e dei tratti viene verificata al momento della creazione grazie ai metodi `controllo_caratteristiche()` e `controllo_sequenze_uniche()`.

## Stampa del batterio

Stampando un oggetto `Bacteria`, si ottiene un output con informazioni dettagliate sul batterio, ad esempio:

```
Batterio Escherichia coli: Specie A con genoma di lunghezza 14 nucleotidi.

Tratti espressi:
  - Nitriti in Nitrati: Si
  - Consumo Ossigeno: Basso
  - Consuma Carbonio: No
  - Ossida Ammoniaca: No
  - Trasforma Pet: No

Probabilità di riproduzione: 0.7
Epoche per riprodursi: 10
Epoca attuale: 0
Epoca dall'ultima riproduzione: 0

Ambiente:
  - nitriti: 41
  - nitrati: 18
  - PET: 21
  - ossigeno: 99
  - carbonio: 76

Consumo dell'ambiente:
  - ossigeno: 4
  - nitriti: 2

Rilascio nell'ambiente:
  - nitrati: 3
```

## Simulazione del passaggio del tempo

È presente un metodo `aggiorna_epoca(n_epoche: int)` che simula il passaggio del tempo. Se è trascorso abbastanza tempo, il batterio proverà a riprodursi, tenendo conto della probabilità di successo.

### Esempio di utilizzo:

```python
SpecieA.aggiorna_epoca(5)
SpecieA.aggiorna_epoca(6)
```

### Output:

```
Passa il tempo...
Sono passate 5 epoche, ora siamo all'epoca 5. Il batterio Escherichia coli: Specie A non si è riprodotto.

Passa il tempo...
Durante l'epoca 10 il batterio si è riprodotto: Escherichia coli: Specie A.
Sono passate 6 epoche, ora siamo all'epoca 11.
```

