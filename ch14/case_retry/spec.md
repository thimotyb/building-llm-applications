# Feature: retry dell'autorizzazione di pagamento

## Obiettivo

Consentire al checkout di recuperare da errori transitori del gateway simulato, rispettando il limite di tentativi descritto nel caso guida e comunicando un esito comprensibile.

## Ambito

Questa feature riguarda il ciclo di autorizzazione del pagamento nel checkout. Il gateway è simulato. Catalogo, login, spedizione, persistenza e integrazione con un provider reale sono fuori ambito.

La ripetizione dell'intero checkout e la creazione di un solo ordine per richieste duplicate restano questioni di prodotto separate; i documenti del caso guida non definiscono una regola completa.

## Attori

- **Cliente:** avvia il pagamento e riceve l'esito.
- **Applicazione web:** richiede l'autorizzazione al gateway e applica la sequenza di retry.
- **Gateway simulato:** restituisce approvazione o errore transitorio secondo una fixture deterministica.

## Requisiti funzionali

- **FR-1 — Retry transitorio:** se il gateway restituisce un errore transitorio, l'applicazione riprova l'autorizzazione, fino a un massimo di tre tentativi complessivi.
- **FR-2 — Arresto all'approvazione:** se un tentativo viene approvato, l'applicazione interrompe il ciclo e prosegue il checkout con il `paymentId` restituito.
- **FR-3 — Esaurimento dei tentativi:** se tutti e tre i tentativi terminano con errore transitorio, l'applicazione informa il cliente del fallimento e suggerisce un metodo di pagamento alternativo.

## Nota sul contratto API

Il contratto OpenAPI dichiara `Idempotency-Key` come header facoltativo di `POST /payments` e consiglia al client di includerlo nei retry. Il contratto ne dichiara l'intento idempotente, ma non precisa generazione, scope, durata o comportamento per payload diversi; non definisce quindi l'idempotenza dell'intero checkout.

## Esiti non descritti

Il caso guida non specifica come classificare e trattare ogni possibile rifiuto non transitorio. Non aggiungere retry o risposte per questi casi senza una decisione approvata.

## Scenari di accettazione

### AC-1 — Approvazione al primo tentativo

**Dato** un gateway che approva il primo tentativo, **quando** il cliente avvia il checkout, **allora** l'applicazione interrompe il retry, riceve un `paymentId` e prosegue la sequenza di checkout.

### AC-2 — Errore transitorio seguito da approvazione

**Dato** un gateway che restituisce un errore transitorio e poi un'approvazione, **quando** il checkout richiede l'autorizzazione, **allora** l'applicazione effettua un secondo tentativo, si ferma dopo l'approvazione e prosegue con il `paymentId` ricevuto.

### AC-3 — Tre errori transitori

**Dato** un gateway che restituisce tre errori transitori, **quando** il checkout richiede l'autorizzazione, **allora** l'applicazione effettua esattamente tre tentativi, non procede alla creazione dell'ordine e informa il cliente suggerendo un'alternativa.

## Criteri di esclusione

- Nessun quarto tentativo.
- Nessun nuovo ordine se il pagamento non è stato approvato.
- Nessuna chiamata a gateway reali.
- Nessuna regola implicita che renda idempotente l'intero checkout.

## Decisioni aperte

Vedere [`decisions.md`](decisions.md). Il lavoro che dipende da tali risposte non è pronto per l'implementazione.
