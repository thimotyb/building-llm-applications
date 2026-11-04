# Piano tecnico — retry dell'autorizzazione

## Contesto tecnico

Il prototipo M01 fornito come `inputs/prototype-demo.py` è uno script Python con `FakeGateway` e scenari deterministici. `inputs/req-seq-payment-retry.puml` descrive il retry dell'autorizzazione fino a tre tentativi; `inputs/req-apis.yaml` descrive `POST /payments` e il parametro `Idempotency-Key`.

## Approccio

1. Mantenere la simulazione locale e il gateway fittizio.
2. Rendere configurabili gli esiti del gateway per osservare errore transitorio, approvazione e sequenze di fallimento.
3. Separare il ciclo di retry dall'assemblaggio dell'ordine quanto basta per verificare numero di tentativi, arresto all'approvazione e assenza di ordine in caso di esaurimento.
4. Aggiungere test deterministici per gli scenari AC-1, AC-2 e AC-3.
5. Mantenere invariati gli scenari di regressione di M01 e documentare il rapporto con `Idempotency-Key` senza inventare semantiche non presenti nel contratto.

## Vincoli

- Runtime del prototipo: Python 3.10 o successivo, libreria standard.
- Nessun servizio esterno, database o dato reale.
- Il limite di tre tentativi deriva dalla sequenza del caso guida.
- Backoff, timeout, forma dell'esito API e gestione dei rifiuti non transitori richiedono decisioni prima di diventare comportamento implementato.

## Sequenza di lavoro

1. Risolvere le decisioni bloccanti in [`decisions.md`](decisions.md).
2. Confermare con il team i tre scenari di accettazione già supportati dalle fonti.
3. Aggiungere test che osservino chiamate al gateway e ordini creati.
4. Implementare solo il ciclo di retry approvato, mantenendo le fixture deterministiche.
5. Eseguire i test della mini demo; rivedere diff e corrispondenza con la specifica.

## Rischi da verificare

- La specifica di `POST /payments` suggerisce l'header di idempotenza per le richieste ripetute, ma non descrive durata, scope o replay della risposta.
- La retry policy della chiamata di autorizzazione non rende automaticamente idempotente una nuova richiesta di checkout.
- L'assenza di una decisione sul timeout del gateway può lasciare incerto l'esito di un'autorizzazione.
