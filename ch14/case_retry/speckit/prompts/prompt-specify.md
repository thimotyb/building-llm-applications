# Prompt per `specify`

Usa gli input in `inputs/` come fonti del caso checkout e autorizzazione del pagamento. Definisci **cosa** deve fare il flusso di retry e perché, senza proporre stack, classi o struttura dei file.

Includi attori, ambito, requisiti funzionali numerati e scenari di accettazione osservabili. Distingui esplicitamente ciò che è supportato dai requisiti e dai diagrammi dalle decisioni ancora aperte. Non dedurre una semantica di idempotenza dal solo nome `Idempotency-Key`; non definire il comportamento della ripetizione dell'intero checkout se gli input non lo specificano.

Limita la feature al retry dell'autorizzazione di pagamento per errori transitori e alla comunicazione dell'esito. Non aggiungere pagamenti reali, autenticazione, catalogo, spedizione o persistenza.
