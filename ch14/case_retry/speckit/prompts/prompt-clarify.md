# Prompt per `clarify`

Concentrati sui comportamenti che impediscono di verificare i tre scenari già descritti: approvazione al primo tentativo, errore transitorio seguito da approvazione e tre errori transitori consecutivi.

Per la mini demo assumi solo il confine esplicito dello scope: autorizzazione simulata, un massimo di tre tentativi, arresto al primo esito approvato e messaggio di alternativa dopo tre errori transitori. Non definire semantiche di `Idempotency-Key`, timeout, backoff, rifiuti definitivi o ripetizione dell'intero checkout. Riporta queste domande in `decisions.md` come fuori ambito e non bloccare i task che verificano i tre scenari.
