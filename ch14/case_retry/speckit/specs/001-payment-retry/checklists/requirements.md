# Checklist qualità dei requisiti

Questa checklist valuta chiarezza e copertura della specifica; non certifica il completamento dell'implementazione.

- [ ] Ogni requisito è riconducibile a un file o a una decisione approvata.
- [ ] Gli scenari hanno precondizioni, risultato osservabile e limite di tentativi.
- [ ] Il successo interrompe esplicitamente la sequenza di retry.
- [ ] Il fallimento dopo tre errori non crea un ordine.
- [ ] Gli esiti non transitori non ricevono semantiche inventate.
- [ ] La chiave di idempotenza dell'API non è confusa con l'idempotenza dell'intero checkout.
- [ ] Le domande aperte sono visibili e i task non inventano risposte al loro posto.
- [ ] Le verifiche sono eseguibili con gateway sintetico e risultati deterministici.
