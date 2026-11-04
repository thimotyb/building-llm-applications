# Task — retry dell'autorizzazione

## Gate preliminare

- [ ] **T-001 — Approvare i criteri di accettazione AC-1, AC-2 e AC-3.** Verificare che ogni risultato sia coerente con requisiti e diagrammi.
- [ ] **T-002 — Delimitare il lavoro implementabile.** Confermare che questa demo implementa solo retry di errori transitori, arresto all'approvazione e comunicazione dopo tre fallimenti; le decisioni elencate in `decisions.md` non devono essere codificate.

## Implementazione dopo il gate

- [ ] **T-003 — Preparare fixture di esito del gateway.** Supportare sequenze riproducibili per approvazione immediata, errore transitorio seguito da approvazione e tre errori consecutivi. Verifica: la fixture registra ogni chiamata.
- [ ] **T-004 — Implementare il limite di tre tentativi.** Collegato a FR-1 e AC-2/AC-3. Verifica: non avvengono più di tre chiamate.
- [ ] **T-005 — Fermare il ciclo all'approvazione.** Collegato a FR-2 e AC-1/AC-2. Verifica: nessun tentativo ulteriore dopo `paymentId`.
- [ ] **T-006 — Gestire l'esaurimento dei tentativi.** Collegato a FR-3 e AC-3. Verifica: il checkout non crea l'ordine e l'esito suggerisce un'alternativa.
- [ ] **T-007 — Aggiornare la documentazione della demo.** Riportare requisiti, limiti e scelte approvate. Verifica: i testi non descrivono come definita una decisione ancora aperta.
- [ ] **T-008 — Eseguire i test della mini demo.** Lanciare `python3 -m unittest -v`. Verifica: tutti e tre gli scenari di accettazione passano.

## Fuori dal piano approvabile

Non implementare la deduplicazione dell'intero checkout, il riuso di una chiave con payload diverso, timeout/backoff o retry per rifiuti non transitori. Questi aspetti richiedono decisioni separate e non fanno parte dei task di questa mini demo.
