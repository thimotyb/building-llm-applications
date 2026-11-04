# Decisioni aperte del team

Queste domande emergono confrontando requisiti, diagrammi, contratto API e prototipo. Non sono risolte dalle fonti del caso guida.

| ID | Domanda | Impatto | Stato |
| --- | --- | --- | --- |
| D-1 | Chi genera `Idempotency-Key`, qual è il suo scope e per quanto tempo resta valida? | Contratto, persistenza e test di retry API | Da decidere |
| D-2 | Cosa deve fare `POST /payments` quando la stessa chiave arriva con un payload diverso? | Risposta API e prevenzione di autorizzazioni incoerenti | Da decidere |
| D-3 | Come distinguere errori transitori, rifiuti definitivi e timeout dall'esito sconosciuto? | Classificazione e retry sicuri | Da decidere |
| D-4 | Sono previsti attesa progressiva e timeout tra tentativi? | Latenza e carico sul gateway | Da decidere |
| D-5 | Come riconoscere un retry dello stesso checkout e garantire un solo ordine? | Deduplicazione del flusso completo | Fuori ambito di questa feature; serve una decisione separata |
| D-6 | Quale formato e codice restituisce l'API quando tutti i tentativi falliscono? | Contratto API e messaggio al cliente | Da decidere |

Prima di implementare una decisione, registrare risposta, responsabile e test di accettazione collegati. Le assunzioni temporanee vanno etichettate e non promosse a requisiti senza approvazione.
