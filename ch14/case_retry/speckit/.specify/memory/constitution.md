# Principi del progetto demo

## Evidenza prima delle inferenze

Requisiti, sequenze e contratto API forniti in `inputs/` sono le fonti del comportamento; in questa demo sono copie temporanee dei file versionati in `caso-guida/`. Ogni requisito generato deve poter essere ricondotto a una fonte o essere marcato come proposta da approvare.

## Decisioni visibili

Non trasformare ambiguità in comportamento implicito. Registrare le decisioni aperte e fermare i task che dipendono da esse finché il team non le approva.

## Confini della demo

Usare solo gateway e dati sintetici. La demo non chiama servizi di pagamento, non usa credenziali e non tratta dati reali.

## Verifica tracciabile

Ogni requisito funzionale ha almeno uno scenario di accettazione. Ogni task di implementazione rimanda a uno o più requisiti e indica come verificarli.

## Revisione umana

Specifiche, piano e task sono proposte da revisionare. Nessun output dell'agente autorizza da solo modifiche al codice o scelte di prodotto.
