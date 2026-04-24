import sqlite3
import os
from pathlib import Path

def setup_hotel_db():
    """
    Inizializza il database SQLite per le prenotazioni degli hotel in Cornwall.
    Legge lo schema e i dati dal file SQL e crea il file .db necessario per gli agenti.
    """
    # 1. Definiamo i percorsi
    current_dir = Path(__file__).parent
    sql_file = current_dir / "hotel_db" / "cornwall_hotels_schema.sql"
    db_file = current_dir / "hotel_db" / "cornwall_hotels.db"

    # 2. Verifichiamo se il file SQL esiste
    if not sql_file.exists():
        print(f"❌ Errore: File SQL non trovato in {sql_file}")
        return

    # 3. Creiamo la directory hotel_db se non esiste
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # 4. Inizializziamo il database
    print(f"🏨 Inizializzazione database hotel in corso...")
    
    try:
        # Se il database esiste già, lo resettiamo per pulizia (opzionale)
        if db_file.exists():
            print(f"♻️  Database esistente trovato. Aggiornamento in corso...")
        
        conn = sqlite3.connect(db_file)
        
        # Pulizia: eliminiamo le tabelle se esistono per ripartire da zero
        conn.execute("DROP TABLE IF EXISTS hotel_room_offers")
        conn.execute("DROP TABLE IF EXISTS hotels")
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # Eseguiamo lo script SQL (Schema + Dati)
        conn.executescript(sql_script)
        conn.commit()
        conn.close()
        
        print(f"✅ Database creato con successo: {db_file}")
        print(f"📊 Tabelle create: hotels, hotel_room_offers")
        
    except Exception as e:
        print(f"❌ Errore durante la creazione del database: {e}")

if __name__ == "__main__":
    setup_hotel_db()

# =============================================================================
# ISTRUZIONI D'USO
# =============================================================================
# Questo script prepara il database SQL necessario per gli agenti del Capitolo 11
# che gestiscono le prenotazioni (Booking Agent).
#
# Per usarlo:
# 1. Assicurati di essere nella cartella 'ch11'
# 2. Lancia lo script:
#    ./env_ch11/bin/python setup_hotel_db.py
#
# Dopo l'esecuzione, gli script main_04_01.py e successivi potranno 
# interrogare il database degli hotel della Cornovaglia.
# =============================================================================
