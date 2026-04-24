import sqlite3
from pathlib import Path

def inspect_hotels():
    """
    Legge e stampa il contenuto del database degli hotel.
    """
    db_path = Path(__file__).parent / "hotel_db" / "cornwall_hotels.db"

    if not db_path.exists():
        print(f"❌ Errore: Il database non esiste in {db_path}")
        print("Esegui prima: python setup_hotel_db.py")
        return

    print(f"🔍 Ispezione del database: {db_path}\n")

    try:
        # Usiamo Row per accedere alle colonne per nome
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. Otteniamo la lista delle tabelle
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row['name'] for row in cursor.fetchall()]

        for table in tables:
            print(f"📋 TABELLA: {table}")
            print("=" * 80)
            
            # 2. Otteniamo i dati della tabella
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()

            if not rows:
                print("(Tabella vuota)")
            else:
                # 3. Stampiamo l'intestazione
                headers = rows[0].keys()
                header_line = " | ".join(f"{h:^15}" for h in headers)
                print(header_line)
                print("-" * len(header_line))

                # 4. Stampiamo le righe
                for row in rows:
                    line = " | ".join(f"{str(row[h])[:15]:<15}" for h in headers)
                    print(line)
            
            print("=" * 80 + "\n")

        conn.close()

    except Exception as e:
        print(f"❌ Errore durante l'ispezione: {e}")

if __name__ == "__main__":
    inspect_hotels()

# =============================================================================
# ISTRUZIONI D'USO
# =============================================================================
# Lancia lo script per vedere rapidamente cosa c'è nel database degli hotel:
#
# ./env_ch11/bin/python inspect_hotel_db.py
# =============================================================================
