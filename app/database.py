import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_procedure_by_name(name):
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT id, base_price, bargain_min, bargain_max 
        FROM procedures 
        WHERE LOWER(name) = LOWER(%s)
    """, (name,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        return {
            "id": result[0],
            "base_price": float(result[1]),
            "bargain_min": float(result[2]),
            "bargain_max": float(result[3])
        }
    return None
