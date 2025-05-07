import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def get_procedure_by_name(name):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, base_price, bargain_min, bargain_max
        FROM procedures
        LEFT JOIN doctor_procedures ON procedures.id = doctor_procedures.procedure_id
        WHERE LOWER(procedures.name) = LOWER(%s)
        LIMIT 1
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


def get_doctors_by_procedure(procedure_name):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT d.id, d.full_name, d.specialization, dp.custom_price, dp.bargain_min, dp.bargain_max
        FROM doctors d
        JOIN doctor_procedures dp ON d.id = dp.doctor_id
        JOIN procedures p ON dp.procedure_id = p.id
        WHERE LOWER(p.name) = LOWER(%s)
    """, (procedure_name,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    doctors = []
    for row in rows:
        doctors.append({
            "id": row[0],
            "name": row[1],
            "specialization": row[2],
            "custom_price": float(row[3]),
            "bargain_min": float(row[4]),
            "bargain_max": float(row[5])
        })
    return doctors

if __name__ == "__main__":
    print(get_procedure_by_name("rhinoplasty"))
    print(get_doctors_by_procedure("rhinoplasty"))