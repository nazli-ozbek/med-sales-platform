-- Doktorlar tablosu
CREATE TABLE IF NOT EXISTS doctors (
    id SERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    specialization TEXT,
    languages TEXT[],
    bio TEXT
);

-- Prosedürler tablosu
CREATE TABLE IF NOT EXISTS procedures (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    base_price NUMERIC(10,2)
);

-- Doktor-prosedür ilişkisi ve fiyatlandırma tablosu
CREATE TABLE IF NOT EXISTS doctor_procedures (
    doctor_id INT REFERENCES doctors(id) ON DELETE CASCADE,
    procedure_id INT REFERENCES procedures(id) ON DELETE CASCADE,
    custom_price NUMERIC(10,2),
    bargain_min NUMERIC(10,2),
    bargain_max NUMERIC(10,2),
    PRIMARY KEY (doctor_id, procedure_id)
);