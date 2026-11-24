import os

import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error

load_dotenv()


def get_db_config(include_database=True):
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
    }
    if include_database:
        config["database"] = os.getenv("DB_NAME", "agrisathi_db")
    return config

def get_db_connection():
    try:
        connection = mysql.connector.connect(**get_db_config())
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def create_tables(connection):
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            mobile VARCHAR(15) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM('farmer','admin') DEFAULT 'farmer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS farmer_profile (
            profile_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            age INT,
            land_size FLOAT,
            land_unit ENUM('acre','hectare'),
            state VARCHAR(50),
            district VARCHAR(50),
            village VARCHAR(50),
            language VARCHAR(30),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crops (
            crop_id INT AUTO_INCREMENT PRIMARY KEY,
            crop_name VARCHAR(100) NOT NULL,
            season ENUM('Kharif','Rabi','Zaid'),
            description TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS farmer_crops (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            crop_id INT NOT NULL,
            sowing_date DATE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (crop_id) REFERENCES crops(crop_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_images (
            image_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            crop_id INT NOT NULL,
            image_path VARCHAR(255),
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (crop_id) REFERENCES crops(crop_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_results (
            result_id INT AUTO_INCREMENT PRIMARY KEY,
            image_id INT NOT NULL,
            disease_name VARCHAR(100),
            confidence FLOAT,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES disease_images(image_id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS remedies (
            remedy_id INT AUTO_INCREMENT PRIMARY KEY,
            disease_name VARCHAR(100),
            remedy_type ENUM('organic','chemical'),
            description TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS govt_schemes (
            scheme_id INT AUTO_INCREMENT PRIMARY KEY,
            scheme_name VARCHAR(150),
            description TEXT,
            min_age INT,
            max_age INT,
            min_land FLOAT,
            max_land FLOAT,
            state VARCHAR(50)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scheme_applications (
            application_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            scheme_id INT,
            status ENUM('eligible','not eligible','applied'),
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (scheme_id) REFERENCES govt_schemes(scheme_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS msp_data (
            msp_id INT AUTO_INCREMENT PRIMARY KEY,
            crop_id INT,
            state VARCHAR(50),
            year INT,
            price_per_quintal FLOAT,
            FOREIGN KEY (crop_id) REFERENCES crops(crop_id)
        )
    ''')

    cursor.close()

def seed_initial_data(connection):
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM crops")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO crops (crop_name, season) VALUES (%s, %s)",
            [
                ('Wheat', 'Rabi'),
                ('Rice', 'Kharif'),
                ('Maize', 'Kharif'),
                ('Mustard', 'Rabi'),
            ],
        )

    cursor.execute("SELECT COUNT(*) FROM remedies")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO remedies (disease_name, remedy_type, description) VALUES (%s, %s, %s)",
            [
                ('Leaf Blight', 'organic', 'Neem oil spray every 7 days'),
                ('Leaf Blight', 'chemical', 'Apply recommended fungicide'),
                ('Powdery Mildew', 'organic', 'Baking soda solution'),
                ('Powdery Mildew', 'chemical', 'Sulphur-based fungicide'),
            ],
        )

    cursor.execute("SELECT COUNT(*) FROM govt_schemes")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            """
            INSERT INTO govt_schemes
                (scheme_name, description, min_age, max_age, min_land, max_land, state)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                ('PM-KISAN', 'Income support for farmers', 18, 100, 0, 5, 'ALL'),
                ('Fasal Bima Yojana', 'Crop insurance scheme', 18, 100, 0, 100, 'ALL'),
            ],
        )

    cursor.close()

def init_db():
    try:
        connection = mysql.connector.connect(**get_db_config(include_database=False))
        cursor = connection.cursor()
        db_name = os.getenv("DB_NAME", "agrisathi_db")
        
        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        cursor.execute(f"USE `{db_name}`")
        cursor.close()

        create_tables(connection)
        seed_initial_data(connection)

        connection.commit()
        print("Database initialized successfully")

    except Error as e:
        print(f"Error initializing database: {e}")
        if 'connection' in locals() and connection.is_connected():
            connection.rollback()

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("MySQL connection is closed")
        print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()
