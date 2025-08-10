import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

def create_sample_database(db_path: str = "sample_store.db"):
    """Create a sample SQLite database with e-commerce data"""
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        first_name VARCHAR(50) NOT NULL,
        last_name VARCHAR(50) NOT NULL,
        registration_date DATE NOT NULL,
        is_active BOOLEAN DEFAULT 1
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name VARCHAR(100) NOT NULL,
        description TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name VARCHAR(200) NOT NULL,
        description TEXT,
        price DECIMAL(10, 2) NOT NULL,
        category_id INTEGER,
        stock_quantity INTEGER DEFAULT 0,
        created_date DATE NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        order_date DATE NOT NULL,
        total_amount DECIMAL(10, 2) NOT NULL,
        status VARCHAR(20) DEFAULT 'pending',
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (order_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Sample data
    categories_data = [
        ("Electronics", "Electronic devices and gadgets"),
        ("Books", "Books and educational materials"),
        ("Clothing", "Apparel and fashion items"),
        ("Home & Garden", "Home improvement and gardening supplies"),
        ("Sports", "Sports equipment and accessories")
    ]
    
    users_data = [
        ("john_doe", "john@email.com", "John", "Doe", "2023-01-15"),
        ("jane_smith", "jane@email.com", "Jane", "Smith", "2023-02-20"),
        ("bob_wilson", "bob@email.com", "Bob", "Wilson", "2023-03-10"),
        ("alice_brown", "alice@email.com", "Alice", "Brown", "2023-04-05"),
        ("charlie_davis", "charlie@email.com", "Charlie", "Davis", "2023-05-12")
    ]
    
    products_data = [
        ("Smartphone", "Latest Android smartphone with 128GB storage", 599.99, 1, 50, "2023-01-01"),
        ("Laptop", "High-performance laptop for work and gaming", 1299.99, 1, 25, "2023-01-01"),
        ("Python Programming Book", "Complete guide to Python programming", 49.99, 2, 100, "2023-01-01"),
        ("T-Shirt", "Cotton t-shirt available in multiple colors", 19.99, 3, 200, "2023-01-01"),
        ("Garden Tools Set", "Complete set of gardening tools", 89.99, 4, 30, "2023-01-01"),
        ("Basketball", "Official size basketball", 29.99, 5, 75, "2023-01-01"),
        ("Headphones", "Wireless noise-canceling headphones", 199.99, 1, 40, "2023-01-01"),
        ("Cookbook", "Healthy recipes for everyday cooking", 24.99, 2, 60, "2023-01-01")
    ]
    
    # Insert sample data
    cursor.executemany("INSERT INTO categories (category_name, description) VALUES (?, ?)", categories_data)
    cursor.executemany("INSERT INTO users (username, email, first_name, last_name, registration_date) VALUES (?, ?, ?, ?, ?)", users_data)
    cursor.executemany("INSERT INTO products (product_name, description, price, category_id, stock_quantity, created_date) VALUES (?, ?, ?, ?, ?, ?)", products_data)
    
    # Generate sample orders
    orders_data = []
    order_items_data = []
    order_id = 1
    
    for i in range(20):  # 20 sample orders
        user_id = random.randint(1, 5)
        order_date = datetime.now() - timedelta(days=random.randint(1, 365))
        status = random.choice(['pending', 'completed', 'shipped', 'cancelled'])
        
        # Generate 1-3 items per order
        num_items = random.randint(1, 3)
        total_amount = 0
        
        for j in range(num_items):
            product_id = random.randint(1, 8)
            quantity = random.randint(1, 3)
            
            # Get product price
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
            unit_price = cursor.fetchone()[0]
            
            order_items_data.append((order_id, product_id, quantity, unit_price))
            total_amount += quantity * unit_price
        
        orders_data.append((user_id, order_date.strftime('%Y-%m-%d'), total_amount, status))
        order_id += 1
    
    cursor.executemany("INSERT INTO orders (user_id, order_date, total_amount, status) VALUES (?, ?, ?, ?)", orders_data)
    cursor.executemany("INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)", order_items_data)
    
    conn.commit()
    conn.close()
    
    print(f"Sample database created successfully at {db_path}")

if __name__ == "__main__":
    create_sample_database()