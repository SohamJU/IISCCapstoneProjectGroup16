import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

password = os.getenv("POSTGRESQL_AIVEN_PASSWORD")
host = "pg-20bad560-myproject123-456.e.aivencloud.com"
port = "12548"
uri = f"postgresql://avnadmin:{password}@{host}:{port}/defaultdb?sslmode=require"

try:
    conn = psycopg2.connect(uri)
    print("Success pg-20bad")
except Exception as e:
    print(f"Failed pg-20bad: {e}")

host2 = "pg-34e61100-pkscurious-bb78.l.aivencloud.com"
port2 = "20868"
uri2 = f"postgresql://avnadmin:{password}@{host2}:{port2}/defaultdb?sslmode=require"
try:
    conn = psycopg2.connect(uri2)
    print("Success pg-34e")
except Exception as e:
    print(f"Failed pg-34e: {e}")
