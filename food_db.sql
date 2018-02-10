
---To create a database, run the following command at the psql prompt
CREATE DATABASE food_db;
\connect food_db

CREATE TABLE recipes (
   id SERIAL PRIMARY KEY,
   post_date TIMESTAMP,
   title TEXT,
   foods TEXT);
