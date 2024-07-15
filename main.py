import sqlite3
 
try:
   
    # Connect to DB and create a cursor
    sqliteConnection = sqlite3.connect('../health/mimiciii/mimic3.db')
    cursor = sqliteConnection.cursor()
    # print('DB Init')
 
    # Write a query and execute it with cursor
    query = 'SELECT TEXT \
            FROM NOTEEVENTS \
            ORDER BY ROW_ID \
            LIMIT 1 \
            OFFSET 1; '
    cursor.execute(query)
 
    # Fetch and output result
    result = cursor.fetchall()[0][0]
    print(result)
 
    # Close the cursor
    cursor.close()
 
# Handle errors
except sqlite3.Error as error:
    print('Error occurred - ', error)