from dohvacanje_zg_burza_podataka import *

print(f'Zadnji datum iz baze: {get_last_date_from_db()[:10]}')
fill_database(start_time=get_last_date_from_db()[:10])
print(f'Izvlačenje završeno, novi zadnji datum iz baze: {get_last_date_from_db()[:10]}')