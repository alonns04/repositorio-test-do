# gunicorn_config.py

# Dirección y puerto donde escucha gunicorn
bind = "0.0.0.0:8080"

# Número de workers para concurrencia
workers = 2

# Timeout máximo en segundos (5 minutos)
timeout = 300

# Directorio temporal en memoria para mejorar rendimiento y evitar errores
worker_tmp_dir = "/dev/shm"

# Nivel de logs
loglevel = "info"
